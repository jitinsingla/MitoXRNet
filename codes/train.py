import torch
import torch.nn as nn
from mitoXRNet import UNet,UNetDeep
from loss import CombinedLoss
import torch.optim as optim
from slice_loader import SliceLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
import time, logging
import argparse
from utils import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def dice_scoreBCE(pred, target, class_index, threshold=0.6, eps=1e-8):
    try:
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        pred_cls = pred[:, class_index, :, : ,:]
        target_cls = target[:, class_index, : , : , :]

        if pred_cls.shape != target_cls.shape:
            print(f"Shape mismatch: pred {pred_cls.shape}, target {target_cls.shape}")
            return torch.tensor(0.0, device=pred.device)
        intersection = torch.logical_and(pred_cls.bool(), target_cls.bool()).sum(dtype=torch.float32)
        union = pred_cls.sum() + target_cls.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice
    except Exception as e:
        print(f"Dice computation failed: {e}")
        return torch.tensor(0.0, device=pred.device)

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice_nucleus = 0.0
    total_dice_mito = 0.0
    batch_idx = 0
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        pred = torch.sigmoid(outputs)
        pred = (pred>0.6).float()
        dice_1 = dice_scoreBCE(pred, labels, 0) # Nucleus
        dice_2 = dice_scoreBCE(pred, labels, 1) # Mito

        total_dice_nucleus += dice_1.item()
        total_dice_mito += dice_2.item()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()        
        batch_idx += 1
        pbar.set_postfix({
            'Loss': total_loss / batch_idx,
            'Dice Nucleus': total_dice_nucleus / batch_idx,
            'Dice Mito': total_dice_mito / batch_idx
        })
        pbar.refresh()
        del outputs, inputs, labels
    pbar.close()
    return total_loss / len(dataloader)

def train_epoch_with_accumulation(model, dataloader, loss_fn, optimizer, device, accumulation_steps=2, beta = 0.85):
    model.train()
    total_loss = 0.0
    total_dice_nucleus = 0.0
    total_dice_mito = 0.0
    
    pbar = tqdm(dataloader, desc="Training", miniters=1)
    # 1. Zero gradients before the accumulation cycle begins
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = model(inputs)
        # Calculate metrics as before
        with torch.no_grad(): # Metrics don't need to track gradients
            pred = torch.sigmoid(outputs)
            pred = (pred > 0.6).float()
            dice_1 = dice_scoreBCE(pred, labels, 0) # Nucleus
            dice_2 = dice_scoreBCE(pred, labels, 1) # Mito
            total_dice_nucleus += dice_1.item()
            total_dice_mito += dice_2.item()
        # 2. Scale the loss by the number of accumulation steps
        loss = loss_fn(outputs, labels)
        loss = loss / accumulation_steps
        # 3. Accumulate the gradients
        loss.backward()
        # 4. Update model weights and zero gradients after N steps or on the last batch
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        # For logging, un-scale the loss to get the correct batch loss
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({
            'Loss': total_loss / (i + 1),
            'Dice Nucleus': total_dice_nucleus / (i + 1),
            'Dice Mito': total_dice_mito / (i + 1)
        })
        pbar.refresh()
        del outputs, inputs, labels, loss, pred # Clean up memory
    pbar.close()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice_nucleus = 0.0
    total_dice_mito = 0.0
    batch_idx = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", miniters=1)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # pred = torch.argmax(outputs[0], dim=1)
            pred = torch.sigmoid(outputs)
            pred = (pred>0.6).float()
            dice_1 = dice_scoreBCE(pred, labels, 0) # Nucleus
            dice_2 = dice_scoreBCE(pred, labels, 1) # Mito
            total_dice_nucleus += dice_1.item()
            total_dice_mito += dice_2.item()

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            batch_idx += 1
            pbar.set_postfix({
                'Loss': total_loss / batch_idx,
                'Val Dice Nucleus': total_dice_nucleus / batch_idx,
                'Val Dice Mito': total_dice_mito / batch_idx
            })
            pbar.refresh()
            del outputs, inputs, labels
        pbar.close()
    return total_loss / len(dataloader), total_dice_nucleus / len(dataloader), total_dice_mito / len(dataloader)

def Initialization(model_tag = 0, loss_tag=0, epochs = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print(f'--> Device: {device}\n')
    # Create UNet mode
    if model_tag == 0:
        model_name = "UNet"
        model = UNet(input_shape = (1,64,704,704), num_classes=2)
        batchSize = 4
        print(f'--> Model: {model_name}\n')
        print(f'--> BatchSize: {batchSize}\n')
    else:
        model_name = "UNetDeep"
        model = UNetDeep(input_shape = (1,64,704,704), num_classes=2)
        batchSize = 2
        print(f'--> Model: {model_name}\n')
        print(f'--> BatchSize: {batchSize}\n')
    # Define the loss function and optimizer
    if loss_tag == 0:
        loss_name = "CombinedLoss"
        loss_fn = CombinedLoss()
        print(f'--> Loss function: {loss_name}\n')
    else:
        loss_name = "BCELoss"
        loss_fn = nn.BCEWithLogitsLoss()
        print(f'--> Loss function: {loss_name}\n')
    print(f'--> Epochs: {epochs}')
        
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    # Parallelize the model for multi-GPU training
    model = nn.DataParallel(model)  
    model = model.to(device)
    # Load train and val datasets using SliceLoader
    train_dataset = SliceLoader("../Data/Slices/", "mrc_train_Slices", "mask_train_Slices") 
    val_dataset = SliceLoader("../Data/Slices/", "mask_val_Slices", "mask_val_Slices")
    # Define DataLoader for training and validating
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
    
    return loss_fn, optimizer, model, train_loader, val_loader, device, model_name, loss_name, epochs

def train(loss_fn, optimizer, model, train_loader, val_loader, device, model_name, loss_name, epochs):
    
    plot_save_dir = "../output/Plots"
    trainedWeightDir = "../output/Trained_Weights/"
    checkpoint_path = os.path.join(trainedWeightDir, f"Trained_model_{model_name}_{loss_name}")
    os.makedirs(plot_save_dir, exist_ok = True)
    os.makedirs(trainedWeightDir, exist_ok = True)
    progress_path = checkpoint_path + "_progress.pth"
    start_epoch = 0
    best_valid_loss = float('inf')

    if os.path.exists(progress_path):
        progress = torch.load(progress_path)
        start_epoch = progress['epoch'] + 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_valid_loss']
        best_epoch = checkpoint['epoch']+1
        print(f"Resuming from epoch {start_epoch}, with best validation loss: {best_valid_loss:.3f} registered at {best_epoch}th epoch")

    num_epochs = start_epoch + epochs
    train_losses = []
    valid_losses = []
    logging.basicConfig(filename = checkpoint_path + '.log',
                        filemode = 'a',
                        format = '%(asctime)s  [%(levelname)s]  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level = logging.INFO)
    logging.info(f'Starting training for: {checkpoint_path}.\n For Mito+Nucleus Classes.')

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch_with_accumulation(model, train_loader, loss_fn, optimizer, device, accumulation_steps=2)
        valid_loss, NucDice, MitoDice = evaluate_model(model, val_loader, loss_fn, device)
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Validation Loss: {valid_loss:.3f}')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print()
            print(data_str)
            best_valid_loss = valid_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss
            }
            torch.save(checkpoint, checkpoint_path)
        torch.save({'epoch': epoch}, progress_path)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t Nucelus DiceScore: {round(NucDice*100,2)}')
        print(f'\t Mito DiceScore: {round(MitoDice*100,2)}')
        torch.cuda.empty_cache()
        del train_loss,valid_loss,NucDice,MitoDice
    print()
    save_train_val_loss_plot(train_losses, valid_losses, plot_save_dir, filename=f"Train_Val_Loss_Graph_of_{os.path.basename(checkpoint_path)}.png")
    
def main():
    
    parser = argparse.ArgumentParser(description="MitoXRNet Training")
    parser.add_argument(
        "--model_tag",
        type=int,
        default=0,
        help="0 = UNet, 1 = UNetDeep (default: 0)"
    )
    parser.add_argument(
        "--loss_tag",
        type=int,
        default=0,
        help="0 = CombinedLoss, 1 = BCEWithLogitsLoss (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="No. of epochs for training (default: 60)"
    )
    args = parser.parse_args()
    print()
    print('------- Configuration to be used -------')
    loss_fn, optimizer, model, train_loader, val_loader, device, model_name, loss_name, epochs = Initialization(
        model_tag=args.model_tag,
        loss_tag=args.loss_tag,
        epochs = args.epochs
    )
    print()
    print(f"---------- Starting training ----------")
    print()

    train(loss_fn, optimizer, model, train_loader, val_loader, device, model_name, loss_name, epochs)
    print()


if __name__ == "__main__":
    main()
