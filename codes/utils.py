import numpy as np
import cupy as cp
import os, shutil, random, skimage, glob
from patchify import patchify, unpatchify
from scipy import ndimage
import struct
import matplotlib.pyplot as plt
def normalisation(img):
    norm_img = (img-img.min())/(img.max()-img.min())
    return norm_img

def center_padding(img, target_size):
    pad_widths = [(max((target_size[i] - img.shape[i]) // 2, 0), max((target_size[i] - img.shape[i] + 1) // 2, 0)) for i in range(3)]
    pad_img = np.pad(img, pad_widths)
    return pad_img

def anti_padding(img, target_size):
    if any(img.shape[i] < target_size[i] for i in range(3)):
        raise ValueError("Target size must be smaller than or equal to the input image size for anti-padding.")
    crop_slices = [slice((img.shape[i] - target_size[i]) // 2, (img.shape[i] - target_size[i]) // 2 + target_size[i]) for i in range(3)]
    cropped_img = img[crop_slices[0], crop_slices[1], crop_slices[2]]
    
    return cropped_img

def mask_crop(image, labels):
    cropped_mrc = np.where(labels>0, image, 0)
    return cropped_mrc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_train_val_loss_plot(
    train_losses,
    val_losses,
    save_dir,
    filename="train_val_loss.png"
):
    """
    Saves training and validation loss plot to disk.

    Args:
        train_losses (list): training loss per epoch
        val_losses (list): validation loss per epoch
        save_dir (str): directory where plot will be saved
        filename (str): name of the image file
    """

    # ---- safety checks ----
    assert len(train_losses) == len(val_losses), \
        "Train and validation loss lists must have same length"

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)   
    plt.close()                       

    print(f"-----Loss plot saved at: {save_path}-----")

def create_and_save_slices(img, img_name, output_folder):
    patches_z = patchify(img, (64,704,704), step=(32,32,32))
    patches_z = patches_z.squeeze((1,2))
    tmp = np.rot90(img, axes=(0,1))
    patches_y = np.flip(tmp, axis = 0)
    patches_y = patchify(patches_y, (64,704,704), step = 32)
    patches_y = patches_y.squeeze((1,2))                 
    tmp = np.rot90(tmp, axes=(2,0))
    patches_x = patchify(tmp, (64 ,704 ,704), step = 32)
    patches_x = patches_x.squeeze((1,2))
    # Iterate through axes
    for axis_name, patches in zip(['z', 'y', 'x'], [patches_z, patches_y, patches_x]):
        for i, patch in enumerate(patches):
            patch_filename = f'{img_name}_{axis_name}_patch{i}.mrc'
            patch_path = os.path.join(output_folder, patch_filename)
            write_mrc(patch_path, patch)
    print(f'Slices created for {img_name} at {output_folder}')

def read_mrc(filename, tag = "Numpy"):
    filename = str(filename)
    input_image=open(filename,'rb')
    num_ints=56
    sizeof_int=4
    nlines_header=10
    num_chars=80
    
    head1=input_image.read(num_ints*sizeof_int) # read 56 long ints
    head2=input_image.read(nlines_header*num_chars) #read 10 lines of 80 chars
    byte_pattern='=' + 'l'*num_ints   #'=' required to get machine independent standard size
    dim=struct.unpack(byte_pattern, head1)[:3][::-1]
    imagetype=struct.unpack(byte_pattern,head1)[3]  #0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
    #print("Imagetype = {}".format(imagetype ))
    if (imagetype == 0):
        imtype='b'
    elif (imagetype ==1):
        imtype='h'
    elif (imagetype ==2):
        imtype='f4'
    elif (imagetype ==6):
        imtype='H'
    else:
        type='unknown'   #should put a fail here
        
    num_voxels=dim[0]*dim[1]*dim[2]
    if tag == "Cupy":
        image_data=cp.fromfile(file=input_image,dtype=imtype,count=num_voxels).reshape(dim)
    else:
        image_data=np.fromfile(file=input_image,dtype=imtype,count=num_voxels).reshape(dim)
        
    input_image.close()
    return image_data

def write_mrc(filename, im, num_ints=56, sizeof_int=4, num_chars=800):
    filename = str(filename)
    type_modes = {
        'b': 0,
        'h': 1,
        'f': 2,
        'H': 6
    }
    mode = type_modes[im.dtype.char]
    dims = im.shape[::-1]
    header1 = struct.pack('=' + 'l'*num_ints, *(dims + (mode,) + (0,)*(num_ints - len(dims) - 1)))
    header2 = struct.pack('=' + 'c'*num_chars, *(b" ",)*num_chars)    
    with open(filename, 'wb') as f:
        f.write(header1)
        f.write(header2)
        f.write(im.tobytes())
        
def split_dataset(mrc_folder, mask_folder, output_folder, split_ratio=0.8, seed=42):
    random.seed(seed)

    mrc_files = glob.glob(os.path.join(mrc_folder, "*.mrc"))
    mask_files = glob.glob(os.path.join(mask_folder, "*.mrc"))

    mrc_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mrc_files}
    mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    common_keys = sorted(set(mrc_dict) & set(mask_dict))
    assert len(common_keys) >= 2, "Need at least 2 samples to split train/val!"

    pairs = [(mrc_dict[k], mask_dict[k]) for k in common_keys]
    random.shuffle(pairs)

    total = len(pairs)

    # --- SAFE split ---
    train_end = max(1, int(total * split_ratio))
    train_end = min(train_end, total - 1)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:]
    }

    for split_name, split_pairs in splits.items():
        mrc_out = os.path.join(output_folder, f"mrc_{split_name}_preprocessed")
        mask_out = os.path.join(output_folder, f"mask_{split_name}_preprocessed")
        os.makedirs(mrc_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for mrc_file, mask_file in split_pairs:
            shutil.copy2(mrc_file, os.path.join(mrc_out, os.path.basename(mrc_file)))
            shutil.copy2(mask_file, os.path.join(mask_out, os.path.basename(mask_file)))

    print(f"Dataset split completed → {output_folder}")
    print(f"Train: {train_end}, Val: {total - train_end}")

    
def IOU_and_DICE(prediction , Ground_Truth):
    intersection = np.logical_and(prediction,Ground_Truth)
    union = np.logical_or(prediction,Ground_Truth)
    IOU = np.sum(intersection)/np.sum(union)
    DICE =  (2.0 * np.sum(intersection)) / (np.sum(prediction) + np.sum(Ground_Truth))
    return IOU , DICE
                    
def Precision_recall_f1_score(prediction, Ground_Truth):
    true_positive = np.sum(np.logical_and(prediction, Ground_Truth))
    false_positive = np.sum(np.logical_and(prediction, np.logical_not(Ground_Truth)))
    false_negative = np.sum(np.logical_and(np.logical_not(prediction), Ground_Truth))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score            

def hpf_sobel(image):
    gradient_x = ndimage.sobel(image, axis=1, mode='constant')
    gradient_y = ndimage.sobel(image, axis=0, mode='constant')
    gradient_z = ndimage.sobel(image, axis=2, mode='constant')
    # Combine the gradients to get the magnitude of the gradient
    sobel_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
    return sobel_magnitude

def ves_mito_separator(img,sigma=2):
    out = img + ndimage.gaussian_filter(img, sigma)
    hpf_out = hpf_sobel(out)
    out = img + 0.75*hpf_out + 0.25*hpf_out*hpf_out
    out = normalisation(out)*normalisation(img)
    out = normalisation(out)
    return out
