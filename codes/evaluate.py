from slice_loader import SliceLoader, SliceLoader_MRC
import numpy as np
import torch
import os
import glob, gc
import torch.nn as nn
from mitoXRNet import UNet, UNetDeep
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import *
import cupy as cp
import json
from rich.table import Table
from rich.console import Console
import shutil, time, re
import argparse

# =========================
# Global path configuration
# Use absolute paths if these paths not working
# =========================

BASE_DATA_DIR = "../Data"
BASE_PRED_DIR = os.path.join(BASE_DATA_DIR, "Prediction")

LABEL_DIR = os.path.join(BASE_PRED_DIR, "Labels")
MRC_DIR = os.path.join(BASE_PRED_DIR, "MRCs")

PROCESSED_DIR = os.path.join(BASE_PRED_DIR, "MRCs_preprocessed")
SLICE_DIR = os.path.join(BASE_PRED_DIR, "slices")

TEMP_SLICE_DIR = os.path.join(BASE_PRED_DIR, "prediction_slices_temp")
TEMP_NUC_DIR   = os.path.join(TEMP_SLICE_DIR, "Nucleus_Prediction")
TEMP_MITO_DIR  = os.path.join(TEMP_SLICE_DIR, "Mitochondria_Prediction")

MERGED_NUC_DIR  = os.path.join(BASE_PRED_DIR, "PredictionMerged_nucleus")
MERGED_MITO_DIR = os.path.join(BASE_PRED_DIR, "PredictionMerged_mitochondria")

def pred_out_dir(threshold):
    return os.path.join(BASE_PRED_DIR, f"PredictedLabels/Predicted_labels_Th{threshold}_Images")

def preprocessing():
    
    print("\n----- Preprocessing Test Data -----\n")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SLICE_DIR, exist_ok=True)

    target_size = (704,704,704)
    count=0
    skipped = 0
    img_name_list = sorted(glob.glob(os.path.join(MRC_DIR, "*.mrc")))
    mask_name_list =  sorted(glob.glob(os.path.join(LABEL_DIR, "*.mrc")))
    
    for img_name,mask_name in zip(img_name_list, mask_name_list):
        img = read_mrc(img_name)
        mask = read_mrc(mask_name)
        assert img.shape == mask.shape, "Mask and image don't match."
        if any(dim > 704 for dim in img.shape):
            print(f"Skipping {img_name} (shape: {img.shape}) — exceeds 704 in one or more dimensions.")
            skipped += 1
            continue

        imgpp2 = ves_mito_separator(img)
        imgpp2 = mask_crop(imgpp2,mask)
        padded_imgpp2 = center_padding(imgpp2,target_size)
        assert padded_imgpp2.shape == (704,704,704), f"Something error after padding in padded_imgpp2 {padded_imgpp2.shape}"
        gc.collect()
        write_mrc(os.path.join(PROCESSED_DIR, os.path.basename(img_name)), padded_imgpp2.astype(np.float32))
        count+=1
        print(f'----- Preprocessing Done for {os.path.basename(img_name)} -----')
                          
    print(f'\n----- Preprocessing Completed , total skipped images {skipped} -----\n')
    time.sleep(2)
    print('--------- Slicing Started ---------\n')
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith('.mrc'):  
            image_path = os.path.join(PROCESSED_DIR, filename)
            image = read_mrc(image_path)
            
            create_and_save_slices(image, os.path.splitext(filename)[0],SLICE_DIR)
            del image, image_path
            gc.collect()
    print()                             
    print('--------- Slicing Completed ---------\n')
    
def prediction(pretrained = 1, Model_name = 'Trained_model_UNet_CombinedLoss', Threshold = 0.6):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\n--> {device} founded, using it for Testing')
    if pretrained == 1:
        print(f'\n---> Loading Pretrained Shallow UNet Model\n')
        checkpoint = torch.load('../output/Pretrained_Weights/UNet_CombinedLoss')
        model = UNet(input_shape = (1,64,704,704), num_classes=2)
    elif pretrained == 2:
        print(f'\n---> Loading Pretrained UNetDeep Model\n')
        checkpoint = torch.load('../output/Pretrained_Weights/UNetDeep_CombinedLoss')
        model = UNetDeep(input_shape = (1,64,704,704), num_classes=2)
    else:
        checkpoint = torch.load(f'../output/Trained_Weights/{Model_name}')
        if "UNetDeep" in Model_name:
            model = UNetDeep(input_shape = (1,64,704,704), num_classes=2)
            print(f'\n---> Loading Trained **{Model_name}** Model, Base model: UNetDeep\n')
        elif "UNet_" in Model_name:
            model = UNet(input_shape = (1,64,704,704), num_classes=2)
            print(f'\n---> Loading Trained **{Model_name}** Model, Base model: UNet\n')
        else:
            raise ValueError("Model name is not correct. Please provide  correct --model_name: Trained_model_<model type>_<loss used>")
    
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    test_data = SliceLoader_MRC(BASE_PRED_DIR, 'slices') 
    test_loader = DataLoader(test_data , batch_size = 1 , shuffle = False)
    os.makedirs(TEMP_NUC_DIR , exist_ok= True)
    os.makedirs(TEMP_MITO_DIR , exist_ok= True)
    
    for i,img in enumerate(test_loader):
        img = img.to(device)
        prediction = model(img)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.detach().cpu().numpy()
        prediction1 = prediction[0, 0, :, :, :]
        prediction2 = prediction[0, 1, :, :, :]
        filename1 = os.path.join(TEMP_NUC_DIR, os.path.basename(test_data.images_list[i]))
        filename2 = os.path.join(TEMP_MITO_DIR, os.path.basename(test_data.images_list[i]))    
        write_mrc(filename1, prediction1)
        write_mrc(filename2, prediction2)
        print(f'---> Predicting Slice = {i+1}', end='\r')
    print()
    print('\n-------- Starting Repatching of Individual Mitochondria and Nuclues prediction slices --------\n')
    
    dirPath = sorted(glob.glob(os.path.join(TEMP_SLICE_DIR,'*')))
    mergeDirPath = [MERGED_MITO_DIR, MERGED_NUC_DIR]
    for src_dir, dst_dir in zip(dirPath, mergeDirPath):
        os.makedirs(dst_dir, exist_ok=True)
        fileList = sorted(os.listdir(src_dir))
        print(f'fileList: {len(fileList)}')
        cells = {}
        for filename in fileList:
            if not filename.endswith(".mrc"):
                continue
            axis_match = re.search(r'_([xyz])_patch(\d+)', filename)
            if axis_match is None:
                continue
            axis = axis_match.group(1)
            patchNum = int(axis_match.group(2))
            parent_name = filename[:axis_match.start()]
            if parent_name not in cells:
                cells[parent_name] = {
                    "x": {},
                    "y": {},
                    "z": {},
                    "mergedfileName": parent_name + "_merged_prediction.mrc"
                }
            cells[parent_name][axis][patchNum] = os.path.abspath(
                os.path.join(src_dir, filename)
            )
        print(f"Detected {len(cells)} unique cells:")
        for k in cells:
            print("  ", k)
        print()
        for cell_name, cell_data in cells.items():
            addImage = np.zeros((704, 704, 704), dtype=np.float32)
            counterImage = np.zeros((704, 704, 704), dtype=np.float32)
            counterSlice = np.ones((64, 704, 704), dtype=np.float32)
            # ---------------- X axis ----------------
            for patchNum in sorted(cell_data["x"]):
                v = read_mrc(cell_data["x"][patchNum])
                s, e = patchNum * 32, patchNum * 32 + 64
                addImage[s:e, :, :] += v
                counterImage[s:e, :, :] += counterSlice
            addImage = np.rot90(addImage, axes=(0, 2))
            counterImage = np.rot90(counterImage, axes=(0, 2))
            addImage = np.flip(addImage, axis=0)
            counterImage = np.flip(counterImage, axis=0)
            # ---------------- Y axis ----------------
            for patchNum in sorted(cell_data["y"]):
                v = read_mrc(cell_data["y"][patchNum])
                s, e = patchNum * 32, patchNum * 32 + 64
                addImage[s:e, :, :] += v
                counterImage[s:e, :, :] += counterSlice
            addImage = np.rot90(addImage, axes=(1, 0))
            counterImage = np.rot90(counterImage, axes=(1, 0))
            addImage = np.flip(addImage, axis=1)
            counterImage = np.flip(counterImage, axis=1)
            # ---------------- Z axis ----------------
            for patchNum in sorted(cell_data["z"]):
                v = read_mrc(cell_data["z"][patchNum])
                s, e = patchNum * 32, patchNum * 32 + 64
                addImage[s:e, :, :] += v
                counterImage[s:e, :, :] += counterSlice
            # ---------------- Finalize ----------------
            addImage /= counterImage
            out_path = os.path.join(dst_dir, cell_data["mergedfileName"])
            write_mrc(out_path, addImage.astype(np.float32))
            print(f"Saved: {out_path}")
            del addImage, counterImage, counterSlice
            gc.collect()
        
    if os.path.exists(TEMP_SLICE_DIR):
        shutil.rmtree(TEMP_SLICE_DIR)
    print('\n-------- Repatching of prediction slices Completed --------')
    print()
    print('-------- Starting Merging of Mitochondria and Nucleus Predictions --------\n')
    
    maskList = sorted(glob.glob(os.path.join(LABEL_DIR,'*.mrc')))
    nucleusList = sorted(glob.glob(os.path.join(MERGED_NUC_DIR,'*.mrc')))
    mitoList = sorted(glob.glob(os.path.join(MERGED_MITO_DIR,'*.mrc')))
    Outputpath =  pred_out_dir(threshold = Threshold) # Can be calculated at different thresholds.
    os.makedirs(Outputpath,exist_ok=True)
    
    count = 0
    for mask,nucleusPath,mitoPath in zip(maskList, nucleusList, mitoList): 
        count+=1
        print(f'---Merging = {count} ---',end='\r')
        original_label = read_mrc(mask)
        original_label = center_padding(original_label,(704,704,704))
        merged_img_nucleus = read_mrc(nucleusPath)
        merged_img_mitochondria = read_mrc(mitoPath)
        nucleus = np.where(merged_img_nucleus<Threshold, 0 , merged_img_nucleus)
        mitochondria = np.where(merged_img_mitochondria<Threshold, 0 , merged_img_mitochondria)
        final_prediction = np.zeros(original_label.shape , dtype = np.int8)
        final_prediction[original_label>0] = 1
        np.place(final_prediction , nucleus>mitochondria , [2.0])
        np.place(final_prediction , mitochondria>nucleus , [5.0])
        name = os.path.basename(nucleusPath)
        write_mrc(os.path.join(Outputpath,name),final_prediction)
        del nucleus, mitochondria,final_prediction,original_label, name, merged_img_mitochondria, merged_img_nucleus
        gc.collect()
                
    PredPath = sorted(glob.glob(os.path.join(Outputpath,'*.mrc')))
    for i,j in zip(PredPath, maskList):
        img = read_mrc(i)
        mask = read_mrc(j)
        cropped_img = anti_padding(img,mask.shape)
        write_mrc(os.path.join(Outputpath, os.path.basename(i)),cropped_img.astype(cp.float32))
    if os.path.exists(MERGED_NUC_DIR) and os.path.exists(MERGED_MITO_DIR):
        shutil.rmtree(MERGED_MITO_DIR)
        shutil.rmtree(MERGED_NUC_DIR)        
    print('\n------- Merging and Resizing Completed -------\n')

def pct(x):
    return f"{x * 100:.2f}"

def metrics_eval(threshold=0.6):

    pred_dir = pred_out_dir(threshold=threshold)

    Predictions = sorted(glob.glob(os.path.join(pred_dir, "*.mrc")))
    GT_Labels   = sorted(glob.glob(os.path.join(LABEL_DIR, "*.mrc")))

    if len(Predictions) == 0:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    console = Console()
    table = Table(title=f"Segmentation Metrics (Threshold = {threshold})")
    table.add_column("Index", justify="center", width = 5)
    table.add_column("File", justify="left")
    table.add_column("Nuc IoU",  justify="center", width=5)
    table.add_column("Nuc Dice", justify="center", width=5)
    table.add_column("Nuc Prec", justify="center", width=5)
    table.add_column("Nuc Rec",  justify="center", width=5)    
    table.add_column("Mito IoU",  justify="center", width=5)
    table.add_column("Mito Dice", justify="center", width=5)
    table.add_column("Mito Prec", justify="center", width=5)
    table.add_column("Mito Rec",  justify="center", width=5)

    results = []
    sums = {
        "nuc_iou": 0.0, "nuc_dice": 0.0, "nuc_prec": 0.0, "nuc_rec": 0.0,
        "mito_iou": 0.0, "mito_dice": 0.0, "mito_prec": 0.0, "mito_rec": 0.0,
    }
    for idx, (pred_path, gt_path) in enumerate(zip(Predictions, GT_Labels)):
        pred = read_mrc(pred_path)
        gt   = read_mrc(gt_path)
        # -------- NUCLEUS (label = 2) --------
        pred_nuc = (pred == 2)
        gt_nuc   = (gt == 2)
        nuc_iou, nuc_dice = IOU_and_DICE(pred_nuc, gt_nuc)
        nuc_prec, nuc_rec, _ = Precision_recall_f1_score(pred_nuc, gt_nuc)
        # -------- MITOCHONDRIA (label = 5) --------
        pred_mito = (pred == 5)
        gt_mito   = (gt == 5)
        mito_iou, mito_dice = IOU_and_DICE(pred_mito, gt_mito)
        mito_prec, mito_rec, _ = Precision_recall_f1_score(pred_mito, gt_mito)
        # accumulate
        sums["nuc_iou"]  += nuc_iou
        sums["nuc_dice"] += nuc_dice
        sums["nuc_prec"] += nuc_prec
        sums["nuc_rec"]  += nuc_rec
        sums["mito_iou"]  += mito_iou
        sums["mito_dice"] += mito_dice
        sums["mito_prec"] += mito_prec
        sums["mito_rec"]  += mito_rec
        fname = os.path.basename(pred_path)
        table.add_row(
            str(idx), fname,
            pct(nuc_iou),  pct(nuc_dice),  pct(nuc_prec),  pct(nuc_rec),
            pct(mito_iou), pct(mito_dice), pct(mito_prec), pct(mito_rec),
        )

        results.append({
            "file": fname,
            "nucleus": {
                "iou": nuc_iou, "dice": nuc_dice,
                "precision": nuc_prec, "recall": nuc_rec
            },
            "mitochondria": {
                "iou": mito_iou, "dice": mito_dice,
                "precision": mito_prec, "recall": mito_rec
            }
        })
    n = len(results)
    table.add_section()
    table.add_row(
        "—", "               MEAN",
        pct(sums['nuc_iou']/n),  pct(sums['nuc_dice']/n),
        pct(sums['nuc_prec']/n), pct(sums['nuc_rec']/n),
        pct(sums['mito_iou']/n), pct(sums['mito_dice']/n),
        pct(sums['mito_prec']/n), pct(sums['mito_rec']/n),
    )

    console.print(table)
    # ---- Save JSON ----
    out_json = f"../output/Evaluation_results/Metrics_threshold_{threshold}_results.json"
    os.makedirs('../output/Evaluation_results/', exist_ok = True)
    
    with open(out_json, "w") as f:
        json.dump({
            "threshold": threshold,
            "mean": {
                "nucleus": {
                    "iou": pct(sums['nuc_iou']/n),
                    "dice": pct(sums['nuc_dice']/n),
                    "precision": pct(sums['nuc_prec']/n),
                    "recall": pct(sums['nuc_rec']/n),
                },
                "mitochondria": {
                    "iou": pct(sums['mito_iou']/n),
                    "dice": pct(sums['mito_dice']/n),
                    "precision": pct(sums['mito_prec']/n),
                    "recall": pct(sums['mito_rec']/n),
                }
            },
            "results": results
        }, f, indent=4)

    print()
    print(f"\nSaved metrics results to: {out_json}")
    
def main():
    parser = argparse.ArgumentParser(description="MitoXRNet Full Pipeline")
    parser.add_argument(
        "--pretrained", type=int, default=1,
        help="0 = Trained Model, 1 = UNet, 2 = UNetDeep (default = 1)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Threshold for prediction and metrics (default = 0.6)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Trained_model_UNet_CombinedLoss",
        help="Name of the trained model checkpoint (without extension)"
    )
    args = parser.parse_args()
    
    preprocessing()
    print()
    print("\n---------- Preprocessing Completed ----------")
    print("\n---------- Starting Prediction ----------\n")
    prediction(
        pretrained=args.pretrained,
        Model_name=args.model_name,
        Threshold=args.threshold
    )
    print("\n-------------- Calculating Metrics Score on Predicted Cells --------------\n")
    metrics_eval(threshold=args.threshold)
    print("\n---------- Pipeline Completed ----------\n")

    
if __name__ == "__main__":
    main()

        
