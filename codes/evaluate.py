from slice_loader import SliceLoader, SliceLoader_MRC
import numpy as np
import torch
import os
import glob, gc
import torch.nn as nn
from MitoXRNet import UNet, UNetDeep
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
# Use absolute paths if these paths not working----##
# =========================

BASE_DATA_DIR = "../Data"
BASE_PRED_DIR = os.path.join(BASE_DATA_DIR, "Prediction")

LABEL_DIR = os.path.join(BASE_PRED_DIR, "Labels")
MRC_DIR = os.path.join(BASE_PRED_DIR, "MRCs")

PROCESSED_DIR = os.path.join(BASE_PRED_DIR, "Processed_MRCs")
SLICE_DIR = os.path.join(BASE_PRED_DIR, "MRC_predict_slices")

TEMP_SLICE_DIR = os.path.join(BASE_PRED_DIR, "MRC_prediction_slices_temp")
TEMP_NUC_DIR   = os.path.join(TEMP_SLICE_DIR, "Nucleus_Prediction")
TEMP_MITO_DIR  = os.path.join(TEMP_SLICE_DIR, "Mitochondria_Prediction")

MERGED_NUC_DIR  = os.path.join(BASE_PRED_DIR, "CombinedLossUNetPredictionMerged_nucleus")
MERGED_MITO_DIR = os.path.join(BASE_PRED_DIR, "CombinedLossUNetPredictionMerged_mitochondria")

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
        write_mrc(os.path.join(PROCESSED_DIR, os.path.basename(img_name)), padded_imgpp2)
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
    

def prediction(model_tag=0, pretrained = 0, Model_name = 'Trained_model_UNet_CombinedLoss', Threshold = 0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\n{device} founded, using it for Testing')
    if model_tag == 0:
        model_name = "UNet"
        model = UNet(input_shape = (1,64,704,704), num_classes=2)
    else:
        model_name = "UNetDeep"
        model = UNetDeep(input_shape = (1,64,704,704), num_classes=2)

    model = nn.DataParallel(model)
    if pretrained == 1:
        print(f'\n---> Loading Pretrained UNet Model\n')
        checkpoint = torch.load('../Outputs/Pretrained_Weights/UNet_CombinedLoss')
    elif pretrained == 2:
        print(f'\n---> Loading Pretrained UNetDeep Model\n')
        checkpoint = torch.load('../Outputs/Pretrained_Weights/UNetDeep_CombinedLoss')
    else:
        print(f'\n---> Loading Trained Model\n')
        checkpoint = torch.load(f'../Outputs/Trained_Weights/{Model_name}')
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    test_data = SliceLoader_MRC(BASE_PRED_DIR, 'mrc_predict_slices') 
    test_loader = DataLoader(test_data , batch_size = 1 , shuffle = False)
    
    os.makedirs(TEMP_NUC_DIR , exist_ok= True)
    os.makedirs(TEMP_MITO_DIR , exist_ok= True)

    for i,img in enumerate(test_loader):
        prediction = model(img)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.detach().cpu().numpy()
        prediction1 = prediction[0, 0, :, :, :]
        prediction2 = prediction[0, 1, :, :, :]
        filename1 = os.path.join(TEMP_NUC_DIR, os.path.basename(test_data.images_list[i]))
        filename2 = os.path.join(TEMP_MITO_DIR, os.path.basename(test_data.images_list[i]))    
        write_mrc(filename1, prediction1.astype(cp.float32))
        write_mrc(filename2, prediction2.astype(cp.float32))
        print(f'---> Predicting Slice = {i+1}', end='\r')
    
    print('\n-------- Starting Repatching of Individual Mitochondria and Nuclues prediction slices --------\n')
    
    dirPath = sorted(glob.glob(os.path.join(TEMP_SLICE_DIR,'*')))
    mergeDirPath = [MERGED_MITO_DIR, MERGED_NUC_DIR]
    
    for i,j in zip(dirPath,mergeDirPath):
        os.makedirs(j, exist_ok = True)
        fileList = sorted(os.listdir(i))
        cellIds = {}
        
        for filename in fileList:
            cell_id = None 
            match = re.search(r'(\d+_\d+)_merged', filename)
            if match:
                cell_id = match.group(1)
            elif (match := re.search(r'(\d+_\d+)_pre_rec', filename)):  
                cell_id = match.group(1)
            elif (match := re.search(r'(\d+_\d+-\d+)_pre_rec', filename)):  
                cell_id = match.group(1)

            axis = re.search(r'(\w+)_patch', filename).group(1)[-1]
            patchNum = int(re.search(r'patch(\d+)', filename).group(1))
            if cell_id not in cellIds:
                cellIds[cell_id] = {}
            if axis not in cellIds[cell_id]:
                cellIds[cell_id][axis] = {}
            if "mergedfileName" not in cellIds[cell_id]:
                if "pre_rec" in filename:
                    cellIds[cell_id]["mergedfileName"] = filename.split("_pre_rec")[0]+"_merged_prediction.mrc"
                else:
                    cellIds[cell_id]["mergedfileName"] = filename.split("_merged_prediction")[0]+"_merged_prediction.mrc"
            if patchNum not in cellIds[cell_id][axis]:
                cellIds[cell_id][axis][patchNum] = os.path.abspath(os.path.join(i, filename))
        print(f'Cell_Id {cellIds.keys()}, total cells detected {len(cellIds)}')
        print()
        
        for cellId in cellIds:
            addImage = np.zeros((704,704,704), dtype=np.float32)
            counterImage = np.zeros((704,704,704), dtype=np.float32)
            counterSlice = np.ones((64, 704, 704), dtype=np.float32)
            axis = "x"
            for patchNum in sorted(cellIds[cell_id][axis].keys()):
                filename = cellIds[cellId][axis][patchNum]
                v = read_mrc(filename, 'Cupy').get()
                startIndex = patchNum*32
                endIndex = startIndex + 64
                addImage[startIndex:endIndex, :, :] += v
                counterImage[startIndex:endIndex, :, :] += counterSlice
            addImage = np.rot90(addImage, axes=(0,2))
            counterImage = np.rot90(counterImage, axes=(0,2))
            addImage = np.flip(addImage, axis = 0)
            counterImage = np.flip(counterImage, axis = 0)
            axis = "y"
            for patchNum in sorted(cellIds[cell_id][axis].keys()):
                filename = cellIds[cellId][axis][patchNum]
                v = read_mrc(filename)
                startIndex = patchNum*32
                endIndex = startIndex + 64
                addImage[startIndex:endIndex, :, :] += v
                counterImage[startIndex:endIndex, :, :] += counterSlice
            addImage = np.rot90(addImage, axes=(1,0))
            counterImage = np.rot90(counterImage, axes=(1,0))
            addImage = np.flip(addImage, axis = 1)
            counterImage = np.flip(counterImage, axis = 1)    
            axis = "z"
            for patchNum in sorted(cellIds[cell_id][axis].keys()):
                filename = cellIds[cellId][axis][patchNum]
                v = read_mrc(filename)
                startIndex = patchNum*32
                endIndex = startIndex + 64
                addImage[startIndex:endIndex, :, :] += v
                counterImage[startIndex:endIndex, :, :] += counterSlice
            addImage = addImage/counterImage
            filename = os.path.join(j, cellIds[cellId]["mergedfileName"])
            write_mrc(filename, addImage.astype(cp.float32))
            print(f"Saved: {filename}\n")
            
            del addImage, counterImage, counterSlice, v, filename
            gc.collect()
        
    if os.path.exists(TEMP_SLICE_DIR):
        shutil.rmtree(TEMP_SLICE_DIR)
        
    print('\n-------- Repatching of prediction slices Completed --------')
    print()
    print('-------- Starting Merging of Mitochondria and Nucleus Predictions --------\n')
    
    maskList = sorted(glob.glob(os.path.join(LABEL_DIR,'*.mrc')))
    nucleusList = sorted(glob.glob(os.path.join(MERGED_NUC_DIR,'*.mrc')))
    
    mitoList = sorted(glob.glob(os.path.join(MERGED_MITO_DIR,'*.mrc')))
    print(len(maskList), len(nucleusList), len(mitoList))
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
        write_mrc(os.path.join(Outputpath,name),final_prediction.astype(cp.float32))
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

    table.add_column("Nuc IoU",  justify="center", width=6)
    table.add_column("Nuc Dice", justify="center", width=6)
    table.add_column("Nuc Prec", justify="center", width=6)
    table.add_column("Nuc Rec",  justify="center", width=6)
    
    table.add_column("Mito IoU",  justify="center", width=6)
    table.add_column("Mito Dice", justify="center", width=6)
    table.add_column("Mito Prec", justify="center", width=6)
    table.add_column("Mito Rec",  justify="center", width=6)

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
            f"{nuc_iou:.2f}",  f"{nuc_dice:.2f}",  f"{nuc_prec:.2f}",  f"{nuc_rec:.2f}",
            f"{mito_iou:.2f}", f"{mito_dice:.2f}", f"{mito_prec:.2f}", f"{mito_rec:.2f}",
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
        f"{sums['nuc_iou']/n:.4f}",  f"{sums['nuc_dice']/n:.4f}",
        f"{sums['nuc_prec']/n:.4f}", f"{sums['nuc_rec']/n:.4f}",
        f"{sums['mito_iou']/n:.4f}", f"{sums['mito_dice']/n:.4f}",
        f"{sums['mito_prec']/n:.4f}", f"{sums['mito_rec']/n:.4f}",
    )
    console.print(table)
    # ---- Save JSON ----
    out_json = f"../Outputs/Evaluation_results/Metrics_threshold_{threshold}_results.json"
    os.makedirs('../Outputs/Evaluation_results/', exist_ok = True)
    
    with open(out_json, "w") as f:
        json.dump({
            "threshold": threshold,
            "mean": {
                "nucleus": {
                    "iou": sums["nuc_iou"]/n,
                    "dice": sums["nuc_dice"]/n,
                    "precision": sums["nuc_prec"]/n,
                    "recall": sums["nuc_rec"]/n,
                },
                "mitochondria": {
                    "iou": sums["mito_iou"]/n,
                    "dice": sums["mito_dice"]/n,
                    "precision": sums["mito_prec"]/n,
                    "recall": sums["mito_rec"]/n,
                }
            },
            "results": results
        }, f, indent=4)
    print()
    print(f"\nSaved metrics results to: {out_json}")
    
def main():
    parser = argparse.ArgumentParser(description="MitoXRNet Full Pipeline")
    parser.add_argument(
        "--model_tag", type=int, default=0,
        help="0 = UNet, 1 = UNetDeep (default = 0)"
    )
    parser.add_argument(
        "--pretrained", type=int, default=0,
        help="0 = Trained Model, 1 = UNet, 2 = UNetDeep (default = 0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Threshold for prediction and metrics (default = 0.6)"
    )
    parser.add_argument(
        "--only_preprocessing",
        action="store_true",
        help="Run ONLY preprocessing and exit"
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing step (assumes preprocessed data already exists)"
    )
    parser.add_argument(
        "--only_prediction",
        action="store_true",
        help="Run ONLY prediction (skip preprocessing and metrics)"
    )
    parser.add_argument(
        "--only_metrics",
        action="store_true",
        help="Run ONLY metric evaluation (assumes predictions already exist)"
    )
    args = parser.parse_args()

    if args.only_prediction and args.only_metrics:
        raise ValueError("Choose only one: --only_prediction OR --only_metrics")
    if args.only_preprocessing and (
        args.only_prediction or args.only_metrics
    ):
        raise ValueError(
            "--only_preprocessing cannot be combined with prediction or metrics flags"
        )
    if args.only_preprocessing:
        print()
        print("---------- Running ONLY Preprocessing ----------\n")

        preprocessing()
        print("\n---------- Preprocessing Completed ----------\n")
        return
    if not args.skip_preprocessing and not args.only_metrics:
        print("\n---------- Starting Preprocessing ----------\n")
        preprocessing()
    # ---- prediction ----
    if not args.only_metrics:
        print("\n---------- Starting Prediction ----------\n")
        prediction(
            model_tag=args.model_tag,
            pretrained=args.pretrained,
            Model_name="Trained_model_UNet_CombinedLoss",
            Threshold=args.threshold
        )
    # ---- metrics ----
    if not args.only_prediction:
        print("\n-------------- Calculating Metrics Score on Predicted Cells --------------\n")
        metrics_eval(threshold=args.threshold)
    print("\n---------- Pipeline Completed ----------\n")

if __name__ == "__main__":
    main()

        
