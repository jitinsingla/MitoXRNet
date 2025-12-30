# 🧬 MitoXRNet 

## Prerequisties / Environment Setup
Anaconda install ([link](https://docs.anaconda.com/anaconda/install/))

## Create environment (recommended):
```
conda env create -f env/environment.yml
conda activate sxt_seg
```
if using env file give error, just ensure all the following packages are installed:
gc, cupy, pytorch, numpy, pandas, tenorflow, shutil, matplotlib.

## MitoXRNet Usage
User can train the model from scratch or use the pretrained weights from the paper to directly predict on their dataset.
For training from sratch follow the instructions mentioned under **"Training from Scratch"** and then predict using **"Prediction"** instructions.
For directly predicting using the pretrained models, skip training instructions and directly go to **"Prediction"** section.

## Training from Scratch
#### Data preparation:
Folder: `Data/Training`
Data in this folder is used for training the model from scratch. The data in this folder will be split into training and validation sets in x, y %
To prepare the data for training, copy raw mrc files in `Data/Traning/MRCs` and ground truth labels in `Data/Traning/Labels`.

#### Notes about data preparation:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg ([link](https://biomedisa.info/gallery/#))
Size example (mrc.shape == label.shape)
Size should be in order for both MRC and Label image `(z,y,x) = (z,y,x)`
Make sure both raw MRCs and Labels are in .mrc format.
Name should be same of both MRC file and corresponding Label file for correct mapping.
Name should be in a specific format only.
`<EXPERIMENT_METADATA>_<CELLID>_pre_rec.mrc`
Example CELLID:- `1111_13` , `1128_1-2`
Example FullName:- `KLW_PBC_INS1e_Ex-4_5min_1111_13_pre_rec.mrc`
Labels of Nucleus = `2` , Labels of Mitochondria = `5` , Labels of Cytoplasm = `1` and rest all `0`
Each 3D Image shape along any axis should be `<=704`. 

Steps:
#### Preprocessing: 
Here all the raw mrcs and masks are split in train and validation folders inside Data/Training folder (i.e. mrc_train, mrc_val, mask_train and mask_val)
Each mrc and mask undergoes padding, to ensure the image size matches the model's input requirement.
Each raw mrc undergoes preprocessing as proposed in MitoXRNet paper.
Further it creates 3D slices for train and validation dataset in folder Data/Training/slices.

```
# Run preprocessing with default 80/20 train–validation split
python codes/preprocessing.py

# Optionally, change the train/validation split ratio
python codes/preprocessing.py --split_ratio 0.7
```

#### Training:
```
# Train using default configuration (UNet + CombinedLoss)
python codes/train.py

# Train using UNetDeep (larger network)
python codes/train.py --model_tag 1

# Train using BCEWithLogitsLoss instead of CombinedLoss
python codes/train.py --loss_tag 0
```
- Default command trains **UNet** with **CombinedLoss**
- `--model_tag 1` → trains **UNetDeep (larger network)**
- `--loss_tag 0` → uses **BCEWithLogitsLoss**
- `--model_tag 1 --loss_tag 0` → trains **UNetDeep** with **BCEWithLogitsLoss**

Training execution will create logs and trained model weights in the `Output/Trained_Weights/` folder
Early stopping has been intentially removed, so stop the training based on val and train error/accuracy.

## Predict
#### Data preparation:
Keep the raw mrcs for prediction in `Data/Prediction/MRCs` folder.
Notes:
MitoRNet requires that idividual cells is masked in raw MRC files (like using ACSeg ([link](https://biomedisa.info/gallery/#))
Size example (mrc.shape == label.shape)
Size should be in order for both MRC and Label image `(z,y,x) = (z,y,x)`
Make sure both raw MRCs and Labels are in .mrc format.
Name should be same of both MRC file and corresponding Label file for correct mapping.
Name should be in a specific format only.
`<EXPERIMENT_METADATA>_<CELLID>_pre_rec.mrc`
Example CELLID:- `1111_13, 1128_1-2`
Example FullName:- `KLW_PBC_INS1e_Ex-4_5min_1111_13_pre_rec.mrc`
Labels of Nucleus `2` , Labels of Mitochondria `5` , Labels of Cytoplasm `1` and rest all `0`
Each 3D Image shape along any axis should be `<=704`.

#### Evaluation
`evaluate.py` runs the complete MitoXRNet inference pipeline, including preprocessing, slice-wise prediction, and class-wise evaluation.  
It supports flexible execution modes such as **preprocessing only**, **prediction only**, **evaluation only**, or the full pipeline.  
Segmentation performance is reported using IoU, Dice, Precision, and Recall for both nucleus and mitochondria.

```
# Full evaluation pipeline (default)
python codes/evaluate.py
```
### Tag & Keyword Summary

| Flag / Mode | Description |
|------------|-------------|
| **(default)** | Runs **preprocessing → prediction → evaluation** using **UNet**, **user-trained weights**, threshold = `0.6` |
| `--model_tag 1` | Use **UNetDeep** architecture (larger network) |
| `--pretrained 0` | Use **user-trained weights** from `Output/Trained_Weights/` |
| `--pretrained 1` | Use **pretrained UNet** weights from paper `Output/Pretrained_Weights/`|
| `--pretrained 2` | Use **pretrained UNetDeep** weights from paper `Output/Pretrained_Weights/` |
| `--threshold <value>` | Set prediction & evaluation threshold (default = `0.6`) |
| `--only_preprocessing` | Run **only preprocessing** |
| `--skip_preprocessing` | Skip preprocessing, run **prediction + evaluation** |
| `--only_prediction` | Run **only prediction** |
| `--only_metrics` | Run **only evaluation** (predictions must exist) |

`--only_prediction` and `--only_metrics` cannot be used together

The above mentioned code performs the following steps:
1. Preprocessing  `Data/Prediction/Processed_MRCs`
2. Slicing  `Data/Prediction/MRC_predict_slices`
3. Model load
4. Predict on each slice  `Data/Prediction/MRC_prediction_slices_temp`
5. Merge slices

Final predicted labels are saved in `Data/Prediction/PredictedLabels` **(at original GT label sizes)**
Evaluation metric results on IOU, Dice, Precision, Recall will be displayed on terminal and a json file will be saved inside `Outputs/Evaluation_results` folder.
