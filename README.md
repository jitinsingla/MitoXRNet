# MitoXRNet 

## Prerequisties / Environment Setup
Install Anaconda for setup ([link](https://docs.anaconda.com/anaconda/install/))

#### Create environment (recommended):
```
conda env create -f env/environment.yml
conda activate sxt_seg
```
if using env file gives error, just ensure all the following packages are installed:
gc, cupy, pytorch, numpy, pandas, tenorflow, shutil, matplotlib.

## MitoXRNet Usage
User can train the model from scratch or use the pretrained weights from the paper to directly predict on their dataset.
For training from scratch follow the instructions mentioned under **"Training from Scratch"** and then predict using **"Prediction"** instructions.
For directly predicting using the pretrained models, skip training instructions and directly go to **"Prediction"** section.

## Training from Scratch
#### Data preparation:
Data in folder `Data/Training` is used for training the model from scratch. Available data will be split into training and validation sets. In paper, 52 cells were split into 49 training and 3 validation cells. Additionally 3 cells were kept for Prediction.
To prepare the data for training, copy raw mrc files in `Data/Traning/MRCs` and ground truth labels in `Data/Traning/Labels`.

#### Data Requirements & Naming Convention:
- MitoXRNet requires that **raw MRC is already segmented for cell membrane/cytosolic region**, e.g. using [ACSeg](https://biomedisa.info/gallery/#) (Cytoplasm).
- Raw MRC and corresponding label must have **identical shape**, e.g. both raw mrc and corresponding label has shape `(425, 430, 410)`. Each independent raw mrc can be of different size.
- Both raw images and labels must be in **`.mrc` format**
- Raw MRC filename and its corresponding label filename **must be same** for correct mapping
- Label encoding must follow:
  - `0` → background
  - `1` → cytoplasm label (as done in [ACSeg](https://biomedisa.info/gallery/#))
  - `2` → nucleus
  - `5` → mitochondria
- Each dimension of the image should be less than 704.

#### Preprocessing: 
Here all the raw mrcs and masks are split in train and validation folders inside `Data/Training/Preprocessed` folder (i.e. new folder are created `mrc_train_preprocessed`, `mrc_val_preprocessed`, `mask_train_preprocessed` and `mask_val_preprocessed`). Each mrc and mask undergoes padding, to ensure the image size matches the model's input requirement. Each raw mrc undergoes preprocessing as proposed in MitoXRNet paper. Further it creates 3D slices for train and validation dataset in folder `Data/Training/Slices`.

Run preprocessing with default 80/20 train–validation split
```
python codes/preprocessing.py
```
Optionally, change the train/validation split ratio
```
python codes/preprocessing.py --split_ratio 0.7
```

#### Training:
Train using default configuration (Shallow U-Net: 1.4M parameters and Combined loss: BCE + Robust Dice)
```
python codes/train.py
```
Train using UNetDeep (Deeper network: ~22M parameters)
```
python codes/train.py --model_tag 1
```
Train using BCEWithLogitsLoss
```
python codes/train.py --loss_tag 1
```
- default command trains **UNet** with **CombinedLoss**
- `--model_tag 1` → trains **UNetDeep (larger network)**
- `--loss_tag 1` → uses **BCEWithLogitsLoss**
- `--model_tag 1 --loss_tag 1` → trains **UNetDeep** with **BCEWithLogitsLoss**
- `--epochs X` → trains for X number of epochs (default: 60)

Training execution will save logs, progress file and trained model weights in the `output/Trained_Weights` folder. It also save final training loss and val loss plot in `output/Plots` folder.

Note: Early stopping has been intentially removed, so stop the training based on val and train error/accuracy.

## Prediction and Evaluation

`evaluate.py` runs the complete MitoXRNet inference pipeline, including preprocessing, slice-wise prediction, and class-wise evaluation on test/prediction data.  
It supports flexible execution modes such as **preprocessing only**, **prediction only**, **evaluation only**, or the full pipeline.  
Segmentation performance is reported using IoU, Dice, Precision, and Recall for both nucleus and mitochondria.

#### Data preparation:
Keep the raw mrcs for prediction in `Data/Prediction/MRCs` folder and corresponding ground truth labels in `Data/Prediction/Labels` folder. In ground truth label atleast cytoplasm label `=1` is required for cropping the raw mrc for prediction. Nucleus and mitochondria labels can be omitted if not available.

#### Data Requirements & Naming Convention
Follow same requirements and naming convection as mention in `Training from scratch` section

Full evaluation pipeline (default)
```
python codes/evaluate.py
```
### Tag & Keyword Summary

| Flag / Mode | Description |
|------------|-------------|
| **(default)** | Runs **preprocessing → prediction → evaluation** using **UNet**, **user-trained weights**, model_name = `Trained_model_UNet_CombinedLoss`, threshold = `0.6` |
| `--pretrained 0` | Use **user-trained weights** from `output/Trained_Weights/` |
| `--model_name` | Used to select one of the user trained models from `output/Trained_Weights/` |
| `--pretrained 1` | Use **pretrained UNet with Combined loss** weights from paper `output/Pretrained_Weights/`|
| `--pretrained 2` | Use **pretrained UNetDeep with Combined loss** weights from paper `output/Pretrained_Weights/` |
| `--threshold <value>` | Set prediction & evaluation threshold (default = `0.6`) |

For evaluaton on **user-trained weights**, correct `--model_name` should be provided from the `output/Trained_Weights/`

The above mentioned code performs the following steps:
1. Preprocessing  `Data/Prediction/Preprocessed`
2. Slicing  `Data/Prediction/Slices`
3. Model load
4. Predict on each slice  `Data/Prediction/prediction_slices_temp`
5. Merged labels probabilistic: 'Data/Prediction/NucleusProbabilisticPrediction' and 'Data/Prediction/MitoProbabilisticPrediction'
6. Final predicted labels are saved in `Data/Prediction/PredictedLabels` **(at original ground-truth label sizes)**.
7. Evaluation metrics (IoU, Dice, Precision, Recall) are displayed in the terminal and saved as a JSON file in `output/Evaluation_results/`.
