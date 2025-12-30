# 🧬 MitoXRNet 

## Prerequisties / Environment Setup
Anaconda install ([link](https://docs.anaconda.com/anaconda/install/))

Create environment (recommended):
```
conda env create -f env/environment.yml
conda activate sxt_seg
```
if using env file give error, just ensure all the following packages are installed:
gc, cupy, pytorch, numpy, pandas, tenorflow, shutil, matplotlib.

## MitoXRNet Usage
User can train the model from scratch or use the pretrained weights from the paper to directly predict on their dataset.
For training from sratch follow the instructions mentioned under "Training from Scratch" and then predict using "Prediction" instructions.
For directly predicting using the pretrained models, skip training instructions and directly go to "Prediction" section.

## Training from Scratch
#### Data preparation:
Folder: Data/Training
Data in this folder is used for training the model from scratch. The data in this folder will be split into training and validation sets in x, y %
1. To prepare the data for training, copy raw mrc files in Data/Traning/MRCs and ground truth labels in Data/Traning/Labels.
Notes about data preparation:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg ([link](https://biomedisa.info/gallery/#))
Size example (mrc.shape == label.shape)
Size should be in order for both MRC and Label image (z,y,x) = (z,y,x)
Make sure both raw MRCs and Labels are in .mrc format.
Name should be same of both MRC file and corresponding Label file for correct mapping.
Name should be in a specific format only.
<EXPERIMENT_METADATA>_<CELLID>_pre_rec.mrc
Example CELLID:- 1111_13, 1128_1-2
Example FullName:- KLW_PBC_INS1e_Ex-4_5min_1111_13_pre_rec.mrc
Labels of Nucleus = 2 , Labels of Mitochondria = 5 , Labels of Cytoplasm = 1 and rest all 0
Each 3D Image shape along any axis should be <=704. 

Steps:
#### Preprocessing: 
Here all the raw mrcs and masks are split in train and validation folders inside Data/Training folder (i.e. mrc_train, mrc_val, mask_train and mask_val)
Each mrc and mask undergoes padding, to ensure the image size matches the model's input requirement.
Each raw mrc undergoes preprocessing as proposed in MitoXRNet paper.
Further it creates 3D slices for train and validation dataset in folder Data/Training/slices.

```bash
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

training execution will create logs and trained model weights in the output/trainingWeights folder
Early stopping has been intentially removed, so stop the training based on val and train error/accuracy

## Predict
#### Data preparation:
Keep the raw mrcs for prediction in Data/Prediction/MRCs folder
Notes:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg (link)
Each 3D Image shape along any axis should be <=704.
make sure raw MRCs are in .mrc format

#### Prediction
codes/predict.py --pretrained 1 (small) 2 (large) 0 (user trained)
0: If user tained the model and weights have been saved in output/trainingWeights folder
1/2: For pretrained model weights reported in paper, 1 for smaller network (1.4 Million paramters) and 2 for ....
The predict.py code performs the folloing steps:
1. Preprocessing (sved in output/Prediction/mrc_predict_preprocessed)
2. Slicing (output/Prediction/mrc_predict_slices)
3. Model load
4. Predict on each slice (output/Prediction/predictedLabels_Slice_temp)
5. Merge slices

Final prdicted labels are saved in **output/Prediction/predictedLabels** (give anti-padded sizes)


#### Evaluation
User can evaluate the predicted labels against the user provided labels.
keep the user provided labels in Data/Prediction/Labels (make sure the raw mrcs and labels follow critetria as metioned under Notes in data preparation of training from scratch setion)
run codes/evaluate.py
Code will evaluate predictions scores like IOU, DICE, Precision, Recall, F1-score

