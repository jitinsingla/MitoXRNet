# MitoXRNet

## Prerequisties / Environment Setup
Anaconda install (link on how to install)
create new environment using envirnemnt file: command: ./env/enviro,
if using env file give error, just ensure all the following packages are installed:
gc, cupy, pytirch, numpy, teno

## MitoXRNet Usage
User can train the model from scratch or use the pretrained weights from the paper to directly predict on their dataset.
For training from sratch follow the instruictions mentioned under "Trainign from Scratch" and then predict using "Prediction" instructions
For directlyu predicting using the pretrained mdoels, skip trining instrcutiuons and directly go to "Prediction" section.

## Training from Scratch
#### Data preparation:
Data/Training
Data in this folder is used for traingin the model from mstratch. The data in this flder will besplit into trining and val in x, y %
1. To prepare the data for training copy raw mrc files in Data/Traning/MRCs and ground truth labels in Data/Traning/Labels
Notes about data preparatiion:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg (link)
size example (mrc==label)
name same ho chahiye mrc and label ka, for mapping
labels of nucleus ==2 and labesl of mito ==5 and rest all 0
Each 3D Image shape along any axis should be <=704. 
make sure both raw MRCs and Labels are in .mrc format

Steps:
#### Preprocessing: 
Here all the raw mrcs and masks are split in train and val folders inside Data/Training folder (i.e. mrc_train, mrc_val, mask_train and mask_val)
Each mrc and mask undergoes padding, to ensure the image size mat hes model requirement.
Each raw mrc undergoes preprocessing as proposed in MitoXRNet paper.
Further it creates 3D slices for train and val dataset in folder Data/Training/slices.

run:
python codes/preprocessing.py

#### Training:
python codes/train.py --modelSize 0 (small), 1 (large UNetDeep)
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

