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
Data/TrainingData
Data in this folder is used for traingin the model from mstratch. The data in this flder will besplit into trining, val and test in x, y, z %
1. To prepare the data for training copy raw mrc files in Data/TraningData/MRCs and ground truth labels in Data/TraningData/Labels
Notes about data preparatiion:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg (link)
size example (mrc==label)
name same ho chahiye mrc and label ka, for mapping
labels of nucleus ==2 and labesl of mito ==5 and rest all 0
Each 3D Image shape along any axis should be <=704. 
make sure both raw MRCs and Labels are in .mrc format

Steps:
#### Preprocessing: 
Here all the raw mrcs and masks are split in train, test and val folders inside Data/TrainingData folder.
Each mrc and mask undergoes padding, to ensure the image size mat hes model requirement.
Each raw mrc undergoes preprocessing as proposed in MitoXRNet paper.
Further it creates 3D slices for all the train, test and val dataset.

run:
python codes/preprocessing.py

#### Training:
python codes/train.py
training execution will create logs and trained model weights in the "output" folder
Early stopping has been intentially removed, so stop the training based on val and train error/accuracy

#### Testing:
python codes/test.py
test.py predicts on the test slices and merge them in single cell predicted labels.
The test prediction labels are saved in output/testPredictions

Start the training using "Train.ipynb" , this will save best model, log files of training along with live tensorboard info.: You can use U-Net or U-NetDeep defined in 'pytorch_network.py' and other Loss functions.
7. Then do predictions on test cell slices using "Test.ipynb" to get Mito, Nucleus Predictions.
8. Merge the predicted slcies using "MergeSlices.ipynb" this will give two seperate folders for Merged Nucleus and Merged Mitochondria Cell predictions.
9. Now merge the Neucleus and Mitochondria into whole cell predictions using thresholding code in "Test.ipynb".
10. Can evaluate predictions scores like IOU, DICE, Precision, Recall, F1-score using functions defined in "utils.py" 


## Directly Predict using MitoXRNet (Trained models from paper)
#### Data preparation:
Keep the raw mrcs for prediction in Data/PredictionData folder
Notes about data preparatiion:
MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg (link)
Each 3D Image shape along any axis should be <=704.
make sure raw MRCs are in .mrc format

#### Prediction
codes/predict.py --pretrained 1 (small) 2 (large) 0 (user trained)
1. Preprocessing
2. Slicing
3. Model load
4. Predict on each slice
5. Merge slices

Final prdicted labels are saved in output/
