# MitoXRNet

## Prerequisties / Environment Setup
Anaconda install (link on how to install)
create new environment using envirnemnt file: command: ./env/enviro,
if using env file give error, just ensure all the following packages are installed:
gc, cupy, pytirch, numpy, teno

## Data preparation:
### Training
Data/TrainingData
Data in this folder is used for traingin the model from mstratch. The data in this flder will besplit into trining, val and test in x, y, z %
1. To prepare the data for training copy raw mrc files in Data/TraningData/MRCs and ground truth labels in Data/TraningData/Labels
2. MitoRNet requires that idivudal cell is masked in raw MRC files (like using ACSeg (link)
Note: size example (mrc==label)
labels of nucleus ==2 and labesl of mito ==5 and rest all 0

1. Each 3D Image shape along any axis should be <=704. 
2. Make sure your files are in .mrc format and inside Data/User_Raw_Data accordingly.
3. Use "Preprocessing.ipynb" to generate preprocessed images.
4. Create Slices for training, validation and test cells using "create_slices.ipynb"
6. Start the training using "Train.ipynb" , this will save best model, log files of training along with live tensorboard info.: You can use U-Net or U-NetDeep defined in 'pytorch_network.py' and other Loss functions.
7. Then do predictions on test cell slices using "Test.ipynb" to get Mito, Nucleus Predictions.
8. Merge the predicted slcies using "MergeSlices.ipynb" this will give two seperate folders for Merged Nucleus and Merged Mitochondria Cell predictions.
9. Now merge the Neucleus and Mitochondria into whole cell predictions using thresholding code in "Test.ipynb".
10. Can evaluate predictions scores like IOU, DICE, Precision, Recall, F1-score using functions defined in "utils.py" 


### Prediction
Data/PredictionData
--> Below is the sequence wise pipeline for training and predictions.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
