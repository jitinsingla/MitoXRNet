# MitoXRNet

--> Below is the sequence wise pipeline for training and predictions.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Each 3D Image shape along any axis should be <=704. 
   
2. Make sure your files are in .mrc format and inside Data/User_Raw_Data accordingly.

3. Use "Preprocessing.ipynb" to generate preprocessed images.

4. Create Slices for training, validation and test cells using "create_slices.ipynb"
   
6. Start the training using "Train.ipynb" , this will save best model, log files of training along with live tensorboard info.: You can use U-Net or U-NetDeep defined in 'pytorch_network.py' and other Loss functions.

7. Then do predictions on test cell slices using "Test.ipynb" to get Mito, Nucleus Predictions.

8. Merge the predicted slcies using "MergeSlices.ipynb" this will give two seperate folders for Merged Nucleus and Merged Mitochondria Cell predictions.

9. Now merge the Neucleus and Mitochondria into whole cell predictions using thresholding code in "Test.ipynb".

10. Can evaluate predictions scores like IOU, DICE, Precision, Recall, F1-score using functions defined in "utils.py" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
