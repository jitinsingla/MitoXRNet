# MitoXRNet



--> Below is the sequence wise pipeline for training and predictions.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Make sure your files are in .mrc format

2. Use "Preprocessing.ipynb" to generate preprocessed image

3. Seperate preprocessed cells into 6 folders trainMrc, testMrc, valMrc, trainMask, testMask, valMask.

4. Create Slices for training, validation and test cells using "create_slices.ipynb"

5. Start the training using "Train.ipynb" , this will save best model, log files of training along with live tensorboard info.: You can use U-Net or U-NetDeep defined in 'pytorch_network.py' and other        Loss functions.

6. Then do predictions on test cell slices using "Test.ipynb"

7. Merge the predicted slcies using "MergeSlices.ipynb" this will give two seperate folders for Merged Nucleus and Merged Mitochondria Cell predictions

8. Now merge the Neucleus and Mitochondria whole predictions using thresholding code in "Test.ipynb".

9. Can evaluate predictions scores like IOU, DICE, Precision, Recall, F1-score using functions defined in "utils.py"

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
