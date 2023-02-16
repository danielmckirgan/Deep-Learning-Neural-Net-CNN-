# Deep Learning Neural Net (Convolutional Neural Net)
A Deep Learning Neural Net that works to classify pixels in an image.

The original project for this code was used in creating a plant image classification model.

# Image Folder

To use the original dataset for this project download the Window.zip file. This contains the original dataset used along with the appropriate metadata. Download both of those to the same folder, this will be used as your directory.

# The Model
The .py file contains the code to load in the metadata and image folder, then it splits them into training and testing groups and feeds the training group into the model. Once the model completes its calculations it will output a final percentage of accuarcy as well as save two graphs to your directory that outline the accuracy and the loss.

# Conclusion

For the original project I sampled around 10,000 pixels from about 1,500 images that I downloaded. The bigger the sample size the better off your model will perform (most of the time). For image classification, it is good to have many different samples from different pictures, this will allow the model to get a better understanding of the types of images it may contact after being deployed. For most projects I would suggest more than 5,000 pixels from several different images.

For more on how the code works, open the .py file in the main section of this repository. The file has the code for loading and prepping the data, the model itself, and some visualization code to help analyze which hyperparameters are needed. There are also extensive comments explaining what the lines of code mean and what should be input in place of pseudocode. Most of the replacements will come from your personal computer's directories and the metadata file that will be created from the image sampler.
