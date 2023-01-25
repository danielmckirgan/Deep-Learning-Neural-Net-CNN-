# Deep Learning Neural Net CNN
A Deep Learning Neural Net that works to classify pixels in an image.

The original project for this code was used in creting a plant image classification model.

# Setting Up 
There is a .jar file in the repository that acts as a pixel data collector and data classifier. Downloaded images can be loaded into the .jar file where they will be displayed on the dashboard. If you wish to use this file, follow these set-up instructions: 

To run this program, start by setting up the following directories (you can name the home folder whatever you want!):

Home

      Attributes 
      
      Characters
      
      Images

In the home/Attributes folder, put a copy of the attached Attributes.txt file.

In the home/Characters folder, put a copy of the attached characters.txt file.

In the home/Images folder, put all the images you downloaded

Once you've got these folders set up, you can run the java program. Start by downloading the attached DNN_ImageSampler-1.0-SNAPSHOT.txt file. Change the file name extension from .txt to .jar (Towson outlook wasn't allowing me to attach a jar file! It is not a text file, so if you try to read the file before downloading, it will look like a mess. Changing the file name extension is just a trick to get outlook to accept the file as an attachment).  Once you've changed the name, try double-clicking the jar file icon. It may open on its own! If not, open a command prompt window and use the command 'cd' to change to the directory where the jar file is located. Then on the command line, type:

java -jar DNN_ImageSampler-1.0-SNAPSHOT.jar

If it says that java was not found, install the java jdk by going to https://www.oracle.com/java/technologies/downloads/#jdk19-windows. Once you've got java installed and run the above java command, a window will come up asking for a project name, your name and the root directory. Use the button to find your above home directory. This is the "root" directory for the project. 

Once the .jar file is up and running, choose the root directory that you have named in place of "Home" and fill the Name and Project Name and press "OK"

This will start the program and you can get to collecting the pixel data.

# Collecting Data

The program should look something like this:
![Image Sampler](https://user-images.githubusercontent.com/90268829/214688629-e3dc2039-e74e-4519-9332-86c7041d8e17.png)

But (hopefully) with one of your downloaded pictures loaded in.

From this point you are free to adjust the parameters to your liking, however I will show you some of the more important features:

This section is directly imported from the characters.txt file you downloaded. The characters in that file act as your categorization types

![Image Sampler_Character Emphasis](https://user-images.githubusercontent.com/90268829/214690418-640be41a-9375-4ec5-bb3a-10132b5605a5.png)

The characters are important to your overall model because they are what will be the model output. In the model program we will remove any other data collected from the image sampler.

The middle section are close-ups of the pixel you have chosen to characterize. The top box shows the area around the pixel you have selected and the lower box is the exact pixel. Once a pixel has been selected, it can be categorized into whichever categories you desire just be sure to be as accurate as possible so your data is not false.


![Image Sampler_Pixel_and_Area_Emphasis](https://user-images.githubusercontent.com/90268829/214692169-2eaba339-96d2-4a3c-b2c4-5691f48ed69b.png)

Finally, I will discuss the dropdown menu labeled "Parameters". After clicking this there are three options, choose set sampling parameters and it will pull up a screen that looks like this:

![Image_Sampler_Parameters](https://user-images.githubusercontent.com/90268829/214693158-8dc682d5-2bee-46ce-9e11-daefc1aac4f8.png)


After all the parameters have been set you can get to sampling your images! Below I have posted a screen recording of doing some sampling, showing how the program works and its functionalities.

https://user-images.githubusercontent.com/90268829/214700253-2ec635bf-de3d-45d9-9644-7ebbbc9bf8b1.mp4


# Conclusion

For the original project I sampled around 10,000 pixels from about 1,500 images that I downloaded. The bigger the sample size the better off your model will perform (most of the time). For image classification, it is good to have many different samples from different pictures, this will allow the model to get a better understanding of the types of images it may contact after being deployed. For most projects I would suggest more than 5,000 pixels from several different images.

For more on how the code works, open the .py file in the main section of this repository. The file has the code for loading and prepping the data, the model itself, and some visualization code to help analyze which hyperparameters are needed. There are also extensive comments explaining what the lines of code mean and what should be input in place of pseudocode. Most of the replacements will come from your personal computer's directories and the metadata file that will be created from the image sampler.
