# Daniel McKirgan
# CNN Plant Image Prediction Model
# This model was created to categorize different parts of a plant image by taking in pixel data with an assigned
# categorical value. This file contains the code to read in a metadata file in the form of a .csv or .tsv as well as the
# pixel images. It will assign both of those to an array and split them into testing and training groups. It will then
# feed the data through the model and give the final accuracy with a graph of the accuracy across the epochs and the
# cross-entropy loss.

# imports
# Packages used to create model
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras_preprocessing import image
# Packages to set arrays and split data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Plotting packages
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
# Misc
import sys
from tqdm import tqdm

# read in csv file, in this case I had a tsv so the sep='\t' will tell the function the file is tab separated.
# Line 32 should be the column name that you put the concatenated Sample Number and Timestamp (if you did not read the README the instructions
# for that are in there). We want the column to be read in as a string because if not the program will think it is a float and will leave off
# any trailing 0's. These 0's are important later when the program is trying to iterate through yout metadata file and match that to the pixel data.

train = pd.read_csv('Filepath to metadata/categorical data: Project/metadata/...', sep='\t',
                    dtype={'Your Concatenated Sample Number': str})

# Print train to ensure the data is correctly formatted
print(train)
print(train.columns)

train_image = []

# Iterate through image file to extract the image data
for i in tqdm(range(train.shape[0])):
    img = image.load_img(
        r'Filepath to image folder: Project/Images/...' + str(train['(Same from line 32'][i]) +
        '.png', target_size=(100, 100, 3))  # target size smaller than 250x250 for quicker processing, but this will
    # change the size they are read in at
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)

# Assign values to images and split images into test and train groups
X = np.array(train_image)

# Here we want to get rid of any category in the metadata file that will not be a categorical output, for example
# "Sample Number" is not a value we want output from the model, so we will drop it. This step is much easier if you have cleaned up your excel data
y = np.array(train.drop({'Any categorical value in metadata file that is not an output'}, axis=1))

# Splitting the dataset into training and testing, this will be done randomly every time the program is run
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# Define Model, this is the model used for the initial dataset, this will require some tweaking depending on your
# dataset
def model_definition():
    model = Sequential()
    # 1st Group w/ Max Pooling and Dropout at 20%
    model.add(keras.Input(shape=(100, 100, 3)))
    model.add(Conv2D(32, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # 2nd Group w/ Max Pooling and Dropout at 30%
    model.add(Conv2D(64, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # 3rd Group w/ Max Pooling and Dropout at 40%
    model.add(Conv2D(128, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Transform model into 1D array and pass into Dense
    model.add(Flatten())

    model.add(Dense(6, activation='sigmoid'))
    model.summary()
    # We use the Adam optimizer and binary_crossentropy as the loss function to categorize the pixels
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Plot the Cross Entropy Loss and Classification Accuracy
# Creates 2 graphs, one plotting the accuracy and one plotting the loss on the Validation sets
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='red', label='train')
    pyplot.plot(history.history['val_accuracy'], color='green', label='test')
    # save plot to file
    plt.subplot_tool()
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# This is the method that tells our model how to run, 35 epochs means the data will run through the model 35 times,
# each batch of pixels used is 64, the validation is our test sets we made on Line 50, verbose will show us each
# individual step, and steps_per_epoch is how many times the weights and biases are getting updated during the epoch.
# It is common practice for the steps_per_epoch to be (size of array/ batch_size), in this case that was 88.
def run_test_harness():
    model = model_definition()
    steps = int(X_train.shape[0] / 64)
    history = model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_test, y_test), verbose=1,
                        steps_per_epoch=steps)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('>%.3f' % (acc * 100))
    summarize_diagnostics(history)


# run the model
# This will run the previous function and train the dataset
run_test_harness()
