# Daniel McKirgan
# CNN Plant Image Prediction Model
# This model was created to categorize different parts of a plant image by taking in pixel data with an assigned
# categorical value. This file contains the code to read in a metadata file in the form of a .csv or .tsv as well as the
# pixel images. It will assign both of those to an array and split them into testing and training groups. It will then
# feed the data through the model and give the final accuracy with a graph of the accuracy across the epochs and the
# cross-entropy loss.

import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras_preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as pyplot
import sys

train = pd.read_csv(
    r'C:\Your Project Path\ ... \sampleMetadata_.txt', sep='\t', dtype={'True_Samp_Num': str})
print(train)
print(train.columns)
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(
        r'C:\Your Project path\... \overview.' + str(train['True_Samp_Num'][i]) +
        '.png', target_size=(100, 100, 3)) #resized to decrease runtime
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)

X = np.array(train_image)
y = np.array(train.drop({'True_Samp_Num', 'Sample Number', 'Timestamp', 'AdjSampNum'}, axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# Define Model
def model_definition():
    model = Sequential()
    # 1st Group w/ Max Pooling and Batch Normalization
    model.add(keras.Input(shape=(100, 100, 3)))
    model.add(Conv2D(32, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # 2nd Group w/ Max Pooling, Batch Normalization, and Dropout at 20%
    model.add(Conv2D(64, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # 3rd Group w/ Max Pooling, Batch Normalization, and Dropout at 30%
    model.add(Conv2D(128, activation='relu', kernel_size=(5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Transform model into 1D array and pass into Dense functions
    model.add(Flatten())
    model.add(Dense(6, activation='sigmoid'))
    model.summary()
    # Adam optimizer, binary_crossentropy for column selection(each value in each column contains either a 1 or 0),
    # accuracy as metric to assess how the model is running
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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


def run_test_harness():
    model = model_definition()
    steps = int(X_train.shape[0] / 64)
    history = model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_test, y_test), verbose=1,
                        steps_per_epoch=steps)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('>%.3f' % (acc * 100))
    summarize_diagnostics(history)
    model.save('base_model.h5')


run_test_harness()
