# RUN IN LSST TERMINAL 

'''
import os
import time
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyl
import matplotlib.gridspec as gridspec
import pickle
import tensorflow as tf
import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv3D, Conv2D, MaxPool3D
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import interval
from trippy import tzscale
from trippy.trippy_utils import expand2d, downSample2d
import glob
''' 
import os
from os import path
import time
from datetime import date 
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as pyl
import pickle
import heapq

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

from astropy.visualization import interval, ZScaleInterval
from astropy.io import fits
zscale = ZScaleInterval()

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(432)

cutout_path = '/arc/projects/uvickbos/ML-MOD/140_pix_cutouts/'


####section for setting up some flags and hyperparameters
batch_size = 16
dropout_rate = 0.2
test_fraction = 0.05


####
#here you'll need to read in source the cutout data.
#the array should be of the shape
#   [n, x, y, 1]
#
# the easiest way to do this is to create an array like:
# cutouts = []
# for i in range(len(files)):
#     cutouts.append(data[i])
# cutouts = np.array(cutouts)
# cutouts = np.expand_dims(cutouts, axis=3)
#
# you might already have a cutouts array of shape (n,x,y) and so only the last
# line might be necessary
#
# the cutouts array needs to include cutouts for both good and bad sources.

def crop_center(img, cropx, cropy):
    '''
    Crops input image array around center to desired (cropx, cropx) size
    
    Taken from stack overflow: 
    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    Parameters:    
        img (arr): image to be cropped
        cropx (int): full width of desired cutout
        cropy (int): full height of desired cutout
    Returns:
        
        cropped_img (arr): cropped image
    '''
    x,y = img.shape 
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    cropped_img = img[int(starty):int(starty+cropy), int(startx):int(startx+cropx)]

    return cropped_img

file_lst = sorted(os.listdir(cutout_path))#.sort()

good_cutouts = []
bad_cutouts = []

good_labels = []
bad_labels = []
# for each triplet
triplet = []
count = 0


check_total = 0
for file in file_lst: 
    with fits.open(cutout_path+file) as han:
        img_data = han[1].data.astype('float64')
        img_header = han[0].header

    count +=1   
    #print(count)   
    img_data = crop_center(img_data, 120, 120) # skip smaller ones, makes it smaller?    
    triplet.append(img_data)
    print(img_data.shape)
    if count == 3: # how does it not hit 3? does
        label = int(file[-6])
        if label == 1:
            good_cutouts.append(triplet)
            good_labels.append(1) # can do after too
        elif label == 0:
            bad_cutouts.append(triplet)
            bad_labels.append(0) 

        check_total +=1 
        triplet = []
        count = 0
        #print(check_total) 

good_labels = np.array(good_labels)
good_cutouts = np.array(good_cutouts, dtype=object)
print(good_cutouts.shape)

bad_labels = np.array(bad_labels)
bad_cutouts = np.array(bad_cutouts, dtype=object)
print(bad_cutouts.shape)

num_good = len(good_cutouts)
num_bad = len(bad_cutouts)
print(num_good, 'good cutouts')
print(num_bad, 'bad cutouts')

if num_good > num_bad: # equalize either way
    print('more good cutouts than bad')
    sys.exit()

'''
for file in file_lst: # make sure gets sorted with 3?
    #print(file) # assuming 3 in a row, can put in dirs
    # load image data
    with fits.open(cutout_path+file) as han:
        img_data = han[1].data.astype('float64')
        img_header = han[0].header

    (aa,bb) = img_data.shape
    print(aa, bb)
    if aa > 240 and bb > 240:  
        count +=1      
        img_data = crop_center(img_data, 240/2, 240/2)    
        triplet.append(img_data)

        if count == 3:
            triplet = []
            count = 0
            cutouts.append(triplet)
            label = file[-6]
            #print(label)
            labels.append(label)
            check_total +=1 
            print(check_total) # doesn't hit this?
    else:
        continue # skip rest? somehow

labels = np.array(labels)
cutouts = np.array(cutouts)
print(cutouts.shape) # more than 3 sometimes

#cutouts = np.expand_dims(cutouts, axis = 4)



### create a labels array with length n , with 1==good star, and 0==else

# 50/50 SPLIT

# REMOVE BACKGROUND

# REGULARIZE


### now divide the cutouts array into training and testing datasets.

skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)#, random_state=41)
print(skf)
skf.split(cutouts, labels)


for train_index, test_index in skf.split(cutouts, labels):
    X_train, X_test = cutouts[train_index], cutouts[test_index]
    y_train, y_test = labels[train_index], labels[test_index]



### define the CNN
# below is a network I used for KBO classification from image data.
# you'll need to modify this to use 2D convolutions, rather than 3D.
# the Maxpool lines will also need to use axa kernels rather than axaxa
def convnet_model(input_shape, training_labels, dropout_rate=dropout_rate):

    unique_labs = len(np.unique(training_labels))

    model = Sequential()

    #hidden layer 1
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid'))

    #hidden layer 2 with Pooling
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid'))

    model.add(BatchNormalization())

    #hidden layer 3 with Pooling
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(unique_labs, activation='softmax'))

    return model



### train the model!
cn_model = convnet_model(X_train.shape[1:], y_train)
cn_model.summary()

cn_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])


checkpointer = ModelCheckpoint('keras_convnet_model.h5', verbose=1)
early_stopper = EarlyStopping(monitor='loss', patience=2, verbose=1)

start = time.time()


classifier = cn_model.fit(X_train, y_train, epochs=80, batch_size=batch_size, callbacks=[checkpointer])

end = time.time()
print('Process completed in', round(end-start, 2), ' seconds')

"""
Plot accuracy/loss versus epoch
"""

fig = pyl.figure(figsize=(10,3))

ax1 = pyl.subplot(121)
ax1.plot(classifier.history['accuracy'], color='darkslategray', linewidth=2)
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')

ax2 = pyl.subplot(122)
ax2.plot(classifier.history['loss'], color='crimson', linewidth=2)
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')

pyl.show()
pyl.close()



### get the model output classifications for the train and test sets
preds_test = cn_model.predict(X_test, verbose=1)

preds_train = cn_model.predict(X_train, verbose=1)
'''