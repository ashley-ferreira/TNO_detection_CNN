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
from keras.layers import Dense, BatchNormalization, Flatten, Conv3D, MaxPool3D
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

cutout_path = '/arc/projects/uvickbos/ML-MOD/new_cutouts_mar2/'

cutout_full_width = 121

####section for setting up some flags and hyperparameters
batch_size = 64 # increase with more data
dropout_rate = 0.5
test_fraction = 0.1
num_epochs = 40




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


data_pull = 'presaved'
if data_pull == 'scratch':
    check_total = 0
    for file in file_lst: 
        #print(file)
        # all get through below
        if file[9] == 'p': # temp solution, file.endswith(".measure3") and
            #print(file)
            sub_file_lst = sorted(os.listdir(cutout_path+file))

            for sub_file in sub_file_lst:
                #print(sub_file)
                if sub_file.endswith('.fits'):
                    try:
                        with fits.open(cutout_path+file+'/'+sub_file) as han:
                            img_data = han[1].data.astype('float64')
                            #img_header = han[0].header

                        print(sub_file) 
                        print(img_data.shape)
                        count +=1 
                        
                        img_data -= np.nanmedian(img_data)
                        img_data = crop_center(img_data, cutout_full_width, cutout_full_width)    
                        print(img_data.shape)
                        (aa,bb) = img_data.shape

                    except Exception as e: 
                        print(e)
                        aa, bb = 0, 0

                    
                    if aa == cutout_full_width and bb == cutout_full_width: 
                        triplet.append(img_data)
                    
                    else:
                        #null_arr = np.zeros((120,120))
                        #print(null_arr.shape)
                        #triplet.append(null_arr)
                        triplet = []
                        break
                
            if len(triplet) == 3:    
                triplet = np.array(triplet)
                #print(triplet.shape)
                label = int(sub_file[-6])
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
        #sys.exit()

    if num_good < num_bad:
        number_of_rows = bad_cutouts.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_good, replace=False)
        random_bad_cutouts = bad_cutouts[random_indices, :]
        random_good_cutouts = good_cutouts
        
        bad_labels = np.zeros(num_good)

    elif num_good > num_bad: 
        number_of_rows = good_cutouts.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_bad, replace=False)
        random_good_cutouts = good_cutouts[random_indices, :]
        random_bad_cutouts = bad_cutouts
        
        good_labels = np.ones(num_bad)

    # combine arrays 
    cutouts = np.concatenate((random_good_cutouts, random_bad_cutouts))
    cutouts = np.expand_dims(cutouts, axis=4)
    print('CNN total data shape', cutouts.shape)

    # make label array for all
    labels = np.concatenate((good_labels, bad_labels))
                
    print(str(len(cutouts)) + ' files used')
    print(len(labels))

    with open(cutout_path + 'presaved_data.pickle', 'wb+') as han:
        pickle.dump([cutouts, labels], han)


elif data_pull == 'presaved':
    with open(cutout_path + 'presaved_data.pickle', 'rb') as han:
        [cutouts, labels] = pickle.load(han) 

# REGULARIZE
cutouts = np.asarray(cutouts).astype('float32')
std = np.nanstd(cutouts)
mean = np.nanmean(cutouts)
cutouts -= mean
cutouts /= std
w_bad = np.where(np.isnan(cutouts))
cutouts[w_bad] = 0.0

with open(cutout_path + 'regularization_data.pickle', 'wb+') as han:
    pickle.dump([std, mean], han)





### now divide the cutouts array into training and testing datasets.

skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)#, random_state=41)
print(skf)
skf.split(cutouts, labels)


for train_index, test_index in skf.split(cutouts, labels):
    X_train, X_test = cutouts[train_index], cutouts[test_index]
    y_train, y_test = labels[train_index], labels[test_index]




#shift left right up down
shift = 1
x_train_l = np.copy(X_train)
x_train_l[: ,:, :, :-shift, :] = X_train[:, :, :, shift:, :]
x_train_r = np.copy(X_train)
x_train_r[: ,:, :, shift:, :] = X_train[:, :, :, :-shift, :]
x_train_u = np.copy(X_train)
x_train_u[: ,:, shift:, :, :] = X_train[:, :, :-shift, :, :]
x_train_d = np.copy(X_train)
x_train_d[: ,:, :-shift, :, :] = X_train[:, :, shift:, :, :]
# make the augmented training array 
X_train = np.concatenate([X_train, x_train_l, x_train_r, X_train, x_train_u, x_train_d])
#duplicate the labels array
y_train = np.concatenate([y_train, y_train, y_train, y_train, y_train, y_train])




### define the CNN
# below is a network I used for KBO classification from image data.
# you'll need to modify this to use 2D convolutions, rather than 3D.
# the Maxpool lines will also need to use axa kernels rather than axaxa
def convnet_model(input_shape, training_labels, unique_labs, dropout_rate=dropout_rate):
    print('CNN input shape', input_shape)

    unique_labs = len(np.unique(training_labels))

    model = Sequential()

    #hidden layer 1
    model.add(Conv3D(filters=16, kernel_size=(1, 3, 3), input_shape=input_shape, activation='relu', padding='valid'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='valid'))

    #hidden layer 2 with Pooling
    model.add(Conv3D(filters=16, kernel_size=(1, 3, 3), input_shape=input_shape, activation='relu', padding='valid'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='valid'))

    model.add(BatchNormalization())

    #hidden layer 3 with Pooling
    model.add(Conv3D(filters=16, kernel_size=(1, 3, 3), input_shape=input_shape, activation='relu', padding='valid'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(3, 4, 4), padding='valid')) # just for this last maxpool, pool_size = ()

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid')) 
    model.add(Dense(unique_labs, activation='softmax'))

    return model

unique_labels = 2
y_train_binary = keras.utils.np_utils.to_categorical(y_train, unique_labels)
y_test_binary = keras.utils.np_utils.to_categorical(y_test, unique_labels)


print('training input shape (X_train.shape[1:])', X_train.shape[1:])
print('model fit input shape (X_train.shape)', X_train.shape)

### train the model!
cn_model = convnet_model(X_train.shape[1:], training_labels = y_train_binary, unique_labs=unique_labels)
cn_model.summary()

cn_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])

start = time.time()

classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

end = time.time()
print('Process completed in', round(end-start, 2), ' seconds')

# save trained model 
cn_model.save(cutout_path + 'model_' + str(end))

### get the model output classifications for the train set
preds_train = cn_model.predict(X_train, verbose=1)
#preds_test = cn_model.predict(X_test, verbose=1)
#help(cn_model.evaluate)
eval_test = cn_model.evaluate(X_test, y_test_binary, batch_size=batch_size, verbose=1)
print("test loss, test acc:", eval_test)

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

c = 0.5
X_train = np.squeeze(X_train, axis=4)

# plot test and train ones that don't agree with labels
for i in range(len(preds_train)):
    triplet_Xtrain = X_train[i]
    #print(triplet_Xtrain.shape)
    num = 0
    for t in triplet_Xtrain:
        num +=1
        #print(t.shape)

        if y_train[i] == 0 and preds_train[i][1] > c: # check confidence index
            
            (c1, c2) = zscale.get_limits(t)
            normer = interval.ManualInterval(c1,c2)
            pyl.title('labeled no TNO, predicted TNO at conf=' + str(preds_train[i][1]) + 'triplet' + str(num))
            pyl.imshow(normer(t))
            pyl.show()
            pyl.close()

        if y_train[i] == 1 and preds_train[i][0] > c:
            (c1, c2) = zscale.get_limits(t)
            normer = interval.ManualInterval(c1,c2)
            pyl.title('labeled TNO, predicted no TNO at conf=' + str(preds_train[i][1]) + 'triplet' + str(num))
            pyl.imshow(normer(t))
            pyl.show()
            pyl.close()


# save triplet names and laod in ds9 to blink through