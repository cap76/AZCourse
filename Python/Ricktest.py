from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from numpy import genfromtxt
import numpy as np
import pandas as pd
import glob
import os
import sys
import copy
import scipy as sc
import imageio
# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

batch_size = 128
num_classes = 2
epochs = 100

# input image dimensions
img_rows, img_cols, img_chans = 90, 160, 3

fil1=sorted(glob.glob('/Users/christopherpenfold/Desktop/AZMachineLearning/intro-machine-learning/data/RickandMorty/data/AllRickImages/*.jpg'))
fil2=sorted(glob.glob('/Users/christopherpenfold/Desktop/AZMachineLearning/intro-machine-learning/data/RickandMorty/data/AllMortyImages/*.jpg'))

X = np.zeros((np.shape(fil1)[0]+np.shape(fil2)[0],90,160,3))
Y = np.zeros((np.shape(fil1)[0]+np.shape(fil2)[0],2))


#Load in the datasets
for k in range(0, np.shape(fil1)[0]):
      #X[k,:,:,:] = sc.ndimage.imread(fil1[k])/255
      X[k,:,:,:] = imageio.imread(fil1[k])/255
      Y[k,0] = 1

for k in range(0, np.shape(fil2)[0]):
      #X[k+np.shape(fil1)[0],:,:,:] = sc.ndimage.imread(fil2[k])/255
      X[k+np.shape(fil1)[0],:,:,:] = imageio.imread(fil2[k])/255
      Y[k+np.shape(fil1)[0],1] = 1

#Here is a simple CNN
visible1 = Input(shape=(90,160,3))
conv1 = Conv2D(20, kernel_size=(5,5), activation='relu')(visible1)
pool1 = MaxPooling2D(pool_size=(3,3))(conv1)
conv2 = Conv2D(20, kernel_size=(5,5), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(3,3))(conv2)
conv3 = Conv2D(64, kernel_size=(5,5), activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(3,3))(conv3)
flat1 = Flatten()(pool3)

conv1a = Conv2D(20, kernel_size=(10,10), activation='relu')(visible1)
pool1a = MaxPooling2D(pool_size=(3,3))(conv1a)
conv2a = Conv2D(20, kernel_size=(10,10), activation='relu')(pool1a)
pool2a = MaxPooling2D(pool_size=(3,3))(conv2a)
#conv3a = Conv2D(64, kernel_size=(10,10), activation='relu')(pool2a)
#pool3a = MaxPooling2D(pool_size=(3,3))(conv3a)
flat1a = Flatten()(pool2a)

merge = concatenate([flat1,flat1a])

hidden1 = Dense(100, activation='relu')(merge)
drop1 = Dropout(0.6)(hidden1)
output1 = Dense(2)(drop1)
output  = Activation('sigmoid')(output1)
model = Model(inputs=[visible1], outputs=output)

#get training and test
randno = np.random.uniform(0,1,(np.shape(Y)[0],))
trainset = (randno>=0.25)
testset = (randno<0.25)

#Checkpoints
model_checkpoint_callback = [keras.callbacks.ModelCheckpoint(filepath='./Models/CNN',save_weights_only=False,monitor='val_acc',mode='max',save_best_only=True)]
#Compile and run
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer = sgd, metrics=['accuracy'])
model.fit([X[trainset,:,:]], Y[trainset,:], validation_data=([X[testset,:,:]], Y[testset,:]), epochs=30, batch_size=500,callbacks=model_checkpoint_callback) #,class_weight = class_weight,callbacks=callbacks)

model = keras.models.load_model('./Models/CNN') 
predictions = model.predict([X], batch_size=1000)


#Now let's get some TPs
TPs = np.where( (predictions[:,0]>predictions[:,1]) & (Y[:,0]==1) )
FPs = np.where( (predictions[:,0]>predictions[:,1]) & (Y[:,0]==0) )
TNs = np.where( (predictions[:,0]<predictions[:,1]) & (Y[:,0]==0) )

with DeepExplain(session=K.get_session()) as de:  
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    target_tensor = fModel(input_tensor)
    xs = X[(predictions[:,0]>predictions[:,1]) & (Y[:,0]==1),:,:,:]
    ys = Y[(predictions[:,0]>predictions[:,1]) & (Y[:,0]==1),:]
    attributions = de.explain('grad*input', target_tensor * ys, input_tensor, xs)

import matplotlib
import matplotlib.pyplot as plt

#Generate activations for TP
for i, a in enumerate(attributions): #attributions: #range(0, 100): #enumerate(attributions):
    plt.figure()
    #filename = "./Plots/Rick_%d_%d_%d_%d.pdf" % (i,ys[i,0],ys[i,1],np.sign(predictions[i,0]))
    filename = "./Plots/Rick_TP_%d.pdf" % i
    plt.subplot(1, 2, 1)
    plt.imshow(xs[i], cmap='hot', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(a)
    plt.savefig(filename)
    plt.close()

#Generate activations for FPs
with DeepExplain(session=K.get_session()) as de:
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    target_tensor = fModel(input_tensor)
    xs2 = X[(predictions[:,0]>predictions[:,1]) & (Y[:,0]==0),:,:,:]
    ys2 = Y[(predictions[:,0]>predictions[:,1]) & (Y[:,0]==0),:]
    attributions2 = de.explain('grad*input', target_tensor * ys2, input_tensor, xs2)

for i, a in enumerate(attributions2): #attributions[0:100]: #range(0, 100): #enumerate(attributions):
    plt.figure()
    filename = "./Plots/Rick_FP_%d.pdf" % i
    plt.subplot(1, 2, 1)
    plt.imshow(xs2[i], cmap='hot', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(a)
    plt.savefig(filename)
