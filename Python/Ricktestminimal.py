#import tempfile, sys, os
#sys.path.insert(0, os.path.abspath('..'))
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, Dropout
from keras import optimizers
import numpy as np

#Load demo
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X = np.zeros((np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2],1))
Xt = np.zeros((np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2],1))
X[:,:,:,0] = x_train[:,:,:]
Xt[:,:,:,0] = x_test

#Here is a simple CNN
visible1 = Input(shape=(28,28,1))
conv1 = Conv2D(5, kernel_size=(2,2), activation='relu')(visible1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
flat1 = Flatten()(pool1)
hidden1 = Dense(50, activation='relu')(flat1)
hidden2 = Dense(1)(flat1)
output  = Activation('sigmoid')(hidden2)
model = Model(inputs=[visible1], outputs=output)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer = sgd, metrics=['accuracy'])
model.fit(X, y_train, validation_data=(Xt, y_test), epochs=2, batch_size=500)
