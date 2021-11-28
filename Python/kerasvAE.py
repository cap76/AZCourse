import numpy as np
import time
import sys
import os


from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from keras.callbacks import ModelCheckpoint

filepath_for_w='denoise_by_VAE_weights_1.h5'



###########
##########
experiment_dir= 'exp_'+str(int(time.time()))
os.mkdir(experiment_dir)
this_script=sys.argv[0]
from shutil import copyfile
copyfile(this_script, experiment_dir+'/'+this_script)
##########
###########


batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 10
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

x1 = Input(batch_shape=(100, original_dim))
x2 = Input(batch_shape=(100, original_dim))
#dec1 = vae(

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#after loading the data, change to the new experiment dir
os.chdir(experiment_dir) #
##########################

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


noise_factor = 0.5

x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


for i in range (10):

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)

    checkpointer=ModelCheckpoint(filepath_for_w, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    vae.fit(x_train_noisy, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test_noisy, x_test),
            callbacks=[checkpointer])
    vae.load_weights(filepath_for_w) 

    #print (x_train.shape)
    #print (x_test.shape)

    decoded_imgs = vae.predict(x_test,batch_size=batch_size)
    np.save('decoded'+str(i)+'.npy',decoded_imgs)


np.save('tested.npy',x_test_noisy)
#np.save ('true_catagories.npy',y_test)
np.save('original.npy',x_test)

