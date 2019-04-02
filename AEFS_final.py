
from keras.models import Sequential , Model
from keras.layers import Input, Dense
from keras.regularizers import  l1, l2
import keras.backend as K
from keras import regularizers
from keras.activations import linear
from keras.datasets import mnist
import tensorflow as tf
from keras.losses import mean_squared_error
from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt



alpha = 0.01
beta = 0.01 


def layer1_reg(weight_matrix):
    return alpha * K.sum(K.sqrt(tf.reduce_sum(K.square(weight_matrix), axis=1))) + (beta/2.)*K.sum(K.square(weight_matrix))

def layer2_reg(weight_matrix):
    return (beta/2.)*K.sum(K.square(weight_matrix))

def frob_loss(y_true,y_pred):
    return 0.5*K.mean(K.sqrt(K.sum(tf.reduce_sum(K.square(y_true-y_pred)))))
    




if __name__ == '__main__':
    dataset = 'face'
    if(dataset == 'face'):
        data = loadmat('E:/MSc/TensorFlow Learn/AEFS/warpPIE10P.mat')
        X = data['X']/255.
        x_train = X[0:180,]
        x_test = X[180:209,]
        Y = data['Y']-1
        input_shape = (44,55)
        input_dim = 44*55

    elif(dataset == 'mnist'):
        (x_train, y_train),(x_test,y_test) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), -1)
        x_train = x_train / 255.
        x_test = x_test.reshape(len(x_test), -1)
        x_test = x_test / 255.
        input_shape = (28,28)
        input_dim = 28*28


    
encoding_dim = 128  


input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu' , kernel_regularizer =layer1_reg)(input_img)
decoded = Dense(input_dim, activation='sigmoid' , kernel_regularizer =layer2_reg)(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
opt = tf.train.ProximalGradientDescentOptimizer(0.003)

autoencoder.compile(optimizer= 'Adadelta', loss= frob_loss)

autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=150,
                batch_size=20,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)




n = 10 
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(44, 55))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(44, 55))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


for layer in autoencoder.layers:
    weights = layer.get_weights() # list of numpy arrays

weights1 = weights[0]
layer1_weights = np.sum(np.square(weights1),0)





