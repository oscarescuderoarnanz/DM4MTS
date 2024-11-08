import keras 
from keras import layers

import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from keras.layers import LeakyReLU

import tensorflow as tf
from keras import backend as K
import os, random

import pandas as pd
import numpy as np
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import History

def normData (X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def normData_minmax (X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def loadData(i, norm, carpetas):    
    print("Window of 7 days without eliminating patients...")
    X_train = sio.loadmat('../../data_generated_by_tck/' + carpetas[i] + '/Ktrtr')
    X_test = sio.loadmat('../../data_generated_by_tck/' + carpetas[i] + '/Ktrte')
    X_train = X_train['Ktrtr']
    X_test = X_test['Ktrte']
    X_test = X_test.T

    y_train = pd.read_csv('../../../df_to_load/DataToPaperAndTFM_Mod1/Otras normalizaciones/Subconjuntos_3D_norm/' 
                          + carpetas[i] + '/y_train_tensor.csv')
    y_test = pd.read_csv('../../../df_to_load/DataToPaperAndTFM_Mod1/Otras normalizaciones/Subconjuntos_3D_norm/' 
                         + carpetas[i] + '/y_test_tensor.csv')

    
    if norm:
        #X_train, X_test = normData(X_train, X_test)
        X_train, X_test = normData_minmax(X_train, X_test)

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)
    
    return X_train, X_test, y_train, y_test

    
####################################################################################################################
########## FUNCTIONS FOR BUIKDING THE AUTOENCODERS #############################################################
####################################################################################################################


def AE(hyperparameters):

    autoencoder = keras.Sequential()
    # INPUT
    autoencoder.add(layers.InputLayer(input_shape=(hyperparameters['h_layers'][0],), name="inputLayer"))
    # ENCODER LAYER
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][0], name="encoder"))
    autoencoder.add(layers.Dropout(hyperparameters['dropout']))
    autoencoder.add(LeakyReLU())
        
    # LATENT LAYER
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][1], name="Latent_layer"))
    autoencoder.add(layers.Dropout(hyperparameters['dropout']))
    autoencoder.add(LeakyReLU())
    
    # OUTPUT  
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][0],  activation='sigmoid', name='output'))
                        

    myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['initial_learning_rate'])
    autoencoder.compile(loss='mse', optimizer=myOptimizer)
    
    return autoencoder



def DAE(hyperparameters):

    autoencoder = keras.Sequential()
    # INPUT
    autoencoder.add(layers.InputLayer(input_shape=(hyperparameters['h_layers'][0],), name="inputLayer"))
    autoencoder.add(layers.GaussianNoise(hyperparameters['std_noise']))
    # ENCODER
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][0], name="encoder"))
    autoencoder.add(layers.Dropout(hyperparameters['dropout']))
    autoencoder.add(LeakyReLU())    
    
    # LATENT LAYER
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][1], name="Latent_layer"))
    autoencoder.add(layers.Dropout(hyperparameters['dropout']))
    autoencoder.add(LeakyReLU())
    
   
    # OUTPUT
    autoencoder.add(layers.Dense(hyperparameters['h_layers'][0],  activation='sigmoid', name='output'))

    myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['initial_learning_rate'])
    autoencoder.compile(loss='mse', optimizer=myOptimizer)

    return autoencoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 
    

def runNetwork(X_train, X_val, hyperparameters, autoencodertype):
    batch_size = hyperparameters['batch_size']
    epochs = hyperparameters['epochs']

    autoencoder = None
        
    if autoencodertype['DAE']:
        autoencoder = DAE(hyperparameters)
    else:
        autoencoder = AE(hyperparameters)
        
    
    earlystopping = None
    try:
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      min_delta=hyperparameters["mindelta"], 
                                                      patience=10, 
                                                      restore_best_weights=True,
                                                      mode="min" 
                                                     )
        history = History()
        trained_model = autoencoder.fit(X_train, X_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_data=(X_val, X_val),
                                    callbacks=[earlystopping, history],
                                    workers=64,
                                    verbose=hyperparameters['verbose'])
        #print(history.history)


        return autoencoder, trained_model, earlystopping
    
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, encoder, y_test, 0, 0


    
def reset_keras(seed=42):
    # Close the previous session
    # sess = tf.compat.v1.keras.backend.get_session()
    # sess.close()
    K.clear_session()
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    # 5. Configure a new global `tensorflow` session
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)
    
 