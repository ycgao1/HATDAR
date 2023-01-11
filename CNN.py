# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:08:45 2020

@author: Yichen Gao

human activity classification 
The CNN part will be use to extract features in voxels
"""


sub_dirs=['boxing','jack','jump','squats','walk']

import glob
import os
import numpy as np
# random seed.
rand_seed = 200
from numpy.random import seed
seed(rand_seed)
#from tensorflow import set_random_seed
#set_random_seed(rand_seed)
from tensorflow.random import set_seed
set_seed(rand_seed)
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras import backend as K

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers.convolutional import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import *
from tensorflow.keras.utils import plot_model
#from action_radar.extract import extract


def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping=dict()

    count=0
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    y_features=y_features.reshape(y_features.shape[0],1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features, categories)
    del y_data

    return y_features

# data extraction
def data_extract(extract_path):
    train_data=[]
    train_label=[]
    i=0
    for sub_dir in sub_dirs:
        Data_path = extract_path+sub_dir
        data_train=np.load(Data_path+'.npz')
        train_data_n = np.array(data_train['arr_0'],dtype=np.dtype(np.float32))
        train_data_n = train_data_n.reshape(train_data_n.shape[0],train_data_n.shape[1], train_data_n.shape[2],train_data_n.shape[3],train_data_n.shape[4],1)
        train_label_n = one_hot_encoding(data_train['arr_1'], sub_dirs, categories=5)
        if i==0:
            train_data=train_data_n
            train_label=train_label_n
        else:
            train_data = np.concatenate((train_data, train_data_n), axis=0)
            train_label = np.concatenate((train_label, train_label_n), axis=0)
        i+=1

        del data_train, 
        del train_data_n, train_label_n
    return train_data, train_label


def attention_3d_block(inputs):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if False:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def full_3D_model(frame):
    print('building the model ... ')
    inputs = Input(shape=(frame,16, 32, 32,1))
    # 1st layer group
    layer1=TimeDistributed(Conv3D(4, (3, 3, 3), strides=(1, 1, 1), name="conv1a", padding="same", activation="relu"))(inputs)
    # 2nd layer group
    layer2=TimeDistributed(Conv3D(8, (3, 3, 3), strides=(1, 1, 1), name="conv1b", padding="same", activation="relu"))(layer1)

    layer2=TimeDistributed(MaxPooling3D(name="pool1", strides=(2, 2, 2), pool_size=(2, 2, 2), padding="valid"))(layer2)
    layer2 = BatchNormalization(axis=5)(layer2)

    # 3rd layer group
    layer3=TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2a", padding="same", activation="relu"))(layer2)
    layer3=TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2b", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), name="pool2", padding="valid"))(layer3)

    layer3=TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), name="conv3a", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), name="conv3b", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), name="pool3", padding="valid"))(layer3)
    layer3 = BatchNormalization(axis=-1)(layer3)


    a=TimeDistributed(Flatten())(layer3)
    a=Dropout(.3)(a)
    
    a = Dense(512, activation='sigmoid')(a)
    
    a = attention_3d_block(a)
    a = BatchNormalization(axis=-1)(a)
    
    a = attention_3d_block(a)
    a = BatchNormalization(axis=-1)(a)
    
    a = attention_3d_block(a)
    a = BatchNormalization(axis=-1)(a)
    
    #lstm
    a = Bidirectional(LSTM(256,return_sequences=False, stateful=False))(a)
    

    a = Dropout(.3)(a)
    
    a = Dense(128)(a)
    
    
    a = Dense(32, activation='sigmoid')(a)

    output=Dense(5, activation='softmax', name = 'output')(a)
    
    model = Model(inputs=[inputs], outputs=output)

    return model


def model_train(frame, lr, beta_1, beta_2, batch_size, epochs, model_path, train_data, train_label, dev_data,dev_label): 
    model = full_3D_model(frame)
    model.summary()
    adam = optimizers.Adam(lr, beta_1, beta_2, epsilon=None,
                       decay=0.0, amsgrad=False)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=adam,
                  metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list= [checkpoint]
    
    learning_hist = model.fit(train_data, train_label,
                              batch_size,
                              epochs,
                              verbose=1,
                              shuffle=True,
                              validation_data=(dev_data,dev_label),
                              callbacks=callbacks_list
                          )
    del model
    return learning_hist

def model_test(frame, model_path, test_data, test_label):
    model = full_3D_model(frame)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    model.load_weights(model_path, by_name=True)

    result=model.evaluate(test_data, test_label, verbose=2)
    print("Accuracy:",result[1])
    
def main():
    extract_path = 'data/extract/raw/'
    checkpoint_model_path="cnn/model.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=120, type=int)
    parser.add_argument("--voxel_path", default='data/voxel/')
    parser.add_argument("--model_path", default='model_data/cnn/model.h5')
    parser.add_argument("--learning_rate", default=0.001, type=int)
    parser.add_argument("--beta_1", default=0.9, type=int)
    parser.add_argument("--beta_2", default=0.999, type=int)
    parser.add_argument("--batch_size", default=15, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    
    args = parser.parse_args()
    
    frame = args.frame
    voxel_path = args.voxel_path
    model_path = args.model_path
    lr = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    batch_size = args.batch_size
    epochs = args.epochs
    
    train_data, train_label = data_extract(voxel_path)
    train_data, dev_data, train_label, dev_label  = train_test_split(train_data, train_label, test_size=0.20, random_state=1)
    dev_data, test_data, dev_label, test_label  = train_test_split(dev_data, dev_label, test_size=0.50, random_state=1)
    print('train_data:',train_data.shape)
    print('train_label',train_label.shape)
    print('dev_data:',dev_data.shape)
    print('dev_label',dev_label.shape)
    print('test_data:',test_data.shape)
    print('test_label',test_label.shape)
    
    model_train(frame, lr, beta_1, beta_2, batch_size, epochs, model_path, train_data, train_label, dev_data,dev_label)
    model_test(frame, model_path, test_data, test_label)
    
    
    
if __name__ == "__main__":
    main()

