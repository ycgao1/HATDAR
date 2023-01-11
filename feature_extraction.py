# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:08:45 2020

@author: Yichen Gao

human activity classification 
The CNN part will be use to extract features in voxels
"""


sub_dirs=['boxing','jack','jump','squats','walk']

import numpy as np
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
import argparse

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
def data_extract(extract_path, sub_dir):
    data=[]
    label=[]
    Data_path = extract_path+sub_dir
    data_raw=np.load(Data_path+'.npz')
    data = np.array(data_raw['arr_0'],dtype=np.dtype(np.float32))
    label = one_hot_encoding(data_raw['arr_1'], sub_dirs, categories=5)
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4],1)
    return data, label




def feature_extract_model(frame):
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
    
    a = Dense(512, activation='sigmoid')(a)



    model = Model(inputs=[inputs], outputs=a)
    return model
    


def feature_extraction(frame, voxel_path, model_path, feature_path):
    
    model = feature_extract_model(frame)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    model.load_weights(model_path, by_name=True)

    for sub_dir in sub_dirs:
        print(sub_dir)
        data, labels = data_extract(voxel_path, sub_dir)
        features=model.predict(data, verbose=2)
        features_train, features_validation, labels_train, labels_validation  = train_test_split(features, labels, test_size=0.20, random_state=1)
        features_validation, features_test, labels_validation, labels_test  = train_test_split(features_validation, labels_validation, test_size=0.50, random_state=1)
        np.savez(feature_path+ '/train/' + sub_dir, features_train, labels_train)
        np.savez(feature_path+ '/validation/' + sub_dir, features_validation, labels_validation)
        np.savez(feature_path+ '/test/' + sub_dir, features_test, labels_test)
        print("train:",features_train.shape)
        print("validation:",features_validation.shape)
        print("test:",features_test.shape)

    
def main():
    extract_path = 'data/extract/raw/'
    checkpoint_model_path="cnn/model.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=120, type=int)
    parser.add_argument("--voxel_path", default='data/voxel/')
    parser.add_argument("--feature_path", default='data/feature/')
    parser.add_argument("--model_path", default='model_data/cnn/model.h5')
    

    
    args = parser.parse_args()
    
    frame = args.frame
    voxel_path = args.voxel_path
    model_path = args.model_path
    feature_path = args.feature_path
    
    feature_extraction(frame, voxel_path, model_path, feature_path)
    
    
    
if __name__ == "__main__":
    main()



