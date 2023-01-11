# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 03:20:07 2023

@author: ycgao
"""

sub_dirs=['boxing','jack','jump','squats','walk']


import random
import numpy as np
import tensorflow as tf
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
    from tf.keras.utils import to_categorical
    y_features = to_categorical(y_features, categories)
    del y_data

    return y_features

def get_data(sub_dir_1, sub_dir_2, Data_path_1, Data_path_2, scope):
    
    print(sub_dir_1+'_'+sub_dir_2)
    data_raw_1 = np.load(Data_path_1)
    data_1=data_raw_1['arr_0']
    label_1=data_raw_1['arr_1']
    data_raw_2 = np.load(Data_path_2)
    data_2=data_raw_2['arr_0']
    label_2=data_raw_2['arr_1']
    del data_raw_1, data_raw_2
    L_1=data_1.shape[0]
    L_2=data_2.shape[0]
    features =np.empty((0, 120, data_1.shape[2]))
    labels_0=np.empty((0,1))
    labels_1=np.empty((0,5))
    labels_2=np.empty((0,5))
    if sub_dir_1==sub_dir_2:
        #print(sub_dir_1)
        resultList=random.sample(range(0,L_2),4)
    else:
        resultList=random.sample(range(0,L_2),1)
    for i in range(L_1):
        for j in resultList:
            r=random.randint(0, scope)
            c=random.randint(0, 1)
            if c==0:
                r=r
            else:
                r=-r
            a=data_1[[i], 0:60+r]
            b=data_2[[j], 60+r:120]
            label_a=label_1[i]
            label_b=label_2[j]
            data_part=np.hstack((a,b))
            features=np.vstack([features, data_part])
            if sub_dir_1==sub_dir_2:
                labels_0=np.vstack([labels_0, np.zeros([1,1])])
            else:
                labels_0=np.vstack([labels_0, np.ones([1,1])])
            labels_1=np.vstack([labels_1, label_a])
            labels_2=np.vstack([labels_2, label_b])
    
    del data_1, data_2
    return features, labels_0, labels_1, labels_2


def transitionGenernation(data_path, save_path, scope):
    Number = 0
    for sub_dir_1 in sub_dirs:
        Data_path_1 = data_path+'/'+sub_dir_1+'.npz'
        for sub_dir_2 in sub_dirs:
            Data_path_2 = data_path+'/'+sub_dir_2+'.npz'
            features, labels_0, labels_1, labels_2=get_data(sub_dir_1, sub_dir_2, Data_path_1, Data_path_2, scope)
            np.savez(save_path+'/'+sub_dir_1+'_'+sub_dir_2, features, labels_0, labels_1, labels_2)
            Number+=features.shape[0]
            print("Activity: "+ sub_dir_1 + "," + sub_dir_2)
            print("Data size: ", features.shape[0])
            del features, labels_0, labels_1, labels_2

        

def main():
    extract_path = 'data/extract/raw/'
    checkpoint_model_path="cnn/model.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default=110, type=int)
    parser.add_argument("--feature_path", default='data/feature/')
    parser.add_argument("--transition_path", default='data/transition/')
    parser.add_argument("--dataset", choices=['train', 'validation', 'test'])
    
    args = parser.parse_args()
    
    scope = args.scope/2
    feature_path = args.feature_path
    transition_path = args.transition_path
    dataset = args.dataset
    
    data_path = feature_path + str(dataset)
    save_path = transition_path + str(dataset)
    
    transitionGenernation(data_path, save_path, scope)
    
    
if __name__ == "__main__":
    main()
