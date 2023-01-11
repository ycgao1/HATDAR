sub_dirs=['boxing','jack','jump','squats','walk']
import os
import numpy as np
import tensorflow as tf 

def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping = dict()

    count = 0
    for i in sub_dirs:
        Mapping[i] = count
        count = count+1

    y_features2 = []
    for i in range(len(y_data)):
        Type = y_data[i]
        lab = Mapping[Type]
        y_features2.append(lab)

    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    from tf.keras.utils import to_categorical
    y_features = to_categorical(y_features, categories)
    del y_data

    return y_features


def get_data_train(data_path):
    train_data=[]
    train_label=[]
    i=0
    data_path_train = data_path+'train/'
    for filename in os.listdir(data_path_train):
        Data_path = data_path_train+filename
        print(Data_path)
        data_load=np.load(Data_path)
        data_n = np.array(data_load['arr_0'],dtype=np.dtype(np.float32))
        label_n_1 = data_load['arr_1']
        label_n_2 = data_load['arr_2']
        label_n_3 = data_load['arr_3']
        print(data_n.shape)
        print(label_n_1.shape)
        print(label_n_2.shape)
        print(label_n_3.shape)
        
        if i==0:
            train_data=data_n
            train_label_1=label_n_1
            train_label_2=label_n_2
            train_label_3=label_n_3
        else:
            train_data = np.concatenate((train_data, data_n), axis=0)
            train_label_1 = np.concatenate((train_label_1, label_n_1), axis=0)
            train_label_2 = np.concatenate((train_label_2, label_n_2), axis=0)
            train_label_3 = np.concatenate((train_label_3, label_n_3), axis=0)
        i+=1
        del data_load
        del data_n, label_n_1, label_n_2, label_n_3
    print("train_data:",train_data.shape)
    print("train_label_1:",train_label_1.shape)
    print("train_label_2:",train_label_2.shape)
    print("train_label_3:",train_label_3.shape)
    
        
    validation_data=[]
    validation_label=[]
    i=0
    data_path_validation = data_path+'validation/'
    for filename in os.listdir(data_path_validation):
        Data_path = data_path_validation+filename
        print(Data_path)
        data_load=np.load(Data_path)
        data_n = np.array(data_load['arr_0'],dtype=np.dtype(np.float32))
        label_n_1 = data_load['arr_1']
        label_n_2 = data_load['arr_2']
        label_n_3 = data_load['arr_3']
        print(data_n.shape)
        print(label_n_1.shape)
        print(label_n_2.shape)
        print(label_n_3.shape)
        
        if i==0:
            validation_data=data_n
            validation_label_1=label_n_1
            validation_label_2=label_n_2
            validation_label_3=label_n_3
        else:
            validation_data = np.concatenate((validation_data, data_n), axis=0)
            validation_label_1 = np.concatenate((validation_label_1, label_n_1), axis=0)
            validation_label_2 = np.concatenate((validation_label_2, label_n_2), axis=0)
            validation_label_3 = np.concatenate((validation_label_3, label_n_3), axis=0)
        i+=1
        del data_load
        del data_n, label_n_1, label_n_2, label_n_3
    print("validation_data:",validation_data.shape)
    print("validation_label_1:",validation_label_1.shape)
    print("validation_label_2:",validation_label_2.shape)
    print("validation_label_3:",validation_label_3.shape)
    
        
    test_data=[]
    test_label=[]
    i=0
    data_path_test = data_path+'test/'
    for filename in os.listdir(data_path_test):
        Data_path = data_path_test+filename
        print(Data_path)
        data_load=np.load(Data_path)
        data_n = np.array(data_load['arr_0'],dtype=np.dtype(np.float32))
        label_n_1 = data_load['arr_1']
        label_n_2 = data_load['arr_2']
        label_n_3 = data_load['arr_3']
        print(data_n.shape)
        print(label_n_1.shape)
        print(label_n_2.shape)
        print(label_n_3.shape)
        
        if i==0:
            test_data=data_n
            test_label_1=label_n_1
            test_label_2=label_n_2
            test_label_3=label_n_3
        else:
            test_data = np.concatenate((test_data, data_n), axis=0)
            test_label_1 = np.concatenate((test_label_1, label_n_1), axis=0)
            testlabel_2 = np.concatenate((test_label_2, label_n_2), axis=0)
            test_label_3 = np.concatenate((test_label_3, label_n_3), axis=0)
        i+=1
        del data_load
        del data_n, label_n_1, label_n_2, label_n_3
    print("test_data:",test_data.shape)
    print("test_label_1:",test_label_1.shape)
    print("test_label_2:",test_label_2.shape)
    print("test_label_3:",test_label_3.shape)
        
    train_label=np.concatenate((train_label_1, train_label_3), axis=1)
    validation_label=np.concatenate((validation_label_1, validation_label_3), axis=1)
    test_label=np.concatenate((test_label_1, test_label_3), axis=1)
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def get_data_test(data_path):
    test_data=[]
    test_label=[]
    i=0
    data_path_test = data_path+'test/'
    for filename in os.listdir(data_path_test):
        Data_path = data_path_test+filename
        print(Data_path)
        data_load=np.load(Data_path)
        data_n = np.array(data_load['arr_0'],dtype=np.dtype(np.float32))
        label_n_1 = data_load['arr_1']
        label_n_2 = data_load['arr_2']
        label_n_3 = data_load['arr_3']
        print(data_n.shape)
        print(label_n_1.shape)
        print(label_n_2.shape)
        print(label_n_3.shape)
        
        if i==0:
            test_data=data_n
            test_label_1=label_n_1
            test_label_2=label_n_2
            test_label_3=label_n_3
        else:
            test_data = np.concatenate((test_data, data_n), axis=0)
            test_label_1 = np.concatenate((test_label_1, label_n_1), axis=0)
            testlabel_2 = np.concatenate((test_label_2, label_n_2), axis=0)
            test_label_3 = np.concatenate((test_label_3, label_n_3), axis=0)
        i+=1
        del data_load
        del data_n, label_n_1, label_n_2, label_n_3
    print("test_data:",test_data.shape)
    print("test_label_1:",test_label_1.shape)
    print("test_label_2:",test_label_2.shape)
    print("test_label_3:",test_label_3.shape)
        
    test_label=np.concatenate((test_label_1, test_label_3), axis=1)
    
    return test_data, test_label
