import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
sub_dirs=['boxing','jack','jump','squats','walk']

import glob
import os
import numpy as np
# random seed.
rand_seed = 200
from tensorflow.random import set_seed
set_seed(rand_seed)

import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
#from tensorflow.keras.layers.convolutional import *
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import *
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from data_loader import get_data_test
from model.BiLSTM import BiLSTM
from model.Attention import Attention
from model.Attention_BiLSTM import Attention_BiLSTM
from model.Attention_trainable import Attention_trainable
from model.Attention_Sinusoida import Attention_Sinusoida
from model.Attention_0_1 import Attention_0_1
from metrics import *
import argparse
        


def model_test(frame, model_name, model_path, test_data, test_label, transition, recognition):
    print("test:")
    if(model_name == "BiLSTM"):
        model = BiLSTM(frame)
    elif(model_name == "Attention"):
        model = Attention(frame)
    elif(model_name == "Attention_BiLSTM"):
        model = Attention_BiLSTM(frame)
    elif(model_name == "Attention_trainable"):
        model = Attention_trainable(frame)
    elif(model_name == "Attention_Sinusoida"):
        model = Attention_Sinusoida(frame)
    elif(model_name == "Attention_0_1"):
        model = Attention_0_1(frame)
    else:
        print("Model name wrong!")
        return
    model.compile(loss=losses, metrics=accuracy)
    model.load_weights(model_path, by_name=True)

    result=model.evaluate(test_data, test_label, verbose=2)
    print("Accuracy:", result[1])
    
    if(transition == True):
        predict=model.predict(test_data, batch_size=200)
        transition_detection(predict, test_label)
        
    if(recognition == True):
        predict=model.predict(test_data, batch_size=200)
        activity_recognition_confusion_matrix(predict, test_label)
        
    del model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=120, type=int)
    parser.add_argument("--data_path", default='data/transition/')
    parser.add_argument("--model_dir", default='model_data/transition/')
    parser.add_argument("--model", choices=['BiLSTM', 'Attention', 'Attention_BiLSTM', 'Attention_trainable', 'Attention_Sinusoida', 'Attention_0_1'])
    parser.add_argument("--transition_detection", default=True, type=bool)
    parser.add_argument("--activity_recognition", default=True, type=bool)
    
    args = parser.parse_args()
    
    frame = args.frame
    data_path = args.data_path
    model_dir = args.model_dir
    model_name = args.model
    model_path = model_dir+ model_name +'_model.h5'
    transition = args.transition_detection
    recognition= args.activity_recognition
    
    
    test_data, test_label = get_data_test(data_path)
     
    model_test(frame, model_name, model_path, test_data, test_label, transition, recognition)
    
if __name__ == "__main__":
    main()
