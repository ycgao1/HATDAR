import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from data_loader import get_data_train
from model.BiLSTM import BiLSTM
from model.Attention import Attention
from model.Attention_BiLSTM import Attention_BiLSTM
from model.Attention_trainable import Attention_trainable
from model.Attention_Sinusoida import Attention_Sinusoida
from model.Attention_0_1 import Attention_0_1
from metrics import *
import argparse


        
def model_train(frame, model_name, lr, beta_1, beta_2, checkpoint_monitor, checkpoint_mode,
                reduce_lr_monitor, reduce_lr_factor, reduce_lr_patience,
                earlystopoing_monitor, earlystopoing_mode, 
                earlystopoing_patience, batch_size, epochs, model_path, 
                train_data, train_label, validation_data, validation_label): 
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
    model.summary()
 
    adam = optimizers.Adam(lr, beta_1, beta_2, epsilon=1E-8,
                           decay=0.0, amsgrad=False)


    model.compile(loss=losses,
                       optimizer=adam,
                      metrics=accuracy)

    checkpoint = ModelCheckpoint(model_path, monitor=checkpoint_monitor, verbose=1, 
                                 save_best_only=True, mode=checkpoint_mode)

    reduce_lr = ReduceLROnPlateau(monitor=reduce_lr_monitor, factor=reduce_lr_factor,
                                  patience=reduce_lr_patience, min_lr=1E-6)

    earlystopping = EarlyStopping(monitor=earlystopoing_monitor, patience=earlystopoing_patience, 
                                  verbose=1, mode=earlystopoing_mode)

    callbacks_list = [checkpoint, earlystopping, reduce_lr]

    learning_hist = model.fit(train_data, train_label,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle=True,
                              validation_data=(validation_data,validation_label),
                              callbacks=callbacks_list
                              )
    del model
    return learning_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=120, type=int)
    parser.add_argument("--data_path", default='data/transition/')
    parser.add_argument("--model_dir", default='model_data/transition/')
    parser.add_argument("--model", choices=['BiLSTM', 'Attention', 'Attention_BiLSTM', 'Attention_trainable', 'Attention_Sinusoida', 'Attention_0_1'])
    parser.add_argument("--learning_rate", default=0.002, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.9999, type=float)
    parser.add_argument("--checkpoint_monitor", default='val_accuracy', choices=['val_loss', 'val_accuracy'])
    parser.add_argument("--checkpoint_mode", default='max', choices=['max', 'min'])
    parser.add_argument("--reduce_lr_monitor", default='val_loss', choices=['val_loss', 'val_accuracy'])
    parser.add_argument("--reduce_lr_factor", default=0.2, type=float)
    parser.add_argument("--reduce_lr_patience", default=10, type=int)
    parser.add_argument("--earlystopoing_monitor", default='val_accuracy', choices=['val_loss', 'val_accuracy'])
    parser.add_argument("--earlystopoing_mode", default='max', choices=['max', 'min'])
    parser.add_argument("--earlystopoing_patience", default=30, type=int)
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--draw", default=1, type=int)
    
    args = parser.parse_args()
    
    frame = args.frame
    data_path = args.data_path
    model_dir = args.model_dir
    model_name = args.model
    model_path = model_dir+ model_name +'_model.h5'
    lr = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    batch_size = args.batch_size
    epochs = args.epochs
    checkpoint_monitor = args.checkpoint_monitor
    checkpoint_mode = args.checkpoint_mode
    reduce_lr_monitor = args.reduce_lr_monitor
    reduce_lr_factor = args.reduce_lr_factor
    reduce_lr_patience = args.reduce_lr_patience
    earlystopoing_monitor = args.earlystopoing_monitor
    earlystopoing_mode = args.earlystopoing_mode
    earlystopoing_patience = args.earlystopoing_patience
    draw = args.draw
    
    train_data, train_label, validation_data, validation_label, test_data, test_label = get_data_train(data_path)
    
    learning_hist = model_train(frame, model_name, lr, beta_1, beta_2, checkpoint_monitor, checkpoint_mode,
                    reduce_lr_monitor, reduce_lr_factor, reduce_lr_patience,
                    earlystopoing_monitor, earlystopoing_mode, 
                    earlystopoing_patience, batch_size, epochs, model_path, 
                    train_data, train_label, validation_data, validation_label)
    
    if(draw ==True):
        draw_history(learning_hist)
        
   
    
if __name__ == "__main__":
    main()
