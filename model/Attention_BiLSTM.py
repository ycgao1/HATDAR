import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from model.attention_keras import *



def Attention_BiLSTM(frame):
    print('building the model ... ')
    inputs = Input(shape=(frame, 512))
    
    a = Dense(120, activation='sigmoid')(inputs)
    
    a = MultiHeadAttention(2,60)([a,a,a])
    a = BatchNormalization(axis=-1)(a)
    a=LeakyReLU()(a)

    
    a = MultiHeadAttention(2,60)([a,a,a])
    a = BatchNormalization(axis=-1)(a)
    a=LeakyReLU()(a)
    
    a = MultiHeadAttention(2,60)([a,a,a])
    a = BatchNormalization(axis=-1)(a)
    a=LeakyReLU()(a)
    
    a = Dropout(.2)(a)
    
    #a=Flatten()(a)
    
    #a = Dense(256, activation='sigmoid')(a)
    
    s = Bidirectional(LSTM(5,return_sequences=False, stateful=False))(a)
    
    Act_1=s[:,0:5]
    Act_1=Dense(5, activation='softmax', name='class_1')(Act_1)
    
    Act_2=s[:,5:10]
    Act_2=Dense(5, activation='softmax', name='class_2')(Act_2)
    
    Transition = Dot(axes=1)([Act_1, Act_2])
    
    Transition = Lambda(lambda Transition: 1-Transition)(Transition)
    
    output = Concatenate()([Transition, Act_2])
    
    model = Model(inputs=inputs, outputs=output)

    return model