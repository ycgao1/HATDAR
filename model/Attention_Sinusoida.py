import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from model.attention_keras import *



def Attention_Sinusoida(frame):
    print('building the model ... ')
    inputs = Input(shape=(120,512))
    
    #a = Dense(512, activation='sigmoid')(inputs)
    
    a= SinusoidalPositionEmbedding(512)(inputs)
    
    a = Dense(120, activation='sigmoid')(a)
    
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
    
    a = Lambda(lambda a: a[:, 0])(a)
    s = Dense(10)(a)
    
    Act_1=s[:,0:5]
    Act_1=Dense(5, activation='softmax', name='Act_1')(Act_1)
    
    Act_2=s[:,5:10]
    Act_2=Dense(5, activation='softmax', name='Act_2')(Act_2)
    
    m = Dot(axes=1)([Act_1, Act_2])
    
    m = Lambda(lambda m: 1-m)(m)
    
    output = Concatenate()([m, Act_2])
    #out=Dense(1,'relu')(m)
    
    model = Model(inputs=[inputs], outputs=output)

    return model