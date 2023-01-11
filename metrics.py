import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    
    y_true = tf.convert_to_tensor(y_true)
    
    threshold = K.cast(0.7, y_pred.dtype)
    
    a = K.equal(y_true[:,0], K.cast(y_pred[:,0] > threshold, y_pred.dtype))
    b = K.equal(K.argmax(y_pred[:,1:6], axis=-1), K.argmax(y_true[:,1:6], axis=-1))
    
    return K.cast(tf.logical_and(a, tf.logical_or(K.cast(1-y_true[:,0], bool),b)), K.floatx())
    
def losses(y_true, y_pred):
    return K.binary_crossentropy(y_true[:,0], y_pred[:,0]) + y_true[:,0]*K.categorical_crossentropy(y_true[:,1:6], y_pred[:,1:6])

def transition_detection(predict, test_label):
    predict_1=np.where(predict[:,0].reshape((predict[:,0].shape[0],1))>0.7,1,0)
    test_label_1=np.where(test_label[:,0].reshape((test_label[:,0].shape[0],1))>0.7,1,0)
    
    TP = sum(predict_1*test_label_1)
    FP = sum((1-test_label_1)*predict_1)
    FN = sum(test_label_1*(1-predict_1))
    TN = sum((1-test_label_1)*(1-predict_1))
    
    Accuracy=(TP+TN)/(TP+FP+TN+FN)
    Recall=TP/(TP+FN)
    Precision=TP/(TP+FP)
    Specificity=TN/(TN+FP)
    F=2*Precision*Recall/(Precision+Recall)
    print("Human activity transition detection")
    print("Accuracy: ", Accuracy)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1: :", F)
    
def activity_recognition_confusion_matrix(predict, test_label):
    predict_1=np.where(predict[:,0].reshape((predict[:,0].shape[0],1))>0.7,1,0)
    predict_2=predict[:,1:6]
    test_label_2=predict[:,1:6]
    y_pre=predict_2[np.where(predict_1==1)[0],:]
    y_tru=test_label_2[np.where(predict_1==1)[0],:]
    
    matrix=confusion_matrix(np.argmax(y_tru, axis=1), np.argmax(y_pre, axis=1), normalize='true')
    matrix=np.round(matrix,4)
    
    plt.matshow(matrix, cmap=plt.cm.Blues) 
    #plt.colorbar()
    labels=['boxing','jack','jump','squats','walk']
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(len(matrix)): 
        for j in range(len(matrix)):
            plt.annotate(round(matrix[i,j],3), xy=(j, i), horizontalalignment='center', verticalalignment='center', color="white" if matrix[i, j] > 0.5 else "black")
    plt.ylabel('Human activity')
    plt.xlabel('Prediction label') 
    plt.savefig("results/confusion_matrix.png")
    
def draw_history(learning_hist):
    fig1, ax_acc = plt.subplots()
    plt.plot(learning_hist.history['accuracy'])
    plt.plot(learning_hist.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("results/accuracy.png")
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss - cnn_lstm batch=3')
    plt.plot(learning_hist.history['loss'])
    plt.plot(learning_hist.history['val_loss'])
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig("results/loss.png")
    plt.show()
    np.savez("results/accurate", learning_hist.history['accuracy'],learning_hist.history['val_accuracy'])
    np.savez("results/loss", learning_hist.history['loss'],learning_hist.history['val_loss'])