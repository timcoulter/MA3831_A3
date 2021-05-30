import os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
from word_processing import clean_words
from collections import OrderedDict

# Imports
# Basic
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, random, math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# DL
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.layers import Layer, Input, Embedding, Dropout, SpatialDropout1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Bidirectional, GRU, Activation, Dense, LSTM, Conv1D
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras import initializers, regularizers, constraints
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.callbacks import History 
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)
    
df = pickle.load(open("review_df2.p","rb"))

user_rating_col= ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

X_train, X_test, Y_train, Y_test = train_test_split(df['review'],df[user_rating_col], train_size=0.8, random_state=1)

all_words = ' '.join(list(df['review']))
all_words = all_words.split(' ')

max_words = len(set(all_words))
embedding_vec_len = 32
max_len = 100
lstm_units = 256
    
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

def get_plots(history,name,show=False):
    hist = history.history
    X = range(1,len(hist['loss'])+1)
    plt.figure()
    plt.plot(X,hist['acc'],label='accuracy')
    plt.plot(X,hist['val_acc'],label='val_accuracy')
    plt.plot(X,hist['loss'],label='loss')
    plt.plot(X,hist['val_loss'],label='val_loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss, Accuracy vs. Validation Loss, Accuracy of {0}'.format(name))
    plt.savefig("{0}.png".format(name))

    plt.figure()
    plt.plot(X,hist['precision'],label='precision')
    plt.plot(X,hist['val_precision'],label='val_precision')
    plt.plot(X,hist['recall'],label='recall')
    plt.plot(X,hist['val_recall'],label='val_recall')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Performance Measure')
    plt.title('Precision, Recall vs. Validation Precision, Recall of {0}'.format(name))
    plt.savefig("{0}_precision_recall.png".format(name))

    plt.figure()
    plt.plot(X,hist['f1'],label='f1')
    plt.plot(X,hist['val_f1'],label='val_f1')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs. Validation F1-Score of {0}'.format(name))
    plt.savefig("{0}_F1_Score.png".format(name))

    plt.figure()
    plt.plot(X,hist['auc'],label='AUC')
    plt.plot(X,hist['val_auc'],label='val_AUC')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('mean AUC')
    plt.title('AUC vs. Validation AUC of {0}'.format(name))
    plt.savefig("{0}_AUC.png".format(name))
    
    if show:
        plt.show()

def get_BiGRU(max_words=max_words,embedding_vec_len=embedding_vec_len,max_len=max_len):

    model = Sequential(name='BiGRU')
    model.add(Embedding(input_dim=max_words,output_dim=embedding_vec_len,input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(SeqWeightedAttention())
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    
    L_func = tfa.losses.SigmoidFocalCrossEntropy()
    model.compile(loss=L_func, metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    #model.compile(loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    
    
    #model.summary()
    return model

def get_LSTM(lstm_units=lstm_units,max_words=max_words,embedding_vec_len=embedding_vec_len,max_len=max_len):
    # Define input tensor
    model = Sequential(name='LSTM')
    model.add(Embedding(input_dim=max_words,output_dim=embedding_vec_len,input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(units=lstm_units, input_shape=(max_len,embedding_vec_len)))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    
    L_func = tfa.losses.SigmoidFocalCrossEntropy()

    model.compile(loss=L_func, metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    #model.compile(loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    
    #model.summary()
    return model

def get_CNN(max_words=max_words,embedding_vec_len=embedding_vec_len,max_len=max_len):
    model = Sequential(name="CNN")
    model.add(Embedding(input_dim=max_words,output_dim=embedding_vec_len,input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(128, 5, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    L_func = tfa.losses.SigmoidFocalCrossEntropy()
    model.compile(loss=L_func, metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    #model.compile(loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
    return model
  
def test_model(model,Y_test=Y_test,test_sequences_matrix=test_sequences_matrix,max_words=max_words,embedding_vec_len=embedding_vec_len,max_len=max_len,save=True):
   
    #evaluate model on test set
    accr = model.evaluate(test_sequences_matrix,Y_test)
        #print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    y_pred = model.predict(test_sequences_matrix)

    c_report = classification_report(y_true=Y_test,y_pred=y_pred>0.5,target_names=user_rating_col)
    print(model.name)
    print(c_report)
    c_report = classification_report(y_true=Y_test,y_pred=y_pred>0.5,target_names=user_rating_col,output_dict=True)
    pickle.dump(c_report,open('{0}_report.p'.format(model.name), "wb" ))
    
batch_size = 128
n_epochs = 50

history = History()
BiGRU = get_BiGRU()
#BiGRU.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[EarlyStopping(monitor='val_acc',min_delta=0.00001),history])
BiGRU.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[history])
test_model(BiGRU)
get_plots(history,"BiGRU")

#serialize model to JSON
model_json = BiGRU.to_json()
with open("BiGRU" + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
BiGRU.save_weights("BiGRU" + ".h5")

tf.keras.backend.clear_session()
history = History()
lstm = get_LSTM()

#lstm.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[EarlyStopping(monitor='val_acc',min_delta=0.00001),history])
lstm.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[history])
test_model(lstm)
get_plots(history,"LSTM")

#serialize model to JSON
model_json = lstm.to_json()
with open("lstm" + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm.save_weights("lstm" + ".h5")

tf.keras.backend.clear_session()
history = History()
cnn = get_CNN()
#cnn.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[EarlyStopping(monitor='val_acc',min_delta=0.00001),history])
cnn.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[history])
test_model(cnn)
get_plots(history,"CNN")

model_json = cnn.to_json()
with open("cnn" + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("cnn" + ".h5")