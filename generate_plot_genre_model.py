import os, pickle, tqdm
from word_processing import clean_words
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import model_from_json
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.callbacks import History 
import tensorflow_addons as tfa


#Functions for keras.metrics
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
    pth = 'C:/Users/Tim/Documents/MA3831 Data/A3'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/A3'
    os.chdir(pth)
    

  
df1 = pickle.load(open("plot_df2.p","rb"))
df1 = df1.reset_index(drop=True)

#get unique genres in alphabetical order
genres = list(df1['genre'].unique())
genres = ', '.join(genres).lower().strip()
genres = genres.split(', ')
genres = list(filter(None, genres))
genres = list(OrderedDict.fromkeys(genres))
genres = sorted(genres)

max_words = 100000
embedding_vec_len = 50
max_len = 100
lstm_units = 256

name = "LSTM"

X_train, X_test, Y_train, Y_test = train_test_split(df1['plot'],df1[genres], train_size=0.8, random_state=1)

#Tokenize and generate sequence matrix
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

#Generate model
model = Sequential(name=name)
model.add(Embedding(input_dim=max_words,output_dim=embedding_vec_len,input_length=max_len))
model.add(LSTM(units=lstm_units))
model.add(Dropout(0.2))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(len(genres), activation='sigmoid'))

L_func = tfa.losses.SigmoidFocalCrossEntropy()
model.compile(loss=L_func, metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')
#model.compile(loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),f1,tf.keras.metrics.AUC()], optimizer='adam')

batch_size = 128
n_epochs = 50

history = History()
model.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[history])
#model.fit(sequences_matrix,Y_train,batch_size=batch_size,epochs=n_epochs,validation_split=0.2,verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.00001),history])

#Plot training progress
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

#process test set data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#evaluate model on test set
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

y_pred = model.predict(test_sequences_matrix)

c_report = classification_report(y_true=Y_test,y_pred=y_pred>0.5,output_dict=True)
pickle.dump(c_report,open('classification_report.p', "wb" ))
c_report = classification_report(y_true=Y_test,y_pred=y_pred>0.5)
print("Classication Report")
print(c_report)

model_json = model.to_json()
with open("lstm" + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("lstm" + ".h5")


 



