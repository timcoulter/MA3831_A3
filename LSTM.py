import os, pickle, tqdm
from word_processing import clean_words
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import model_from_json


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.callbacks import EarlyStopping
from keras import backend as K

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

def f1_m(y_true, y_pred):
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
df2 = pickle.load(open("review_df2.p","rb"))
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

genres = list(df1['genre'].unique())
genres = ', '.join(genres).lower().strip()
genres = genres.split(', ')
genres = list(filter(None, genres))


genres = list(OrderedDict.fromkeys(genres))
genres = sorted(genres)

print(df1.columns)

max_words = 100000
embedding_vec_len = 50
max_len = 100
lstm_units = 100

print(df1.head())

#trained_models = list()
c_reports = list()
c_mats = list()

for i in tqdm.tqdm(range(len(genres))):
    #print(genres[i])
    X_train, X_test, Y_train, Y_test = train_test_split(df1['plot'], df1[genres[i]], train_size=0.8, random_state=1)

    # conv_filters = 32
    # kernel_size = 3
    # hidden_nodes = Ns / (a * (Ni + No))

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    model = Sequential()
    model.add(Embedding(input_dim=max_words,output_dim=embedding_vec_len,input_length=max_len))
    #model.add(Dropout(0.2))
    #model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation='relu'))
    # #model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))

    model.add(LSTM(units=lstm_units, input_shape=(max_len,embedding_vec_len)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    
    #compile model
    
    #model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc',f1_m])
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    #fit training data
    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,validation_split=0.2,verbose=0, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    # serialize model to JSON
    model_json = model.to_json()
    with open(genres[i] + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(genres[i] + ".h5")
    #print("Saved model to disk")

    #load json and create model
    json_file = open(genres[i] + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(genres[i] + ".h5")
    #print("Loaded model from disk")

    #process test set data
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    #evaluate model on test set
    accr = model.evaluate(test_sequences_matrix,Y_test)
    #print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    y_pred = model.predict(test_sequences_matrix) > 0.5

    conf_mat = confusion_matrix(y_true=Y_test,y_pred=y_pred)
    #print("Confusion Matrix")
    #print(conf_mat)

    acc = (conf_mat[0,0] + conf_mat[1,1]) / np.sum(np.concatenate(conf_mat))

    c_report = classification_report(y_true=Y_test,y_pred=y_pred)
    #print("Classicaition Report")
    #print(c_report)

    #trained_models.append(model)
    c_reports.append(c_report)
    c_mats.append(conf_mat)

    
#pickle.dump(trained_models,open('RNNs.p', "wb" ))
pickle.dump(c_reports,open('classification_report.p', "wb" ))
pickle.dump(c_mats,open('confusion_matrix.p', "wb" ))
pickle.dump(genres,open('genres.p', "wb" ))






