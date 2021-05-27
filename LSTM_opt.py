import os, pickle, tqdm
from word_processing import clean_words
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.callbacks import EarlyStopping

from scipy.optimize import minimize, brute
from scipy.optimize import OptimizeResult


try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data'
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

X_train, X_test, Y_train, Y_test = train_test_split(df1['plot'], df1['action'], train_size=0.8, random_state=1)

max_words = 1000
embedding_vec_len = 50
max_len = 150

lstm_units = 50
conv_filters = 30
kernel_size = 3


def RNN_CNN(x, X_train, X_test, Y_train, Y_test):
    # x = max_words, embedding_vec_len, max_len, lstm_units, conv_filters, kernel_size

    x = [int(i) for i in x]
    print(x)


    tok = Tokenizer(num_words=x[0])
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=x[2])

    model = Sequential()

    model.add(Embedding(input_dim=x[0],output_dim=x[1],input_length=x[2]))
    #model.add(Dropout(0.2))
    #model.add(Conv1D(filters=x[4], kernel_size=3, padding='same', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=x[3]))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    
    #compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #fit training data
    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    #process test set data
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=x[2])

    #evaluate model on test set
    accr = model.evaluate(test_sequences_matrix,Y_test)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    return 1 - accr[1]


x0 = [max_words]

#out = minimize(RNN_CNN, x0, method='BFGS', args=(X_train, X_test, Y_train, Y_test), options= {'eps':100})


ranges = ((500,2000),(30,60),(100,400),(20,80))

out = brute(func=RNN_CNN, ranges=ranges,args=(X_train, X_test, Y_train, Y_test), Ns=4, full_output=True)
print(out)
print()
print(out.x0)
print()
print(out.grid)