import os, pickle, tqdm
from word_processing import clean_words
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

plt.hist(x=genres)
plt.show()


genres = list(filter(None, genres))
genres = list(OrderedDict.fromkeys(genres))
genres = sorted(genres)

words = list(df1['plot'])
p_lengths = [len(i.split(' ')) for i in words]

words = list(df2['review'])
r_lengths = [len(i.split(' '))  for i in words]

plt.figure()
plt.hist(x=p_lengths,bins=100,range=(min(p_lengths),max(p_lengths)))
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Plot Lengths')

plt.figure()
plt.hist(x=r_lengths,bins=100,range=(min(p_lengths),max(p_lengths)))
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Review Lengths')
plt.show()

all_words = ' '.join(words)
print(len(list(set(words))))


