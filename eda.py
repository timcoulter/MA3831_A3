import os, pickle, tqdm
from word_processing import clean_words
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/A3'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/A3'
    os.chdir(pth)
    
df1 = pickle.load(open("plot_df2.p","rb"))
df1 = df1.reset_index(drop=True)

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)
    
df2 = pickle.load(open("review_df2.p","rb"))
df2 = df2.reset_index(drop=True)

print(df1.shape)
print(df2.shape)

genres = list(df1['genre'].unique())
genres = ', '.join(genres).lower().strip()
genres = genres.split(', ')

# plt.hist(x=genres,bins=len(list(set(genres)))-1, edgecolor='black', linewidth=1.2,align='left')
# plt.title('Histogram of Movie Genre Frequency')
# plt.xlabel('Genre')
# plt.ylabel('Frequency')
# plt.savefig("genre_freq.png")

genres = list(filter(None, genres))
genres = list(OrderedDict.fromkeys(genres))
genres = sorted(genres)

words = list(df1['plot'])
p_lengths = [len(i.split(' ')) for i in words]

words = list(df2['review'])
r_lengths = [len(i.split(' '))  for i in words]

plt.figure()
plt.hist(x=p_lengths,bins=100,range=(min(p_lengths),max(p_lengths)),edgecolor='black', linewidth=1.2,align='left')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Plot Lengths')
plt.savefig("plot_lengths.png")

plt.figure()
plt.hist(x=r_lengths,bins=100,range=(min(p_lengths),max(p_lengths)),edgecolor='black', linewidth=1.2,align='left')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Review Lengths')
plt.savefig("review_lengths.png")

all_words = ' '.join(words)
print(len(list(set(words))))

ratings = list(df2['user_rating'])
ratings = [int(x) for x in ratings]
ratings = sorted(ratings)
ratings = [str(x) for x in ratings]

plt.figure()
plt.hist(x=ratings,bins=len(list(set(ratings))), edgecolor='black', linewidth=1.2)
plt.title('Histogram of Movie Rating Frequency')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig("rating_freq.png")

plt.show()

