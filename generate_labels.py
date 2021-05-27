import os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
from word_processing import clean_words
from collections import OrderedDict

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data'
    os.chdir(pth)
    
def add_genre_columns(df):
    #get unique genres in list
    genres = list(df['genre'].unique())
    genres = ', '.join(genres).lower().strip()
    genres = genres.split(', ')
    genres = list(filter(None, genres))
    genres = list(OrderedDict.fromkeys(genres))
    genres = sorted(genres)
    
    #print(genres)

    #add column for each genre
    for g in genres:
        df[g] = 0
        
    df_genres = list(df['genre'])
    for i in tqdm.tqdm(range(len(df_genres))):
        g = df_genres[i].lower().split(', ')
        for k in genres:
            if k in g:
                #print([i,k])
                df.loc[i,k] = 1      
    return df
    
    
df1 = pickle.load(open("plot_df.p","rb"))
df2 = pickle.load(open("review_df.p","rb"))
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

df1 = add_genre_columns(df1)
df2 = add_genre_columns(df2)

pickle.dump(df1,open('plot_df2.p', "wb" ))
pickle.dump(df2,open('review_df2.p', "wb" ))












