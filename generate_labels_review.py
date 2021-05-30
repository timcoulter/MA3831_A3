import os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
from word_processing import clean_words
from collections import OrderedDict

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new2'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new2'
    os.chdir(pth)
    
def add_review_columns(df):

    #process user ratings
    user_rating = list(df['user_rating'].unique())
    user_rating = list(filter(None, user_rating))
    user_rating = list(OrderedDict.fromkeys(user_rating))
    user_rating = [int(x) for x in user_rating]
    user_rating = sorted(user_rating)
    user_rating = [str(x) for x in user_rating]

    #make categories
    user_rating2= ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    #add columns to df
    for r in user_rating2:
        df[r] = 0
        
    print(df.columns)
    
    #update new df
    df_ratings = list(df['user_rating'])
    for i in tqdm.tqdm(range(len(df_ratings))):
        g = df_ratings[i]
        for j in range(len(user_rating)):
            if user_rating[j] == g:
                df.loc[i,user_rating2[j]] = 1
                break
    
    return df

df2 = pickle.load(open("review_df.p","rb"))
df2 = df2[df2['user_rating'] != '']
df2 = df2.reset_index(drop=True)
df2 = add_review_columns(df2)

print(df2.columns)
print(df2.head())

pickle.dump(df2,open('review_df2.p', "wb" ))












