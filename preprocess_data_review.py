import os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
from word_processing import clean_words

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new2'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new2'
    os.chdir(pth)

pd.set_option('mode.chained_assignment', None)
    
df = pickle.load(open("movie_data2.p","rb"))
print(df.columns)
#print(df.head())

columns_review = ['title', 'votes', 'rating', 'url', 'genre','review', 'user_rating']


title = list(df['title'])

#lemmatize sentences and paragraphs for reviews and plots
# try:
#     plot_df = pickle.load(open("plot_df.p","rb"))
# except Exception:
#     plot_df = pd.DataFrame(columns=columns_plot)
    
try:
    review_df = pickle.load(open("review_df.p","rb"))
except Exception:
    review_df = pd.DataFrame(columns=columns_review)
    
    
print(review_df.head())


for i in tqdm.tqdm(range(len(title))):
    
    processed_titles = list(review_df['title'])
    if title[i] in processed_titles:
        continue

    X = df.loc[df['title'] == title[i]]
    X = X.reset_index()
    X = X.head(1)
    #p = X.loc[0,'plot']
    r = X.loc[0,'review']
    rating = X.loc[0,'user_rating']
    
    #p = p.replace('...','.').split('<>')
    r = r.replace('...','.').split('<>')
    rating = rating.split('<>')
    
    if len(r) > len(rating):
        r = r[0:len(rating)]
    if len(rating) > len(r):
        rating = rating[0:len(r)]
    
    for k in range(len(r)):
        r2 = r[k].split('. ')
        r2 = ' '.join([clean_words(x,remove_stopwords=False) for x in r2])
        X2 = X[columns_review]
        X2.loc[0,'review'] = r2
        X2.loc[0,'user_rating'] = rating[k]
        review_df = pd.concat([review_df,X2])
        
    #pickle.dump(plot_df,open('plot_df.p', "wb" ))
    pickle.dump(review_df,open('review_df.p', "wb" ))
        
        



    
