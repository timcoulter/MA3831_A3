import os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
from word_processing import clean_words

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data'
    os.chdir(pth)

pd.set_option('mode.chained_assignment', None)
    
df = pickle.load(open("movie_data2.p","rb"))

#First fix the date and country

date = df['date']

country = list()
new_date = list()

for i in range(len(date)):
    country.append(date[i][date[i].find("(")+1:date[i].find(")")])
    x = date[i].split(' (')
    new_date.append(x[0])
    
df['date'] = new_date
df['country'] = country


columns_plot = ['title','votes','rating','length','genre','country','date','plot']
columns_review = ['title','votes','rating','length','genre','country','date','review']


title = list(df['title'])

#lemmatize sentences and paragraphs for reviews and plots
plot_df = pd.DataFrame(columns=columns_plot)
review_df = pd.DataFrame(columns=columns_review)


for i in tqdm.tqdm(range(len(title))):
    
    X = df.loc[df['title'] == title[i]]
    X = X.reset_index()
    X = X.head(1)
    p = X.loc[0,'plot']
    r = X.loc[0,'review']
    
    for i in range(len(p)):
        try:
            p2 = p[i].split('. ')
            p2 = ' '.join([clean_words(x) for x in p])
            X2 = X[columns_plot]
            X2.loc[0,'plot'] = p2
            plot_df = pd.concat([plot_df,X2])
        except Exception:
            print(title[i])

    for i in range(len(r)):
        try:
            r2 = r[i].split('. ')
            r2 = ' '.join([clean_words(x) for x in r])
            X2 = X[columns_review]
            X2.loc[0,'review'] = r2
            review_df = pd.concat([review_df,X2])
        except Exception:
            print(title[i])
        
pickle.dump(plot_df,open('plot_df.p', "wb" ))
pickle.dump(review_df,open('review_df.p', "wb" ))
        
        



    
