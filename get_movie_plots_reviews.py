import asyncio, aiohttp, aiohttp_retry, os, bs4, requests, re, pickle, time, tqdm
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import datetime as dt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new'
    os.chdir(pth)

try:
    df = pickle.load(open("movie_data2.p","rb"))
except Exception:
    df = pickle.load(open("movie_data.p","rb"))
    df = df.reset_index(drop=True)
    df['length'] = 0
    df['genre'] = ''
    df['date'] = ''
    df['plot'] = ''
    df['review'] = ''
    df['user_rating'] = ''
    
    pickle.dump(df,open('movie_data2.p', "wb" ))

print(df.head())
urls = df.url

chrome_options = Options()
#chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

for i in tqdm.tqdm(range(len(urls))):
    
    #Get movie length, genre and date
    x = df.loc[df['url'] == urls[i],'length']
    y = x.iloc[0]
    if y == 0:
        print(urls[i])
        r1 = requests.get(urls[i],verify=False)
        r1 = requests.get('https://www.imdb.com/title/tt0111161/')
        #print(r1.content)
        if r1.status_code != 200:
            continue

        driver.get(urls[i]);
        content = driver.page_source
        
        soup = bs4.BeautifulSoup(content,"lxml")
        subtext = soup.find("div", {"class" : "subtext"})

        length  = subtext.time.string.strip()
        length = length.split(' ')
        if len(length) == 1:
            length = 60*int(re.sub("[^0-9]", "", length[0]))
        else:
            length = 60*int(re.sub("[^0-9]", "", length[0])) + int(re.sub("[^0-9]", "", length[1]))

        genre = subtext.find_all('a')
        date = genre[-1].string.strip()
        genre = [x.string for x in genre[:-1]]
        genre = ', '.join(genre)

        df.loc[df['url'] == urls[i],'length'] = length
        df.loc[df['url'] == urls[i],'genre'] = genre
        df.loc[df['url'] == urls[i],'date'] = date
    
    #Get movie plot summaries
    x = df.loc[df['url'] == urls[i],'plot']
    y = x.iloc[0]
    if y == '':
        r1 = requests.get(urls[i] + 'plotsummary?ref_=tt_stry_pl')
        if r1.status_code != 200:
            continue

        soup1 = bs4.BeautifulSoup(r1.content,'lxml')

        summaries = soup1.find("ul", {'id': 'plot-summaries-content'})
        summaries = summaries.find_all('li', {})

        summary = list()
        for s in summaries:
            txt = s.text.split('\n\n')
            summary.append(txt[0].strip())
        summary = '<>'.join(summary)
        df.loc[df['url'] == urls[i],'plot'] = summary
    
    #Get top 10 helpful movie reviews
    x = df.loc[df['url'] == urls[i],'review']
    y = x.iloc[0]
    if y == '':
        r3 = requests.get(urls[i] + 'reviews?ref_=tt_urv')
        if r3.status_code != 200:
            continue
        
        soup2 = bs4.BeautifulSoup(r3.content,'lxml')
        reviews = soup2.find_all("div", {"class" : 'text show-more__control'})
        if len(reviews) > 10:
            reviews = reviews[0:10]
        review = list()
        
        for r in reviews:
            review.append(r.text)
        review = '<>'.join(review)
        df.loc[df['url'] == urls[i],'review'] = review
        
        user_ratings = soup2.find_all("svg", {"class" : "ipl-icon ipl-star-icon"})
        if len(user_ratings) > 10:
                user_ratings = user_ratings[0:10]
        user_rating = list()
        
        for u in user_ratings:
            print()
        print()
        

    if i % 100 == 0:
        pickle.dump(df,open('movie_data2.p', "wb" ))

