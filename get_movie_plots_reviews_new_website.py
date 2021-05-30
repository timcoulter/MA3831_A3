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
    df['review'] = ''
    df['user_rating'] = ''
    df['genre'] = ''
    
    pickle.dump(df,open('movie_data2.p', "wb" ))
    
    
df = df.iloc[0:2100]
urls = df.url

chrome_options = Options()
#chrome_options.add_argument("--headless")
#chrome_options.add_argument("--log-level=3");
driver = webdriver.Chrome(options=chrome_options)

for i in tqdm.tqdm(range(len(urls))):
    
    #Get movie length, genre and date
    x = df.loc[df['url'] == urls[i],'genre']
    y = x.iloc[0]
    if y == "":
        print(urls[i])
        r1 = requests.get(urls[i],verify=False)
        #r1 = requests.get('https://www.imdb.com/title/tt0111161/')
        driver.get(urls[i]);
        content = driver.page_source
        
        soup = bs4.BeautifulSoup(content,"lxml")
        subtext = soup.find("div", {"class" : "subtext"})
        
        cl = "ipc-metadata-list ipc-metadata-list--dividers-all Storyline__StorylineMetaDataList-sc-1b58ttw-1 esngIX ipc-metadata-list--base"
        
        subtext = soup.find("ul", {"class" : cl})
       
        genre = subtext.find_all('a')
        #date = genre[-1].string.strip()
        genre = [x.string for x in genre[:-1]]
        genre = ', '.join(genre)

        df.loc[df['url'] == urls[i],'genre'] = genre
    
    #Get top 10 helpful movie reviews
    x = df.loc[df['url'] == urls[i],'review']
    y = x.iloc[0]
    if y == '':
        r3 = requests.get(urls[i] + 'reviews?ref_=tt_urv')
        if r3.status_code != 200:
            continue
        
        N = 50
        soup2 = bs4.BeautifulSoup(r3.content,'lxml')
        reviews = soup2.find_all("div", {"class" : 'lister-item-content'})
        
        if len(reviews) > N:
            reviews = reviews[0:N]
        
        review = list()
        user_rating = list()
        
        for r in reviews:
            content = r.find("div", {"class" : "text show-more__control"})
            rating_content = r.find("span", {"class" : "rating-other-user-rating"})
            if rating_content == None:
                continue
            
            rating_content = rating_content.text.split('/')
            user_rating.append(re.sub('[^0-9]','',rating_content[0]))
            review.append(content.text)
                
        review = '<>'.join(review)
        user_rating = '<>'.join(user_rating)
        
        df.loc[df['url'] == urls[i],'review'] = review
        df.loc[df['url'] == urls[i],'user_rating'] = user_rating
        
    if i % 100 == 0:
        pickle.dump(df,open('movie_data2.p', "wb" ))

