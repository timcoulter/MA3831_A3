import asyncio, aiohttp, aiohttp_retry, os, bs4, requests, re, pickle, time
import pandas as pd

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data'
    os.chdir(pth)
    
try:
    dataset = pickle.load(open("movie_data.p","rb"))
except Exception:
    cnames = ['title', 'votes', 'rating', 'url']
    dataset = pd.DataFrame(columns=cnames)
    pickle.dump(dataset,open('movie_data.p', "wb" ))

main_url = 'https://www.imdb.com/search/title/?title_type=feature&num_votes=1000,&has=plot&view=simple&sort=user_rating,desc&count=250&ref_=adv_prv'
sz = 0
sz_prev = 1000000

while sz != sz_prev:
    r = requests.get(main_url)

    soup = bs4.BeautifulSoup(r.content,'lxml')

    nav = soup.find_all("a", {"class" : 'lister-page-next next-page'})
    nav_link = 'https://www.imdb.com' + nav[0]['href']
    
    total = soup.find("div", {"class" : 'desc'})
    txt = total.contents[1].contents[0]
    
    print(txt)
    
    table = soup.find_all("div", {"class": 'lister list detail sub-list'})
    table = table[0]
    all_rows = table.find_all("div", {"class" : 'lister-col-wrapper'})

    url = list()
    title = list()
    rating = list()
    votes = list()

    for row in all_rows:
        
        contents = str()
        ratings = row.find("div", {'class' : 'col-imdb-rating'})
        for r in ratings.contents:
            contents = contents + str(r)
            
        quoted = re.compile('"[^"]*"')
        out1 = quoted.findall(contents)[0]
        
        if out1 == '"ghost"':
            continue
        
        out = out1[1:-2].split(' base on ')
        rating.append(float(out[0]))
        out = out[1].split(' ')
        votes.append(int(out[0].replace(',','')))
        
        #Get link to page
        links = row.a
        title.append(links.text)
        url.append('https://www.imdb.com' + links.attrs['href'])
        
    df = pd.DataFrame({'title' : title, 'votes' : votes, 'rating' : rating, 'url' : url})
    dataset = pickle.load(open("movie_data.p","rb"))
    dataset = pd.concat([dataset,df])
    dataset.drop_duplicates()
    sz_prev = sz
    sz = dataset.shape[0]
    
    print(sz)
    
    pickle.dump(dataset,open('movie_data.p', "wb" ))
    
    time.sleep(3)
    
    main_url = nav_link

        
        