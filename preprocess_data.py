import os, bs4, requests, re, pickle, time
import pandas as pd

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data'
    os.chdir(pth)
    
    
df1 = pickle.load(open("movie_data.p","rb"))
df2 = pickle.load(open("movie_data2.p","rb"))

print(df1.head())
print(df2.head())