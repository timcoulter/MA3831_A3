import os, pickle,tqdm

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


from word_processing import clean_words
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from word_processing import clean_words

try:
    pth = 'C:/Users/Tim/Documents/MA3831 Data/movie_data_new/L_func'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/movie_data_new/L_func'
    os.chdir(pth)

user_rating = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

def predict_review_rating(txt):
    
    user_rating = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    user_rating2 = [1,2,3,4,5,6,7,8,9,10]


    tok = pickle.load(open("tok.p","rb"))

    json_file = open('lstm.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('lstm.h5') 
    
    #Process plot
    review = txt.split('. ')
    review = ' '.join([clean_words(x) for x in review]).strip()
    #print(review)
    # print(len(plot))
    review = [review]

    tok.fit_on_texts(review)
    sequences = tok.texts_to_sequences(review)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=100)
        
    pred = model.predict(sequences_matrix)
    
    list1, list2, list3 = zip(*sorted(zip(list(pred[0]), user_rating, user_rating2)))
    pred, user_rating, = (list(t) for t in zip(*sorted(zip(list1, list2))))
    pred.reverse()
    user_rating.reverse()
    
    return [user_rating,pred]

#A Mid Summer Nights dream
#Positive Review
review1 = "I absolutely LOVED this book. So adventures and deep. \nThe detail of description really helped me enjoy and full fill my book reading addiction, \nI would totally recommend this book to anyone who likes deep literature and a tale for love."

#Negative Review
review2 = "I think the story was diffcult why the many story lines it was too weird and tough"

out1 = predict_review_rating(review1)
out2 = predict_review_rating(review2)

print()
print(review1)
for i in range(len(user_rating)):
    print('{0} : {1}'.format(out1[0][i], out1[1][i]))
print()
    
print(review2)
for i in range(len(user_rating)):
    print('{0} : {1}'.format(out2[0][i], out2[1][i]))
    
print()




