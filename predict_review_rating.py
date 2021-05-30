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
    pth = 'C:/Users/Tim/Documents/MA3831 Data/A3'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/A3'
    os.chdir(pth)
    
user_rating = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
user_rating2 = [1,2,3,4,5,6,7,8,9,10]

def predict_review_rating(txt):

    tok = pickle.load(open("tok.p","rb"))

    json_file = open('lstm.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('lstm.h5') 
    
    #Process plot
    review = txt.split('. ')
    review = ' '.join([clean_words(x) for x in review]).strip()
    print(review)
    # print(len(plot))
    review = [review]

    tok.fit_on_texts(review)
    sequences = tok.texts_to_sequences(review)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=100)
        
    pred = model.predict(sequences_matrix)
    
    return pred

#Shawshank Redemption
review = 'The Shawshank Redemption has great performances, extremely well written script and story all leading to a deeply emotional climax! One of the best dramas of all time!'
review = "Pulp Fiction may be the single best film ever made, and quite appropriately it is by one of the most creative directors of all time, Quentin Tarantino. This movie is amazing from the beginning definition of pulp to the end credits and boasts one of the best casts ever assembled with the likes of Bruce Willis, Samuel L. Jackson, John Travolta, Uma Thurman, Harvey Keitel, Tim Roth and Christopher Walken. The dialog is surprisingly humorous for this type of film, and I think that's what has made it so successful. Wrongfully denied the many Oscars it was nominated for, Pulp Fiction is by far the best film of the 90s and no Tarantino film has surpassed the quality of this movie (although Kill Bill came close). As far as I'm concerned this is the top film of all-time and definitely deserves a watch if you haven't seen it."

out = predict_review_rating(review)

list1, list2, list3 = zip(*sorted(zip(list(out[0]), user_rating, user_rating2)))
out, user_rating, user_rating2 = (list(t) for t in zip(*sorted(zip(list1, list2, list3))))
out.reverse()
user_rating.reverse()
user_rating2.reverse()

for i in range(len(user_rating)):
    print('{0} : {1}'.format(user_rating[i],out[i]))
