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
    pth = 'C:/Users/Tim/Documents/MA3831 Data/A3/multi_class_focal'
    os.chdir(pth)
except Exception:
    pth = 'C:/Users/timco/Documents/MA3831 Data/A3/multi_class_focal'
    os.chdir(pth)
    
genres = pickle.load(open("genres.p","rb"))

def predict_genres(txt):

    genres = pickle.load(open("genres.p","rb"))
    tok = pickle.load(open("tok.p","rb"))

    json_file = open('lstm.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('lstm.h5') 
    
    #Process plot
    plot = txt.split('. ')
    plot = ' '.join([clean_words(x) for x in plot]).strip()
    print(plot)
    # print(len(plot))
    plot = [plot]

    tok.fit_on_texts(plot)
    sequences = tok.texts_to_sequences(plot)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=100)
        
    pred = model.predict(sequences_matrix)
    
    return pred

#Pulp Fiction
plot = 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.'
out = predict_genres(plot)

list1, list2 = zip(*sorted(zip(list(out[0]), genres)))
out, genres = (list(t) for t in zip(*sorted(zip(list1, list2))))
out.reverse()
genres.reverse()

for i in range(len(genres)):
    print('{0} : {1}'.format(genres[i],out[i]))
    
