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
    #print(plot)
    # print(len(plot))
    plot = [plot]

    tok.fit_on_texts(plot)
    sequences = tok.texts_to_sequences(plot)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=100)
        
    pred = model.predict(sequences_matrix).tolist()
    pred = pred[0]
    
    list1, list2 = zip(*sorted(zip(pred, genres)))
    out, genres = (list(t) for t in zip(*sorted(zip(list1, list2))))
    out.reverse()
    genres.reverse()
    
    return [genres,out]

#A mid summernights dream (Book)
plot1 = 'Four Athenians run away to the forest only to have Puck the fairy make both of the boys fall in love with the same girl. \nThe four run through the forest pursuing each other while Puck helps his master play a trick on the fairy queen. \nIn the end, Puck reverses the magic, and the two couples reconcile and marry.'

#Harry Potter and The Philosphers Stone
plot2 = "It is a story about Harry Potter, an orphan brought up by his aunt and uncle because his parents were killed when he was a baby. \nHarry is unloved by his uncle and aunt but everything changes when he is invited to join \nHogwarts School of Witchcraft and Wizardry and he finds out he's a wizard."

out1 = predict_genres(plot1)
out2 = predict_genres(plot2)

print()
print(plot1)
for i in range(len(genres)):
    print('{0} : {1}'.format(out1[0][i],out1[1][i]))
          
          
print()
print(plot2)
for i in range(len(genres)):
    print('{0} : {1}'.format(out2[0][i],out2[1][i]))
    
print()
          
          
    
