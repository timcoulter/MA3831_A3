import nltk
#nltk.download()
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
import string

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re


lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def clean_words(s):
    # Remove Unicode
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)
    # Remove Mentions
    s = re.sub(r'@\w+', '', s)
    # Lowercase the document
    s = s.lower()
    # Remove punctuations
    s = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', s)
    # Lowercase the numbers
    s = re.sub(r'[0-9]', '', s)
    # Remove the doubled space
    s = re.sub(r'\s{2,}', ' ', s)
                    
    s = word_tokenize(s)
    #Remove stopwords and lemmatize with tags
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in s if word not in all_stopwords and word != '']
    words = list(set(words))
    words = ' '.join(words)
    return words

def get_linked_words(responses):
    linked_words = list()
    for response in responses:
        for k in response['response']['docs']:
            linked_words.append(clean_words(k['suggestall']))
    return linked_words