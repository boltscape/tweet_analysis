import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Params
stop_words = set(stopwords.words('english'))

# Load tweet dataset
def load_tweets(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

# Clean dataset columns
def clean_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def preprocess_tweet_text(tweet,stemlem):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove username tags and hashtags
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    #Stemming
    if stemlem == 'stem':
        ps = PorterStemmer()
        cleaned_words = [ps.stem(w) for w in filtered_words]
    elif stemlem == 'lem':    
        lemmatizer = WordNetLemmatizer()
        cleaned_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(cleaned_words)