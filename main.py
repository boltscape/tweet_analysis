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

def pre_process_tweet(tweet):
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
    
    """Stemming
    if stemlem == 'stem':
        ps = PorterStemmer()
        cleaned_words = [ps.stem(w) for w in filtered_words]
    """
    # Lemmatization
    #elif stemlem == 'lem':    
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(cleaned_words)

# Vectorize the tweet raw data
def vectorize(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

def sintiment_to_stringtiment(sentiment):
    if sentiment==0:
        return 'Negative'
    elif sentiment==2:
        return 'Neutral'
    else:
        return 'Positive'

# Load the dataset
primitive_tweetset = load_tweets("training.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
# Remove unnecessary columns
tweetset = clean_cols(primitive_tweetset, ['t_id', 'created_at', 'query', 'user'])
# Preprocess tweets
tweetset.text = tweetset['text'].apply(pre_process_tweet)

# Split tweets set into training and testing parts

tf_vector = vectorize(np.array(tweetset.iloc[:, 1]).ravel())
x = tf_vector.transform(np.array(tweetset.iloc[:, 1]).ravel())
y = np.array(tweetset.iloc[:, 0]).ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(x_train, y_train)
y_predict_nb = NB_model.predict(x_test)
print(accuracy_score(y_test, y_predict_nb))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(x_train, y_train)
y_predict_lr = LR_model.predict(x_test)
print(accuracy_score(y_test, y_predict_lr))