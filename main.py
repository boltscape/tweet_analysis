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

# Twitter bot
import bot

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
    # Lemmatization  
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

# Training Logistic Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(x_train, y_train)

# Fetching tweets using the bot
bot.main()

# Opening the tweets fetched by the bot
test_file_name = "trending_tweets/tweets.csv"
test_ds = load_tweets(test_file_name, ["t_id", "hashtag", "created_at", "user", "text"])
test_ds = clean_cols(test_ds, ["t_id", "created_at", "user"])

# Creating text feature
test_ds.text = test_ds["text"].apply(pre_process_tweet)
test_feature = tf_vector.transform(np.array(test_ds.iloc[:, 1]).ravel())

# Using Logistic Regression model for prediction
test_prediction_lr = LR_model.predict(test_feature)

# Averaging out the hashtags result
test_result_ds = pd.DataFrame({'hashtag': test_ds.hashtag, 'prediction':test_prediction_lr})
test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
test_result.columns = ['hashtag', 'predictions']
test_result.predictions = test_result['predictions'].apply(sintiment_to_stringtiment)

print(test_result)