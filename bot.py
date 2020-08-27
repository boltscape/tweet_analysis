import csv
import datetime
import json
import os
import shutil
import time

# Twitter API Client
import tweepy

def initiate_api():
    try: 
        with open('config.json', 'r') as f:
            config = json.load(f)        
        auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
        auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
        api = tweepy.API(auth)
        return api
    except:
        print("Problems with config.json")
        return None

# Filtering only english tweets
def isEnglish(twext):
    try:
        twext.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

#WOEID is a unique identifier in the Twitter API for locations around the world.

#Get the WOEIDs for the specified list of locations
def get_woeid(api, locations):
    world = api.trends_available()
    places = {place['name'].lower() : place['woeid'] for place in world};
    woeids = []
    for location in locations:
        if location in places:
            woeids.append(places[location])
        else:
            print("err: ",location," woeid does not exist in trending topics")
    return woeids

# Fetching popular tweets from API
def get_tweets(api, query):
    tweets = []
    for status in tweepy.Cursor(api.search, q=query, count=1000, result_type='popular', include_entities=True, monitor_rate_limit=True,  wait_on_rate_limit=True, lang="en").items():  
        # Fetching only English tweets
        if isEnglish(status.text):
            tweets.append([status.id_str, query, status.created_at.strftime('%d-%m-%Y %H:%M'), status.user.screen_name, status.text])
    return tweets

# Fetch trending hashtags
def get_hashtags(api, location):
    woeids = get_woeid(api, location)
    trending_tags = set()
    for woeid in woeids:
        try:
            trends = api.trends_place(woeid)
        except:
            print("API limit exceeded. Trying again in 1 hour.")
            time.sleep(3605)
            trends = api.trends_place(woeid)
        # Filtering out only English tweets and hash symbols
        topics = [trend['name'][1:] for trend in trends[0]['trends'] if (trend['name'].find('#') == 0 and isEnglish(trend['name']) == True)]
        trending_tags.update(topics)
    
    return trending_tags

def twitbot(api, locations):
    #Refresh Tweets and Hashtags
    shutil.rmtree("trending_tweets")
    os.makedirs("trending_tweets")
    
    tweets_file = open("trending_tweets/tweets.csv", "a+")
    hashtags_file = open("trending_tweets/hashtags.csv", "w+")
    tweesv = csv.writer(tweets_file)
    
    hashtags = get_hashtags(api, locations)
    hashtags_file.write("\n".join(hashtags))
    print("Hashtags written to file.")
    hashtags_file.close()
    
    for hashtag in hashtags:
        try:
            print("Getting tweets with hashtag ", hashtag)
            tweets = get_tweets(api, "#"+hashtag)
        except:
            print("Too many calls made to API. Try again after 1 hour.")
            time.sleep(3605)
            tweets = get_tweets(api, "#"+hashtag)
        for tweet in tweets:
            tweesv.writerow(tweet)
    
    tweets_file.close()

def main():
    locations = ['india'] #Add more locations as you see fit
    api = initiate_api()
    twitbot(api, locations)

if __name__ == '__main__':
    main()