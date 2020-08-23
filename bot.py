import csv
import datetime
import json
import os
import time

import schedule
import tweepy


def initiate_api():
    try: 
        with open('config.json', 'r') as f:
            config = json.load(f)        
        auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
        auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
        api = tweepy.API(auth)
        #print("API connection successful")
        return api
    except:
        print("Problems with config.json")
        return None

# Customzizing fetched tweets 

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

