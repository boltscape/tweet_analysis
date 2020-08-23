import tweepy
import json
import schedule
import time
import datetime
import os
import csv

def initiate_api():
    try: 
        with open('config.json', 'r') as f:
            config = json.load(f)        
        auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
        auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
        api = tweepy.API(auth)
        print("API connection successful")
        return api
    except:
        print("Problems with config.json")
        return None

initiate_api()