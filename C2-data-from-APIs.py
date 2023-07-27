import tweepy 
import time

BEARER_TOKEN = "YOUR_KEY_HERE"
ACCESS_TOKEN = "YOUR_KEY_HERE"
ACCESS_TOKEN_SECRET = "YOUR_KEY_HERE" 
CONSUMER_KEY = "YOUR_KEY_HERE"
CONSUMER_SECRET = "YOUR_KEY_HERE"

client = tweepy.Client(BEARER_TOKEN, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
auth = tweepy.OAuth1UserHandler(CONSUMER_KEY,CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

class TwitterStream(tweepy.StreamingClient):

    def on_tweet(self, tweet):
        print(tweet.text)
        time.sleep(0.2)

stream = TwitterStream(bearer_token=BEARER_TOKEN)
stream.add_rules(tweepy.StreamRule("#lfc"))
print(stream.get_rules())
stream.filter()
