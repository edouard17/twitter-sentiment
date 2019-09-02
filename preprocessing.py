import numpy as np
import pandas as pd
from utils import normalize_tweet, text_to_indices



def get_tweets(data_path):
	print("Getting tweets from dataset...")
	df = pd.read_csv(data_path, encoding='ISO-8859-1', header=None)
	# tweets are on the fifth column
	tweets = df[5]
	polarity = df[0]
	return tweets, polarity


def normalize_tweets(tweets):
	print("Normalizing tweets...")
	tweets_norm = []
	for tweet in tweets:
		tweet_norm = normalize_tweet(tweet)
		tweets_norm.append(tweet_norm)
	return tweets_norm


def get_max_len(tweets_norm):
	return len(max(tweets_norm, key=len))
