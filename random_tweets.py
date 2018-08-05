#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
import os
from difflib import SequenceMatcher as sm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.spatial.distance import pdist, squareform
np.set_printoptions(precision=2)

file_path ='/home/ijdutse/tweets_separation_degrees/short_tweets.txt'

# ... crux of the activity!
def main():
    with open(file_path,'r') as f:
    	tweets = f.readlines() # discard duplicate tweets or near duplicate based on sequence matching:
    	relevant_tweets = get_tweets_matching(tweets)
    	# transform tweets to tfidf matrix and compute cosine similarities:
    	existing_tweets_tfidf_matrix, transformed_incoming_tweet, tweets_similarity, indices_of_most_similar_tweets = get_tfidf_matrix_and_cosine(relevant_tweets)
    	# compare the scores of first n (e.g n=5) tweets or a single incoming tweet with the rest of existing tweets:
    	n_tweets_similarities = np.round(cosine_similarity(existing_tweets_tfidf_matrix.toarray()[0:5], existing_tweets_tfidf_matrix), 4)
    	tweets_cosine_sim = np.round(cosine_similarity(existing_tweets_tfidf_matrix, transformed_incoming_tweet), 3)
    	#print(print(n_tweets_similarities, tweets_cosine_sim, sep='\n')

        # compare an incoming tweet to existing tweets in iterative fashion:
    	results = []
    	new_tweet = transformed_incoming_tweet
    	for tweet_tfidf in existing_tweets_tfidf_matrix:
    		tweets_cosine_sim = np.round(cosine_similarity(tweet_tfidf, new_tweet), 3)
    		results.append(tweets_cosine_sim[0][0])
    	#sort the list of results in decreasing magnitude to return scores for most similar tweets ...
    	most_similar_tweets = sorted(results, reverse = True)
    	# map the highest scores to corresponding tweets: 
    	#print('The following tweets are the most similar to {} :'.format(input()))
    	for tweet, score in zip(relevant_tweets, most_similar_tweets):
    		index = most_similar_tweets.index(score)
    		print(index, relevant_tweets[index], score, sep=':')

# a function pick random tweets and compute matching score in order to discard duplicates ... matching score > 50 are likely to be same/duplicate ...
def get_tweets_matching(tweets):
    scores = []
    relevant_tweets = []
    k = len(tweets)
    tracker = 0
    while tracker < k:
    	random.shuffle(tweets)
    	tweet1 = tweets[0]
    	tweet2 = tweets[1]
    	sim = sm(None, tweet1, tweet2)
    	matching_score = round(sim.ratio()*100,2)
    	scores.append(matching_score)
    	if matching_score > 60:
    		continue
    	else:
    		relevant_tweets.append(tweet1)
    		relevant_tweets.append(tweet2)
    	tracker +=1
    return relevant_tweets

# transform tweets using TFIDF scheme and compute cosine similarity ... 
def get_tfidf_matrix_and_cosine(tweets):
	vectorizer = TfidfVectorizer()
	existing_tweets_tfidf_matrix = vectorizer.fit_transform(tweets)

	# transform an incoming tweet for comparison with existing tweets ....
	incoming_tweet = input('Provide a sample tweet to compare with existing tweets :::')
	transformed_incoming_tweet = vectorizer.transform([incoming_tweet])

	# cosine similarity computation:
	tweets_similarity = np.round(cosine_similarity(existing_tweets_tfidf_matrix, existing_tweets_tfidf_matrix), 4)
	indices_of_most_similar_tweets = tweets_similarity.argsort()[:,:-1]
	#print('Indices for the most tweets in the corpus ===>',tweet_similarity[indices_of_most_similar_tweets], tweets_similarity[indices_of_most_similar_tweets], sep='\n')
	return existing_tweets_tfidf_matrix, transformed_incoming_tweet.toarray(), tweets_similarity, indices_of_most_similar_tweets

# alternate transformation using Countvectrorizer and sqaureform/pdist from scipy::
def get_words_count_matrix(tweets):
	count_vectorizer = CountVectorizer()
	words_count = count_vectorizer.fit_transform(tweets)
	#use the scipy squareform to return concise version of the words from pairwise distance(pdist) space ... pdist takes numerous distance metrics e.g. 'euclidean, cosine'
	dense_words_matrix = squareform(pdist(words_count.toarray(), 'cosine'))
	return dense_words_matrix 


# instantiate ..... !
if __name__=='__main__':
	main()