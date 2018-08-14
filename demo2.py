#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
import os
from difflib import SequenceMatcher as sm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.spatial.distance import pdist, squareform
np.set_printoptions(precision=2)
import preprocessor as p
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
import time
from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds

file_path ='/home/ijdutse/tweets_separation_degrees/short_tweets.csv'

# ... crux of the activity!
def main():
	tweets_stream = get_tweets_stream('short_tweets.csv')

	vectorizer = HashingVectorizer()
	x_axis = []
	similarity_scores = []
	mapper = defaultdict(list)
	ld, ls = [],[]
	for _ in range(10):
	    cdate, tweet = get_mini_batch(tweets_stream,10)
	    vectorised_tweet = vectorizer.transform(tweet)
	    for date_item in cdate:
	        if date_item not in x_axis:
	            x_axis.append(date_item)
	    doc_len = vectorised_tweet.shape[0]

	    # call svd:
	    reduced_vectorised_tweet = get_projected_matrix(vectorised_tweet)

	    for date_item, single_tweet_tfidf in zip(cdate, reduced_vectorised_tweet):
	        tweets_sim = np.round(cosine_similarity(single_tweet_tfidf.reshape(1,-1),\
	                                reduced_vectorised_tweet, 3))
	        date_score = date_item, np.mean(tweets_sim)
	        ld.append(date_item)
	        ls.append(np.mean(tweets_sim))
	    plt.plot(ld,ls)
	    #plt.ylim(0.0,1.0)
	    plt.xticks(rotation=90)
	    plt.show()
	    #break

	



def get_tweets_stream(data_file):
    with open(data_file, 'r', encoding='utf-8') as csv:
        next(csv) # this skips the header in the file
        for line in csv:
            cdate, tweet = line[:19], line[20:-1]
            yield cdate, p.clean(tweet)

def get_mini_batch(tweet_stream, batch_size):
    cdates, tweets = [],[]
    try:
        for _ in range(batch_size):
            cdate, tweet = next(tweet_stream)
            cdates.append(cdate)
            tweets.append(tweet)
    except StopIteration:
        return None, None
    return cdates, tweets

def get_projected_matrix(tfidf_matrix, k=5):
	tfidf_matrix = tfidf_matrix.tocsc()
	U, sigma, Vt = svds(tfidf_matrix, k)
	#sigma=np.diag(sigma)

	projected_matrix = tfidf_matrix.T*U/sigma
	return projected_matrix


# instantiate ..... !
if __name__=='__main__':
	main()

########### EARLIER FUNCTIONS/SCRIPTS ####################
# a function pick random tweets and compute matching score in order to discard duplicates ... matching score > 50 are likely to be same/duplicate ...
"""
def get_tweets_matching(tweets):
    scores = []
    relevant_tweets = []
    k = len(tweets)
    tracker = len(relevant_tweets)
    while tracker < k:
    	random.shuffle(tweets)
    	tweet1 = p.clean(str(tweets[0])).lower()
    	tweet2 = p.clean(str(tweets[1])).lower()
    	sim = sm(None, tweet1, tweet2)
    	matching_score = sim.ratio()
    	scores.append(matching_score)
    	if matching_score < 60:
    		relevant_tweets.append(tweet1)
    	elif tweet1 or tweet2 in relevant_tweets:
    		pass
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
		#incoming_tweet = input('Provide a sample tweet to compare with existing tweets :::')
		#transformed_incoming_tweet = vectorizer.transform([incoming_tweet])

	# cosine similarity computation:
	tweets_similarity = np.round(cosine_similarity(existing_tweets_tfidf_matrix, existing_tweets_tfidf_matrix), 4)
	indices_of_most_similar_tweets = tweets_similarity.argsort()[:,:-1]
	#print('Indices for the most tweets in the corpus ===>',tweet_similarity[indices_of_most_similar_tweets], tweets_similarity[indices_of_most_similar_tweets], sep='\n')
	return existing_tweets_tfidf_matrix, tweets_similarity, indices_of_most_similar_tweets # transformed_incoming_tweet.toarray(), 

# alternate transformation using Countvectrorizer and sqaureform/pdist from scipy::
def get_words_count_matrix(tweets):
	count_vectorizer = CountVectorizer()
	words_count = count_vectorizer.fit_transform(tweets)
	#use the scipy squareform to return concise version of the words from pairwise distance(pdist) space ... pdist takes numerous distance metrics e.g. 'euclidean, cosine'
	dense_words_matrix = squareform(pdist(words_count.toarray(), 'cosine'))
	return dense_words_matrix 

"""
