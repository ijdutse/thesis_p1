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
        tweets = f.readlines()
        
        # discard duplicate tweets or near duplicate based on sequence matching:
        relevant_tweets = get_tweets_matching(tweets)

        # transform tweets to tfidf matrix and compute cosine similarities:
        existing_tweets_tfidf_matrix, transformed_incoming_tweet, tweet_similarity, indices_of_most_similar_tweets = get_tfidf_matrix_and_cosine(relevant_tweets)
        	#existing_tweets_tfidf_matrix, transformed_incoming_tweet, tweet_similarity = get_tfidf_matrix_and_cosine(relevant_tweets)
        	#sanity check ==> print(existing_tweets_tfidf_matrix, existing_tweets_tfidf_matrix.toarray(), existing_tweets_tfidf_matrix.toarray()[0:1], transformed_incoming_tweet, relevant_tweets, sep='\n')

        # uncomment me:
        	#for index in indices_of_most_similar_tweets[0]:
        		#print (relevant_tweets[index])
        		#break

        # compare an incoming tweet to existing tweets in iterative fashion:
        results = []
        new_tweet = transformed_incoming_tweet
        for tweet_tfidf in existing_tweets_tfidf_matrix:
        	tweets_cosine_sim = np.round(cosine_similarity(tweet_tfidf, new_tweet), 3)
        	results.append(tweets_cosine_sim[0][0])
        # sort the list of results in decreasing magnitude to return scores for most similar tweets ...
        most_similar_tweets = sorted(results, reverse = True)
        print(most_similar_tweets)

        
       #3: compare the scores of first n (e.g n=5) tweets or a single incoming tweet with the rest of existing tweets:
        	#n_tweets_similarities = np.round(cosine_similarity(existing_tweets_tfidf_matrix.toarray()[0:5], existing_tweets_tfidf_matrix), 4)
        	#print(n_tweets_similarities)
       		#tweets_cosine_sim = np.round(cosine_similarity(existing_tweets_tfidf_matrix, transformed_incoming_tweet), 3)
       		#print(tweets_cosine_sim)

  
        #4: A MAPPER FUNCTION .... to map respective similarity score to corresponding tweet .... 
        #x = score_tweet_mapper()

        #t = score_tweet_mapper()
        #print(t)


# A FUNCTION TO PICK RANDOM TWEETS AND COMPUTE THE MATCHING SCORE ... matching score > 50 are likely to be same/duplicate ... hence discarded
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
    return relevant_tweets # tweet1, tweet2, matching_score, [score for score in scores if score>30], keep


# TRANSFORM TWEETS USING TFIDF SCHEME and compute cosine similarity ... 

# using tfidf:
def get_tfidf_matrix_and_cosine(tweets):
	# TWEETS TFIDF MATRIX:
	vectorizer = TfidfVectorizer()
	existing_tweets_tfidf_matrix = vectorizer.fit_transform(tweets)
	# transform an incoming tweet for comparison with existing tweets ....
	incoming_tweet = input('Provide a sample tweet to compare with existing tweets :::')
	transformed_incoming_tweet = vectorizer.transform([incoming_tweet])
	# COSINE SIMILARITIES MATRIX:
	tweet_similarity = np.round(cosine_similarity(existing_tweets_tfidf_matrix, existing_tweets_tfidf_matrix), 4)
	indices_of_most_similar_tweets = tweet_similarity.argsort()[:,:-1]
	#print('Indices for the most tweets in the corpus ===>')
	#print(tweet_similarity[indices_of_most_similar_tweets])
	#print(indices_of_most_similar_tweets, existing_tweets_tfidf_matrix.shape, tweet_similarity[indices_of_most_similar_tweets], sep='\n')
	return existing_tweets_tfidf_matrix, transformed_incoming_tweet.toarray(), tweet_similarity, indices_of_most_similar_tweets

# alternate transformation using Countvectrorizer and sqaureform/pdist from scipy::
def get_words_count_matrix(tweets):
	count_vectorizer = CountVectorizer()
	words_count = count_vectorizer.fit_transform(tweets)
	#use the scipy squareform to return concise version of the words from pairwise distance(pdist) space ... pdist takes numerous distance metrics e.g. 'euclidean, cosine'
	dense_words_matrix = squareform(pdist(words_count.toarray(), 'cosine'))
	return dense_words_matrix 


# A MAPPER FUNCTION .... to map respective similarity score to corresponding tweet .... 
#def score_tweet_mapper(external_tweet):
#	x = map (lambda tweets_doc: round(cosine_similarity(tweets_tfidf, compare_tweet),3)




# INSTANTITATE THE MAIN FUNCTION
if __name__=='__main__':
	main()