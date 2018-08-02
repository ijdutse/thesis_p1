#!/usr/bin/env python3
# PACKAGES IMPORT
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


#file_path ='/home/ijdutse/tweets_separation_degrees/full_tweets.txt'
file_path ='/home/ijdutse/tweets_separation_degrees/short_tweets.txt'


# THE MAIN FUNCTION ... crux of the activity!
def main():
    with open(file_path,'r') as f:
        tweets = f.readlines()

        #1: return tweets devoid of duplicates or near duplicates:
        tweet1, tweet2, m_score, relevant_tweets = get_tweets_matching(tweets) 
        #sanity check ==> print(tweet1, tweet2,m_score, relevant_tweets, sep='\n') or _, _, _, relevant_tweets = get_tweets_matching(tweets)
        
        #2: convert relevant tweets (from #1) to tfidf matrix for ease of similarity computation:
        tweets_tfidf_matrix = get_tfidf_matrix(relevant_tweets) 
        #sanity check ==> print(tweets_tfidf_matrix, tweets_tfidf_matrix.toarray(), tweets_tfidf_matrix.toarray()[0:1], relevant_tweets, sep='\n')
        
        # using get_words_count_matrix:
        tweets_count_matrix = get_words_count_matrix(relevant_tweets)
        #sanity check ==> print(tweets_count_matrix, tweets_tfidf_matrix[0:1], relevant_tweets, sep='\n')
        
        #3: print the similarities matrix of the first 5 tweets with the rest using both get_tfidf_matrix and get_words_count_matrix functions:
        #tweet_similarity = np.round(cosine_similarity(tweets_tfidf_matrix.toarray()[0:5], tweets_tfidf_matrix), 4)
        #tweet_similarity = np.round(cosine_similarity(tweets_count_matrix[0:5], tweets_count_matrix), 4)
        #sanity check ==> print(tweet_similarity, relevant_tweets, sep='\n')
        # compute the angle between similar tweets:
		#co_sim = #sim_score of any document in the matrix
		#radian_angle = math.acos(co_sim)
		#degree_angle = math.degrees(radian_angle)
		#print(degree_angle)

  
        #4: A MAPPER FUNCTION .... to map respective similarity score to corresponding tweet .... 
        #x = score_tweet_mapper()


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
    return tweet1, tweet2, matching_score, relevant_tweets # [score for score in scores if score>30], keep


# TRANSFORM TWEETS TO USING TFIDF SCHEME ... numeric form for ease of computation ... 

# using tfidf:
def get_tfidf_matrix(tweets):
	vectorizer = TfidfVectorizer()
	tweets_tfidf_matrix = vectorizer.fit_transform(tweets)
	return tweets_tfidf_matrix

# using Countvectrorizer and sqaureform/pdist from scipy::
def get_words_count_matrix(tweets):
	count_vectorizer = CountVectorizer()
	words_count = count_vectorizer.fit_transform(tweets)
	#use the scipy squareform to return concise version of the words from pairwise distance(pdist) space ... pdist takes numerous distance metrics e.g. 'euclidean, cosine'
	dense_words_matrix = squareform(pdist(words_count.toarray(), 'cosine'))
	return dense_words_matrix 


# A MAPPER FUNCTION .... to map respective similarity score to corresponding tweet .... 
        #x = score_tweet_mapper()

#def score_tweet_mapper():
#	x = map (lambda tweets_doc: round(cosine_similarity(tweets_tfidf, compare_tweet),3)




# INSTANTITATE THE MAIN FUNCTION
if __name__=='__main__':
	main()