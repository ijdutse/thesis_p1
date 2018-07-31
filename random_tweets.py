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
        text = f.readlines()

        # 1. return tweets devoid of duplicates or near duplicates:
        tweet1, tweet2, m_score, relevant_tweets = get_tweets_matching(text) #print('The two random tweets have a matching score of: {}'.format(m_score))
        
        # 2. convert the relevant tweets to tfidf matrix for ease of similarity computation:
        tweets_tfidf = get_tfidf_matrix(relevant_tweets) 
        #print(tweets_tfidf)
        #print(tweets_tfidf.toarray())
        
        #3. print the similarities matrix of the first 5 tweets with the rest:
        #print(cosine_similarity(tweets_tfidf.toarray()[0:5], tweets_tfidf))
        
        # map respective similarity score to corresponding tweet .... 
        #x = score_tweet_mapper()


# FUNCTION TO PICK RANDOM TWEETS AND COMPUTE THE MATCHING SCORE ... matching score > 50 are likely to be same/duplicate ... hence discarded
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
    return tweet1, tweet2, matching_score, relevant_tweets
    # [score for score in scores if score>30], keep


# TRANSFORM TWEETS TO USING TFIDF SCHEME ... numeric form for ease of computation ... 

def get_tfidf_matrix(tweets):
	# using tfidf:
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(tweets) #print(tfidf_matrix.shape), print(vectorizer.get_feature_names())
	
	# using countvectorizer:
	#count_vect = CountVectorizer()
	#words_count = count_vect.fit_transform(tweets) #concise_X = squareform(pdist(bag_of_words.toarray(), 'cosine'))
	#concise_words_count = squareform(pdist(words_count.toarray()))

	return tfidf_matrix, #words_count, concise_words_count #return tfidf_matrix, tfidf_matrix.toarray().flatten()[:5]


# COMPUTE THE SIMILARITY BETWEEN TWO RANDOM TWEETS USING THE COSINE MEASURE
#def get_tweets_sim(x, y):
#	return lambda document: round()cosine_similarity(x,y)
	# IF USING transform_tweets function:
	#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	#print(cosine_similarity(concise_X[0:2], concise_X))
	#print(cosine_similarity(concise_X[:], concise_X))
	# using the TFIDF Matrix to compute similarities:
	#sim1 = cosine_similarity(tfidf_matrix[0:5], tfidf_matrix)
	# using concise_words_count to compute similaritiy:

	#sim2 = cosine_similarity(concise_X[:,:], concise_X)
	
	# COMPUTE THE ANGLE BETWEEN THE TWO TWEETS:
	#co_sim = #sim_score of any document in the matrix
	#radian_angle = math.acos(co_sim)
	#degree_angle = math.degrees(radian_angle)
	#print(degree_angle)

#def score_tweet_mapper():
#	x = map (lambda tweets_doc: round(cosine_similarity(tweets_tfidf, compare_tweet),3)



# INSTANTITATE THE MAIN FUNCTION
if __name__=='__main__':
	main()
