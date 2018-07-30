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


#file_path ='/home/ijdutse/thesis_p1/full_tweets.txt'
file_path ='/home/ijdutse/thesis_p1/short_tweets.txt'


# THE MAIN FUNCTION
def main():
    with open(file_path,'r') as f:
        text = f.readlines()
        x, y, m_score = get_tweets_matching(text)
        #print(x)
        #print(y)
        #print('The two random tweets have a matching score of: {}'.format(m_score))
        tf = get_tfidf_matrix((x,y))
        print(tf)
        #tf = transform_tweets(text)
        #print(tf)

# FUNCTION TO PICK RANDOM TWEETS AND COMPUTE THE MATCHING SCORE ... MATCHING SCORE > 50 ARE LIKELY TO BE SAME/DUPLICATE TWEETS, HENCE DISCARDED
def get_tweets_matching(tweets):
    scores = []
    keep = []
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
    		keep.append((tweet1,tweet2))
    	tracker +=1
    return tweet1, tweet2, matching_score #{matching_score}, (tweet1, tweet2)
    #return len(scores),scores, matching_score, tweet1, tweet2, [score for score in scores if score>30], keep


# TRANSFORM TWEETS TO NUMERIC FORM FOR EASE OF COMPUTATION

def get_tfidf_matrix(tweets):
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(tweets)
	#print(tfidf_matrix.shape), print(tfidf_matrix), print(len(vectorizer.get_feature_names())),print(vectorizer.get_feature_names())
	# USING THE COUNTVECTORIZER:
	count_vect = CountVectorizer()
	bag_of_words = count_vect.fit_transform(tweets)
	#concise_X = squareform(pdist(bag_of_words.toarray(), 'cosine'))
	concise_X = squareform(pdist(bag_of_words.toarray()))
	#print(concise_X.shape)
	#print(bag_of_words.toarray().shape)	
	small = concise_X[:6]
	#print(small)
	#print(cosine_similarity(concise_X[0:2], concise_X))
	#print(cosine_similarity(concise_X[:], concise_X))
	sim1 = cosine_similarity(concise_X[0:3], concise_X)
	sim2 = cosine_similarity(concise_X[:,:], concise_X)
	#return concise_X, bag_of_words
	return concise_X #small, sim1, sim2, 
	




# COMPUTE THE SIMILARITY BETWEEN TWO RANDOM TWEETS USING THE COSINE MEASURE
def get_tweets_sim(x, y):
	return cosine_similarity(x,y)
	# IF USING transform_tweets function:
	#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	# COMPUTE THE ANGLE BETWEEN THE TWO TWEETS:
	#co_sim = #sim_score of any document in the matrix
	#radian_angle = math.acos(co_sim)
	#degree_angle = math.degrees(radian_angle)
	#print(degree_angle)



# INSTANTITATE THE MAIN FUNCTION
if __name__=='__main__':
	main()
