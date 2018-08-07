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
import preprocessor as p
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
import time

file_path ='/home/ijdutse/tweets_separation_degrees/short_tweets.txt'
#file_path ='/home/ijdutse/tweets_separation_degrees/full_tweets.txt'

# ... crux of the activity!
def main():
    with open(file_path,'r') as f:
        tweets = f.readlines()
        print('Some background tasks currently running: tweets preprocessing, transformation, similarity computation and visualisation of results ....')
        relevant_tweets = get_tweets_matching(tweets)
        existing_tweets_tfidf_matrix, tweets_similarity, indices_of_most_similar_tweets = get_tfidf_matrix_and_cosine(relevant_tweets)
        # compare the scores of first n (e.g n=5) tweets or a single incoming tweet with the rest of existing
        n_tweets_similarities = np.round(cosine_similarity(existing_tweets_tfidf_matrix.toarray()[0:5], existing_tweets_tfidf_matrix), 4)
        #tweets_cosine_sim = np.round(cosine_similarity(existing_tweets_tfidf_matrix, transformed_incoming_tweet), 3)
        # keep track of similarity computation:
        similarity_scores = [] 
        doc_len = existing_tweets_tfidf_matrix.shape[0]
        outer_iterations=0
        mapper = defaultdict(list)
        while doc_len !=0:
        	#for single_tweet_tfidf, bar in zip(existing_tweets_tfidf_matrix.toarray(), tqdm.tqdm(range(doc_len))):
            #for single_tweet_tfidf in existing_tweets_tfidf_matrix.toarray():
            for single_tweet_tfidf, bar in zip(existing_tweets_tfidf_matrix.toarray(), tqdm.tqdm(range(doc_len))):
                tweets_sim = np.round(cosine_similarity(single_tweet_tfidf.reshape(1,-1), existing_tweets_tfidf_matrix.toarray()).flatten(), 3)
                indices = tweets_sim.argsort()[::-1]
                # mapping individiual tweet to the rest of tweets:
                for index, score in zip(indices, tweets_sim):
                    mapper[index].append(score)
                similarity_scores.append(tweets_sim.flatten()[0]) 
                #print(tweets_sim)
                time.sleep(0.0002)
            doc_len -=1
            outer_iterations +=1
        sorted_scores = sorted(similarity_scores, reverse = True)
        # trim scores by removing perfect (1) or unrelated (0) or duplicate scores:
        trimmed_mapper = []
        #for score in mapper[3]:
        for score in zip(mapper[3],mapper[5],mapper[7],mapper[9],mapper[11]):
        	#if score ==1.0 or score == 0.0 or score in trimmed_mapper:
            if score ==1.0 or score == 0.0 or score in trimmed_mapper:
                pass#continue
            else:
                trimmed_mapper.append(score)
       
        x = [trimmed_mapper.index(i) for i in trimmed_mapper]
        y = trimmed_mapper
        #print(trimmed_mapper,x,y, sep='\n')
        plt.xlim(0,len(x))
        plt.xticks(range(len(x)))
        plt.xlabel('Indexed Tweets')
        plt.ylabel('Similarity Score')
        plt.plot(x,y, label='T3')
        plt.legend(loc='best')
        plt.show()

    #############	

# a function pick random tweets and compute matching score in order to discard duplicates ... matching score > 50 are likely to be same/duplicate ...
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


# instantiate ..... !
if __name__=='__main__':
	main()