#!/usr/bin/env python3

# PACKAGES IMPORT
import numpy as np
import pandas as pd
import random
import os
from difflib import SequenceMatcher as sm

# THE FILE PATH
file_path ='/home/ijdutse/thesis_p1/short_tweets.txt'


# THE MAIN FUNCTION
def main():
    with open(file_path,'r') as f:
        text = f.readlines()
        get_random_tweets_sim(text)
        # return round(sim.ratio() * 100, 2)

# FUNCTION TO PICK RANDOM TWEETS AND COMPUTE THE MATCHING SCORE ... MATCHING SCORE > 50 ARE LIKELY TO BE SAME/DUPLICATE TWEETS, HENCE DISCARDED
def get_random_tweets_sim(tweets):
    random.shuffle(tweets)
    x = tweets[0]
    y = tweets[1]
    sim = sm(None,x, y)
    print (round(sim.ratio()*100,2), sim.ratio())
    #return round(sim.ratio()*100,2), sim.ratio()

# INSTANTITATE THE MAIN FUNCTION
if __name__=='__main__':
	main()
