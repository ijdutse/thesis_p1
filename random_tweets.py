#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import os
from difflib import SequenceMatcher as sm

# Specify the file path to use:
file_path ='/home/ijdutse/phase2/separation_degrees/text.txt'



def main():
    with open(file_path,'r') as f:
        text = f.readlines()
        get_random_tweets_sim(text)
        #print ('ALL DONE!')
        #random.shuffle(tweets)
        #x = tweets[0]
        #y = tweets[1]
        #sim = sm(None, x, y)
        #print('IM YET TO BE EXCEUTED!')
        #print(round(sim.ratio() * 100, 2))
        # return round(sim.ratio() * 100, 2)


def get_random_tweets_sim(tweets):
    random.shuffle(tweets)
    x = tweets[0]
    y = tweets[1]
    sim = sm(None,x, y)
    #print('IM YET TO BE EXCEUTED!')
    print (round(sim.ratio()*100,2), sim.ratio())
    #return round(sim.ratio()*100,2), sim.ratio()


if __name__=='__main__':
    #with open(file_path,'r') as f:
     #   text = f.readlines()
      #  get_random_tweets_sim(text)
       # print (len(text))

	main()
