import numpy as np
import pandas as pd
import random
from difflib import SequenceMatcher as sm

# Specify the file path to use:
file_path =  'home/ijdutse/phase2/separation_degrees/short_tweets.csv'

eets.csv')

def main():
	list_random()
	


def list_random(ran):
	with open('home/ijdutse/phase2/separation_degrees/short_tweets.csv', 'r') as f:
		tweet = f.readlines()
    random.shuffle(ran)
    x = ran[0]
    y = ran[1]
    sim = sm(None,x, y)
    return round(sim.ratio()*100, 2), x, y

if __name__=='__main__':
	main()
