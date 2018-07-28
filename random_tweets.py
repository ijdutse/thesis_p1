import numpy as np
import pandas as pd
import random

df = pd.read_csv('short_tweets.csv')

print(len(df.Text))
from difflib import SequenceMatcher as sm
def list_random(ran):
    random.shuffle(ran)
    x = ran[0]
    y = ran[1]
    sim = sm(None,x, y)
    return round(sim.ratio()*100, 2), x, y
