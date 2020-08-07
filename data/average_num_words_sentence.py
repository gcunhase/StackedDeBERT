import argparse
import os
import csv
from utils import ensure_dir, get_project_path
from collections import defaultdict
from nltk.tokenize import word_tokenize
import numpy as np


root_dir = './twitter_sentiment_data/sentiment140/'
#root_dir = './intent_data/stterror_data/chatbot/gtts_witai/'
tsv_file = open(root_dir + 'test.tsv', 'r')
reader = csv.reader(tsv_file, delimiter='\t')

row_count = 0
data_dict = defaultdict(lambda: [])
for row in reader:
    if row_count != 0:  # header
        label = row[1]
        data_dict[label].append(row[0])

    row_count += 1

for k, v in data_dict.items():
    num_words_arr = []
    for v_i in v:
        words_arr = word_tokenize(v_i)
        num_words_arr = len(words_arr)
        # print(num_words_arr, words_arr)

    print(k, np.mean(num_words_arr))
