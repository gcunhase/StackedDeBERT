import os
import numpy as np
import csv


INTENTION_TAGS = {
    'ChatbotCorpus': {'DepartureTime': 0,
                      'FindConnection': 1},
}

SENTIMENT_TAGS = {'Sentiment140': {'Negative': 0,
                                   'Positive': 1}
                  }

MISS_TAG = ''


def get_project_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + "/" + filename, 'wt')
    dict_writer = csv.writer(file_test, delimiter='\t')
    dict_writer.writerow(keys)
    r = zip(*dict.values())
    for d in r:
        dict_writer.writerow(d)
