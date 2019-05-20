import os
import numpy as np
import csv

__author__ = "Gwena Cunha"


INTENTION_TAGS = {
    'snips': {'AddToPlaylist': 0,
              'BookRestaurant': 1,
              'GetWeather': 2,
              'PlayMusic': 3,
              'RateBook': 4,
              'SearchCreativeWork': 5,
              'SearchScreeningEvent': 6},
    'ChatbotCorpus': {'DepartureTime': 0,
                      'FindConnection': 1},
    'AskUbuntuCorpus': {'Make Update': 0,
                        'Setup Printer': 1,
                        'Shutdown Computer': 2,
                        'Software Recommendation': 3,
                        'None': 4},
    'WebApplicationsCorpus': {'Change Password': 0,
                              'Delete Account': 1,
                              'Download Video': 2,
                              'Export Data': 3,
                              'Filter Spam': 4,
                              'Find Alternative': 5,
                              'Sync Accounts': 6,
                              'None': 7}
}

MISS_TAG = '_'  # <miss> -> tokenizer recognizes as ['<', 'miss', '>]


def get_project_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + "/" + filename, 'wt')
    dict_writer = csv.writer(file_test, delimiter='\t')
    dict_writer.writerow(keys)
    r = zip(*dict.values())
    for d in r:
        dict_writer.writerow(d)
