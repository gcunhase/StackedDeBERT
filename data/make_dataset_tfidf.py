import argparse
import requests
import os
import glob
import csv
from utils import ensure_dir, get_project_path, MISS_TAG
from collections import defaultdict, OrderedDict
import operator

from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = "Gwena Cunha"


# percentage of words missing in sentence
PERC_MISSING = 0.8


def delete_tags(sent, dictionary, use_lower=False):
    missing_words = ''
    incomplete_sentence = ''
    sent_lower = sent.lower()
    tokens_lower = sent_lower.split()
    tokens = sent.split()

    num_tokens = int(len(tokens)*PERC_MISSING)

    t = 0
    missing = []
    for dk, dv in dictionary.items():
        if t == num_tokens:
            break
        if use_lower:
            if dk in tokens_lower:
                missing.append(dk)
                t += 1
        else:
            if dk in tokens:
                missing.append(dk)
                t += 1

    for t_lower, tok in zip(tokens_lower, tokens):
        if use_lower:
            if t_lower in missing:
                missing_words += tok + ' '
                incomplete_sentence += MISS_TAG + ' '
            else:
                incomplete_sentence += tok + ' '
        else:
            if tok in missing:
                missing_words += tok + ' '
                incomplete_sentence += MISS_TAG + ' '
            else:
                incomplete_sentence += tok + ' '

    return missing_words, incomplete_sentence


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + "/" + filename, 'wt')
    dict_writer = csv.writer(file_test, delimiter='\t')
    dict_writer.writerow(keys)
    r = zip(*dict.values())
    for d in r:
        dict_writer.writerow(d)


def obtain_dictionary(files_dictionary, data_dir_path):
    dictionary_all = defaultdict(lambda: [])
    for k, v in files_dictionary.items():
        S = []
        for v_i in v:
            tsv_file = open(data_dir_path + '/' + k + '/' + v_i, 'r')
            reader = csv.reader(tsv_file, delimiter='\t')
            row_count = 0
            for row in reader:
                if row_count != 0:
                    S.append(row[0])
                row_count += 1

        vectorizer = TfidfVectorizer()
        response = vectorizer.fit_transform(S)
        feature_names = vectorizer.get_feature_names()
        dictionary = defaultdict(lambda: [])
        for col in response.nonzero()[1]:
            dictionary[feature_names[col]] = response[0, col]
        dictionary_sorted = OrderedDict(sorted(dictionary.items(), key=operator.itemgetter(1)))  # , reverse=True))
        dictionary_all[k] = dictionary_sorted
    return dictionary_all


def make_dataset(root_data_dir, data_dir, results_dir):
    """
    :param root_data_dir: directory to save data
    :param data_dir: subdirectory with complete data
    :param results_dir: subdirectory with incomplete data
    :return:
    """
    print("Making incomplete intention classification dataset...")
    data_dir_path = root_data_dir + '/' + data_dir

    results_dir_path = root_data_dir + '/' + results_dir
    ensure_dir(results_dir_path)

    # Traverse all sub-directories
    files_dictionary = defaultdict(lambda: [])
    for sub_dir in os.walk(data_dir_path):
        if len(sub_dir[1]) == 0:
            data_name = sub_dir[0].split('/')[-1]
            files_dictionary[data_name] = sub_dir[2]

    dictionary_all = obtain_dictionary(files_dictionary, data_dir_path)
    # Open train and test tsv files
    for k, v in files_dictionary.items():
        save_path = results_dir_path + '/' + k
        ensure_dir(save_path)
        for v_i in v:
            tsv_file = open(data_dir_path + '/' + k + '/' + v_i, 'r')
            reader = csv.reader(tsv_file, delimiter='\t')
            sentences, labels = [], []
            missing_words_arr = []
            row_count = 0
            for row in reader:
                if row_count != 0:
                    missing_words, incomplete_sentence = delete_tags(row[0], dictionary_all[k], use_lower=True)
                    sentences.append(incomplete_sentence)
                    labels.append(row[1])
                    missing_words_arr.append(missing_words)
                row_count += 1

            # Make dictionaries to save in .tsv
            data_dict = {'sentence': sentences, 'label': labels}
            missing_dict = {'sentence': sentences, 'label': labels, 'missing': missing_words_arr}
            words, score = [], []
            for wk, wv in dictionary_all[k].items():
                words.append(wk)
                score.append(wv)
            words_dictionary = {'word': words, 'score': score}

            # Save train, test, val in files in the format (sentence, label)
            keys = ['sentence', 'label']
            write_tsv(save_path, v_i, keys, data_dict)

            keys = ['sentence', 'label', 'missing']
            write_tsv(save_path, v_i.split('.tsv')[0]+'_withMissingWords.tsv', keys, missing_dict)

            keys = ['word', 'score']
            write_tsv(save_path, 'dictionary.tsv', keys, words_dictionary)

    print("Incomplete intention classification dataset completed")


def init_args():
    parser = argparse.ArgumentParser(description="Script to make intention recognition dataset")
    parser.add_argument('--root_data_dir', type=str, default=get_project_path() + "/data",
                        help='Directory to save subdirectories, needs to be an absolute path')
    parser.add_argument('--data_dir', type=str, default="complete_data",
                        help='Subdirectory to save processed Intention Classification data')
    parser.add_argument('--results_dir', type=str, default="incomplete_data_tfidf_lower_{}".format(PERC_MISSING),
                        help='Subdirectory to save processed Incomplete Intention Classification data')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    make_dataset(args.root_data_dir, args.data_dir, args.results_dir)
