import argparse
import os
import csv
from utils import ensure_dir, get_project_path
from collections import defaultdict


# POS-tag for irrelevant tag selection
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

__author__ = "Gwena Cunha"


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + "/" + filename, 'w')
    dict_writer = csv.DictWriter(file_test, fieldnames=keys, delimiter='\t')
    dict_writer.writeheader()
    for data in dict:
        dict_writer.writerow(data)


def make_dataset(root_data_dir, complete_data_dir, incomplete_data_dir, results_dir):
    """
    :param root_data_dir: directory to save data
    :param complete_data_dir: subdirectory with complete data
    :param incomplete_data_dir: subdirectory with incomplete data
    :param results_dir: subdirectory with incomplete data
    :return:
    """
    print("Making incomplete intention classification dataset...")
    complete_data_dir_path = root_data_dir + '/' + complete_data_dir
    incomplete_data_dir_path = root_data_dir + '/' + incomplete_data_dir

    results_dir_path = root_data_dir + '/' + results_dir
    ensure_dir(results_dir_path)

    # Traverse all sub-directories
    files_dictionary = defaultdict(lambda: [])
    for sub_dir in os.walk(complete_data_dir_path):
        if len(sub_dir[1]) == 0:
            data_name = sub_dir[0].split('/')[-1]
            files_dictionary[data_name] = sub_dir[2]

    # Open train and test tsv files
    for k, v in files_dictionary.items():
        save_path = results_dir_path + '/' + k
        ensure_dir(save_path)
        keys = ['sentence', 'label', 'missing', 'target']
        for comp_v_i, inc_v_i in zip(['test.tsv', 'train.tsv'], ['test_withMissingWords.tsv', 'train_withMissingWords.tsv']):
            complete_tsv_file = open(complete_data_dir_path + '/' + k + '/' + comp_v_i, 'r')
            incomplete_tsv_file = open(incomplete_data_dir_path + '/' + k + '/' + inc_v_i, 'r')
            reader_complete = csv.reader(complete_tsv_file, delimiter='\t')
            reader_incomplete = csv.reader(incomplete_tsv_file, delimiter='\t')

            row_count = 0
            data_dict = []
            for row_comp, row_inc in zip(reader_complete, reader_incomplete):
                if row_count != 0:
                    # Incomplete
                    data_dict.append({keys[0]: row_inc[0], keys[1]: row_inc[1], keys[2]: row_inc[2], keys[3]: row_comp[0]})

                row_count += 1

            # Save train, test, val in files in the format (sentence, label)
            write_tsv(save_path, comp_v_i, keys, data_dict)

    print("Incomplete intention classification dataset with complete target completed")


def init_args():
    parser = argparse.ArgumentParser(description="Script to make intention recognition dataset")
    parser.add_argument('--root_data_dir', type=str, default=get_project_path() + "/data",
                        help='Directory to save subdirectories, needs to be an absolute path')
    parser.add_argument('--complete_data_dir', type=str, default="complete_data",
                        help='Subdirectory with complete data')
    parser.add_argument('--incomplete_data_dir', type=str, default="incomplete_data_tfidf_lower_0.8_noMissingTag",
                        help='Subdirectory with incomplete data')
    parser.add_argument('--results_dir', type=str, default="incomplete_data_tfidf_lower_with_target_0.8_noMissingTag",
                        help='Subdirectory to save Joint Complete and Incomplete data')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    make_dataset(args.root_data_dir, args.complete_data_dir, args.incomplete_data_dir, args.results_dir)
