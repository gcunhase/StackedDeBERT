
import argparse
from data.make_dataset import write_tsv
from utils import get_project_path, MISS_TAG
import csv
from collections import defaultdict
import os

# OLD_MISS_TAG = '<miss>'
OLD_MISS_TAG = '_'


def change_tag(root_data_dir, data_dir):
    print("Making incomplete intention classification dataset...")
    data_dir_path = root_data_dir + '/' + data_dir

    # Traverse all sub-directories
    files_dictionary = defaultdict(lambda: [])
    for sub_dir in os.walk(data_dir_path):
        if len(sub_dir[1]) == 0:
            data_name = sub_dir[0].split('/')[-1]
            files_dictionary[data_name] = sub_dir[2]

    # Open train and test tsv files
    for k, v in files_dictionary.items():
        save_path = data_dir_path + '/' + k

        for v_i in v:
            tsv_file = open(data_dir_path + '/' + k + '/' + v_i, 'r')
            reader = csv.reader(tsv_file, delimiter='\t')
            sentences, labels, missing = [], [], []
            row_count = 0
            for row in reader:
                if row_count != 0:
                    incomplete_sentence = row[0].replace(OLD_MISS_TAG, MISS_TAG)
                    sentences.append(incomplete_sentence)
                    labels.append(row[1])
                    if '_withMissingWords' in v_i:
                        missing.append(row[2])
                row_count += 1

            # Make dictionaries to save in .tsv
            if '_withMissingWords' in v_i:
                missing_dict = {'sentence': sentences, 'label': labels, 'missing': missing}
                keys = ['sentence', 'label', 'missing']
                write_tsv(save_path, v_i, keys, missing_dict)

            else:
                data_dict = {'sentence': sentences, 'label': labels}
                keys = ['sentence', 'label']
                write_tsv(save_path, v_i, keys, data_dict)


def init_args():
    parser = argparse.ArgumentParser(description="Script to make intention recognition dataset")
    parser.add_argument('--root_data_dir', type=str, default=get_project_path() + "/data",
                        help='Directory to save subdirectories, needs to be an absolute path')
    parser.add_argument('--data_dir', type=str, default="incomplete_data_tfidf_lower_0.8_MASKtag",
                        help='Subdirectory to save processed Intention Classification data')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    change_tag(args.root_data_dir, args.data_dir)
