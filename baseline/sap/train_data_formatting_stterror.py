import json
import csv
import os
from baseline.base_utils import INTENTION_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
expression;language
example1;en
example2;en
example3;en
'''

tts_stt_arr = ["gtts_witai", "macsay_witai"]
dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in tts_stt_arr:
    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        # Data dir path
        data_dir_path = "../../data/stterror_data/{}/{}/train.tsv".format(dataset, tts_stt)

        # Read tsv
        tsv_file = open(data_dir_path)
        reader = csv.reader(tsv_file, delimiter='\t')

        dict_intents = defaultdict(lambda: [])
        row_count = 0
        sentences = []
        for row in reader:
            if row_count != 0:
                dict_intents[tags[row[1]]].append(row[0])
            row_count += 1

        for k, v in dict_intents.items():
            # Write csv
            results_dir_path = data_dir_path.split('.tsv')[0] + "_sap_{}.csv".format(k)
            file_test = open(results_dir_path, 'wt')
            dict_writer = csv.writer(file_test, delimiter=';')
            dict_writer.writerow(['expression', 'language'])

            for i, text in enumerate(dict_intents[k]):
                if text.strip() == "":
                    text = 'w'
                dict_writer.writerow([text, 'en'])
