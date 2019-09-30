import csv
from baseline.base_utils import SENTIMENT_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
expression;language
example1;en
example2;en
example3;en
'''

data_type = "inc_with_corr"
dataset_arr = ['sentiment140']

for dataset in dataset_arr:
    tags = SENTIMENT_TAGS[dataset]

    # Data dir path
    data_dir_path = "../../data/twitter_sentiment_data/sentiment140"
    if data_type == "corr":
        data_dir_path += "_corrected_sentences/"
    elif data_type == "inc":
        data_dir_path += "/"
    else:
        data_dir_path += "_inc_with_corr_sentences/"
    data_dir_path += 'train.tsv'

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

        for text in dict_intents[k]:
            dict_writer.writerow([text, 'en'])
