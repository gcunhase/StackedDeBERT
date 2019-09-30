import csv
from baseline.base_utils import SENTIMENT_TAGS

''' Convert .tsv to .csv without header for each intent
Format:
example1;intent1
example2;intent1
...
exampleN;intentM
'''

data_type = "inc_with_corr"  # inc, inc_with_corr
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
    data_dir_path += 'test.tsv'

    # Read tsv file
    tsv_file = open(data_dir_path)
    reader = csv.reader(tsv_file, delimiter='\t')

    # Write csv
    results_dir_path = data_dir_path.split('.tsv')[0] + "_sap.csv"
    file_test = open(results_dir_path, 'wt')
    dict_writer = csv.writer(file_test, delimiter=',')

    row_count = 0
    sentences, intents = [], []
    for row in reader:
        if row_count != 0:
            dict_writer.writerow([row[0], row[1]])
        row_count += 1
