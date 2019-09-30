import csv
from baseline.base_utils import SENTIMENT_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
example1
example2
...
exampleN
'''

# Data type: inc, corr, inc_with_corr
data_type = "corr"
dataset_arr = ['sentiment140']

for dataset in dataset_arr:
    tags = SENTIMENT_TAGS[dataset]

    for type in ['train', 'test']:

        data_dir_path = "../../data/twitter_data/sentiment140"
        if data_type == "corr":
            data_dir_path += "_corrected_sentences/"
            # app_description = "Sentiment Recognition App for Corrected Sentences"
        elif data_type == "inc":
            data_dir_path += "/"
            # app_description = "Sentiment Recognition App for Original, Incorrect Sentences"
        else:
            data_dir_path += "_inc_with_corr_sentences/"
            # app_description = "Sentiment Recognition App for Original and Corrected Sentences"
        data_dir_path += type + '.tsv'

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
            results_dir_path = data_dir_path.split('.tsv')[0] + "_dialogflow_{}.csv".format(k)
            file_test = open(results_dir_path, 'wt')
            dict_writer = csv.writer(file_test, delimiter=';')

            for text in dict_intents[k]:
                dict_writer.writerow([text])
