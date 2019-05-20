import csv
from baseline.base_utils import INTENTION_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
expression;language
example1;en
example2;en
example3;en
'''

complete = False
perc = 0.8
dataset_arr = ['ChatbotCorpus', 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']

for dataset in dataset_arr:
    tags = INTENTION_TAGS[dataset]

    # Data dir path
    data_dir_path = "/mnt/gwena/Gwena/"
    if complete:
        data_dir_path += "IntentionClassifier/data/processed/"
        if 'snips' in dataset:
            data_dir_path += "{}/train.tsv".format(dataset.lower())
        else:
            data_dir_path += "nlu_eval/{}/train.tsv".format(dataset.lower())
    else:
        data_dir_path += "IncompleteIntentionClassifier/data/incomplete_data_tfidf_lower_{}/".format(perc)
        if 'snips' in dataset:
            data_dir_path += "{}/train.tsv".format(dataset.lower())
        else:
            data_dir_path += "nlu_eval_{}/train.tsv".format(dataset.lower())

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
