import csv
from baseline.base_utils import INTENTION_TAGS

''' Convert .tsv to .csv without header for each intent
Format:
example1;intent1
example2;intent1
...
exampleN;intentM
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
            data_dir_path += "{}/test.tsv".format(dataset.lower())
        else:
            data_dir_path += "nlu_eval/{}/test.tsv".format(dataset.lower())
    else:
        data_dir_path += "IncompleteIntentionClassifier/data/incomplete_data_tfidf_lower_{}/".format(perc)
        if 'snips' in dataset:
            data_dir_path += "{}/test.tsv".format(dataset.lower())
        else:
            data_dir_path += "nlu_eval_{}/test.tsv".format(dataset.lower())

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
