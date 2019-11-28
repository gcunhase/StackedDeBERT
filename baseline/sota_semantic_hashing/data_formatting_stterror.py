import csv
from baseline.base_utils import INTENTION_TAGS_WITH_SPACE

''' Convert .tsv to .csv without header for each intent
Format:
example1 intent1
example2 intent1
...
exampleN intentM
'''

dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in ["gtts_witai", "macsay_witai"]:
    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS_WITH_SPACE[dataset_fullname]

        for type in ['test', 'train']:
            # Data dir path
            data_dir_path = "../../data/stterror_data/"
            data_dir_path += "{}/{}/{}.tsv".format(dataset.lower(), tts_stt, type)

            tsv_file = open(data_dir_path)
            reader = csv.reader(tsv_file, delimiter='\t')

            # Write csv
            results_dir_path = data_dir_path.split('.tsv')[0] + "_semantic_hashing.csv"
            file_test = open(results_dir_path, 'wt')
            dict_writer = csv.writer(file_test, delimiter='\t')

            row_count = 0
            sentences, intents = [], []
            for row in reader:
                if row_count != 0:
                    dict_writer.writerow([row[0], tags[row[1]]])
                row_count += 1
