import csv
from baseline.base_utils import INTENTION_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
example1
example2
...
exampleN
'''

complete = False
tts_stt_arr = ["gtts_witai", "macsay_witai"]
dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in tts_stt_arr:
    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        for type in ['train', 'test']:
            # Data dir path
            data_dir_path = "../../data/stterror_data/"
            data_dir_path += "{}/{}/{}.tsv".format(dataset.lower(), tts_stt, type)

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
