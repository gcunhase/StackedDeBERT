import csv
from baseline.base_utils import INTENTION_TAGS

''' Convert .tsv to .csv without header for each intent
Format:
example1;intent1
example2;intent1
...
exampleN;intentM
'''

tts_stt_arr = ["gtts_witai", "macsay_witai"]
dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in tts_stt_arr:
    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        # Data dir path
        data_dir_path = "../../data/stterror_data/{}/{}/test.tsv".format(dataset, tts_stt)

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
                text = row[0]
                if text.strip() == "":
                    text = 'w'
                dict_writer.writerow([text, row[1]])
            row_count += 1
