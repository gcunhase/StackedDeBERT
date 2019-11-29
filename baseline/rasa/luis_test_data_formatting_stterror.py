import json
import csv
from baseline.base_utils import INTENTION_TAGS
from utils import get_project_path
import os

''' Convert .tsv to .json
Format:
[
    {
        "text": "when is the next bus from garching forschungzentrum",
        "intent": "Departure Time",
        "entities": []
    },
    ...
]
'''

# Complete or incomplete data
complete_data = False
dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in ["gtts_witai", "macsay_witai"]:
    for dataset_name, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        data_dir_path = "../../data/stterror_data/{}/{}/test.tsv".\
            format(dataset_name.lower(), tts_stt)

        results_dir_path = data_dir_path.split('.tsv')[0] + "_luis.json"

        # Read csv
        tsv_file = open(data_dir_path)
        reader = csv.reader(tsv_file, delimiter='\t')
        row_count = 0
        data_dict = []
        for row in reader:
            if row_count != 0:
                data_dict.append({"text": row[0], "intent": tags[row[1]], "entities": []})
            row_count += 1

        # Write in json
        with open(results_dir_path, 'w') as outfile:
            json.dump(data_dict, outfile, indent=2, ensure_ascii=False)
