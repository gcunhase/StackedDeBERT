import json
import csv
from baseline.base_utils import INTENTION_TAGS
from utils import get_project_path
import os


# Complete or incomplete data
complete_data = False
dataset_arr = ['chatbot', 'snips']
dataset_fullname_arr = ['ChatbotCorpus', 'snips']

for tts_stt in ["gtts_witai", "macsay_witai"]:
    for dataset_name, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        data_dir_path = "../../data/stterror_data/{}/{}/train.tsv".\
            format(dataset_name.lower(), tts_stt)

        results_dir_path = data_dir_path.split('.tsv')[0] + "_luis.json"

        app_name = "SttErrorIntentClassification-{}-{}".format(dataset_name, tts_stt)
        app_description = "STT Error Intent Recognition App"

        # Read csv
        tsv_file = open(data_dir_path)
        reader = csv.reader(tsv_file, delimiter='\t')
        row_count = 0
        utterances = []
        for row in reader:
            if row_count != 0:
                utterances.append({"text": row[0], "intent": tags[row[1]], "entities": []})
            row_count += 1

        # intents
        intents = []
        for k, v in tags.items():
            intents.append({"name": v})

        data_dict = {
            "luis_schema_version": "3.2.0",
            "versionId": "0.1",
            "name": app_name,
            "desc": app_description,
            "culture": "en-us",
            "tokenizerVersion": "1.0.0",
            "intents": intents,
            "entities": [],
            "composites": [],
            "closedLists": [],
            "patternAnyEntities": [],
            "regex_entities": [],
            "prebuiltEntities": [],
            "model_features": [],
            "regex_features": [],
            "patterns": [],
            "utterances": utterances,
            "settings": []
        }

        # Write in json
        with open(results_dir_path, 'w') as outfile:
            json.dump(data_dict, outfile, indent=2, ensure_ascii=False)
