import json
import csv
import os
import sapcai
from base_utils import get_label, INTENTION_TAGS, LABELS_ARRAY_INT

# Eval
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

'''
resp = response.raw
pred_intent = response.intent.slug

Source: https://github.com/SAPConversationalAI/SDK-python
'''

def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

# ======= Params =========
complete = False
perc = 0.8

results_dir = './results/notag/'
ensure_dir(results_dir)

if complete:
    results_dir += 'complete/'
else:
    results_dir += 'comp_inc_{}/'.format(perc)
ensure_dir(results_dir)

if complete:
    paramsfilename = "sap_params.json"
else:
    paramsfilename = "comp_inc_sap_params_noTag_{}.json".format(perc)

with open(paramsfilename) as parmFile:
    params = json.load(parmFile)

# ========= Eval ==========
dataset_arr = ['snips']

for dataset in dataset_arr:
    print("Evaluating {} dataset".format(dataset))
    tags = INTENTION_TAGS[dataset]
    params_data = params[dataset]
    REQUEST_TOKEN = params_data["REQUEST_TOKEN"]
    DEVELOPER_TOKEN = params_data["DEVELOPER_TOKEN"]

    # ===== Authenticate ======
    # client = sapcai.Client(DEVELOPER_TOKEN)
    # reply = client.request.analyse_text("Hi")
    # print(reply.raw)
    request = sapcai.Request(REQUEST_TOKEN, 'en')

    data_dir_path = "../../data/snips_intent_data/"
    if complete:
        data_dir_path += "complete_data/"
    else:
        data_dir_path += "comp_with_incomplete_data_tfidf_lower_{}_noMissingTag/".format(perc)
    data_dir_path += "test_sap.csv"

    # Read .csv file
    tsv_file = open(data_dir_path)
    reader = csv.reader(tsv_file, delimiter=',')

    row_count = 0
    target_intents, pred_intents = [], []
    for row in reader:
        if row_count != 0:
            # print("{}: {}".format(row_count, row[0]))
            response = request.analyse_text(row[0])
            target_intents.append(int(row[1]))
            if response.intent is None:
                pred_intents.append(get_label(dataset, "None"))
            else:
                pred_intents.append(get_label(dataset, response.intent.slug))
        row_count += 1

    # print(target_intents)
    # print(pred_intents)

    # Calculate: precision, recall and F1
    labels = LABELS_ARRAY_INT[dataset.lower()]
    result = {}
    result['precision_macro'], result['recall_macro'], result['f1_macro'], support = \
        precision_recall_fscore_support(target_intents, pred_intents, average='macro', labels=labels)
    result['precision_micro'], result['recall_micro'], result['f1_micro'], support = \
        precision_recall_fscore_support(target_intents, pred_intents, average='micro', labels=labels)
    result['precision_weighted'], result['recall_weighted'], result['f1_weighted'], support = \
        precision_recall_fscore_support(target_intents, pred_intents, average='weighted', labels=labels)
    result['confusion_matrix'] = confusion_matrix(target_intents, pred_intents, labels=labels).tolist()

    output_eval_filename = "eval_results_" + dataset.lower()

    output_eval_file = os.path.join(results_dir, output_eval_filename + ".json")
    with open(output_eval_file, "w") as writer:
        json.dump(result, writer, indent=2)
