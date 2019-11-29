import json
import csv
import os
import sapcai
from baseline.base_utils import get_label, INTENTION_TAGS, LABELS_ARRAY_INT

# Eval
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

'''
resp = response.raw
pred_intent = response.intent.slug

Source: https://github.com/SAPConversationalAI/SDK-python
'''


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ======= Params =========
tts_stt = "macsay_witai"

results_dir = './results/stterror/inc_{}/'.format(tts_stt)
ensure_dir(results_dir)

paramsfilename = "inc_stterror_sap_params_{}.json".format(tts_stt)

with open(paramsfilename) as parmFile:
    params = json.load(parmFile)

# ========= Eval ==========
dataset_arr = ['snips']
dataset_fullname_arr = ['snips']

for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
    print("Evaluating {} dataset with {}".format(dataset, tts_stt))
    tags = INTENTION_TAGS[dataset_fullname]
    params_data = params[dataset]
    REQUEST_TOKEN = params_data["REQUEST_TOKEN"]
    DEVELOPER_TOKEN = params_data["DEVELOPER_TOKEN"]

    # ===== Authenticate ======
    # client = sapcai.Client(DEVELOPER_TOKEN)
    # reply = client.request.analyse_text("Hi")
    # print(reply.raw)
    request = sapcai.Request(REQUEST_TOKEN, 'en')

    data_dir_path = "../../data/stterror_data/{}/{}/test_sap.csv".format(dataset, tts_stt)

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
                pred_intents.append(get_label(dataset_fullname, "None"))
            else:
                pred_intents.append(get_label(dataset_fullname, response.intent.slug))
        row_count += 1

    # print(target_intents)
    # print(pred_intents)

    # Calculate: precision, recall and F1
    labels = LABELS_ARRAY_INT[dataset_fullname.lower()]
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
