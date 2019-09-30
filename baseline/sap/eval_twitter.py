import json
import csv
import os
import sapcai
from base_utils import get_label, SENTIMENT_TAGS, LABELS_ARRAY_INT

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
data_type_arr = ["inc_with_corr"]  # ["corr", "inc", "inc_with_corr"]

results_dir_tmp = './results/'
ensure_dir(results_dir_tmp)

paramsfilename = "sap_params_example_twitter.json"
with open(paramsfilename) as parmFile:
    params = json.load(parmFile)

# ========= Eval ==========
dataset_arr = ['sentiment140']

for dataset in dataset_arr:
    print("Evaluating {} dataset".format(dataset))
    tags = SENTIMENT_TAGS[dataset]

    for data_type in data_type_arr:
	    results_dir = '{}twitter_{}_{}/'.format(results_dir_tmp, dataset, data_type)
	    ensure_dir(results_dir)
	    data_name = dataset + "_" + data_type
	    params_data = params[data_name]
	    REQUEST_TOKEN = params_data["REQUEST_TOKEN"]
	    DEVELOPER_TOKEN = params_data["DEVELOPER_TOKEN"]

	    # ===== Authenticate ======
	    request = sapcai.Request(REQUEST_TOKEN, 'en')
	    data_dir_path = "../../data/twitter_sentiment_data/sentiment140"
	    if data_type == "corr":
	        data_dir_path += "_corrected_sentences/"
	    elif data_type == "inc":
	        data_dir_path += "/"
	    else:
	        data_dir_path += "_inc_with_corr_sentences/"
	    data_dir_path += "test_sap.csv"

	    # Read .csv file
	    tsv_file = open(data_dir_path)
	    reader = csv.reader(tsv_file, delimiter=',')

	    row_count = 0
	    target_intents, pred_intents = [], []
	    for row in reader:
	        if row_count != 0:
	            response = request.analyse_text(row[0])
	            target_intents.append(int(row[1]))
	            if response.intent is None:
	                pred_intents.append(get_label(dataset, "None", dict_type="sentiment"))
	            else:
	                pred_intents.append(get_label(dataset, response.intent.slug, dict_type="sentiment"))
	        row_count += 1

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
