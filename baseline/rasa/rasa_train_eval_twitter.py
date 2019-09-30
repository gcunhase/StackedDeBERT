import json
from base_utils_sota import SENTIMENT_TAGS, LABELS_ARRAY_INT, LABELS_ARRAY, get_label
import os
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter
import numpy as np

# Eval
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from plot_confusion_matrix_sota import plot_confusion_matrix

'''
Dataset: LUIS v2.x.x

spacy download en

pip install rasa rasa_nlu sklearn_crfsuite tensorflow
Reference: https://rasa.com/docs/nlu/python/
'''


def get_project_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def write_json(results_dir_path, data_dict):
    with open(results_dir_path, 'w') as outfile:
        json.dump(data_dict, outfile, indent=2, ensure_ascii=False)


def read_json(data_dir_path):
    with open(data_dir_path, 'r') as f:
        datastore = json.load(f)
        return datastore


# ======= Params =========
data_type = "inc_with_corr"  # "corr", "inc", "inc_with_corr"
do_train = True
do_eval = True
spacy = True
save_dir = "rasa_saved_model_sentiment140"

# prefix_comp = data_type
if spacy:
   prefix_spacy = '_spacy'
else:
   prefix_spacy = '_tf'

for dataset_name in ['Sentiment140']:  # ['Sentiment140']:

    print(dataset_name)
    tags = SENTIMENT_TAGS[dataset_name]
    root_dir = "../../data/twitter_sentiment_data/"  # get_project_path() + '/data/'
    config_dir = "/mnt/gwena/Gwena/IncompleteIntentionClassifier/baseline/rasa/"  # get_project_path() + '/baseline/rasa/'

    data_dir_path = root_dir + "sentiment140"
    if data_type == "corr":
        data_dir_path += "_corrected_sentences/"
        app_name = "SentimentClassification-{}-corr".format(dataset_name)
        app_description = "Sentiment Recognition App for Corrected Sentences"
    elif data_type == "inc":
        data_dir_path += "/"
        app_name = "SentimentClassification-{}-inc".format(dataset_name)
        app_description = "Sentiment Recognition App for Original, Incorrect Sentences"
    else:  # data_type == "inc_with_corr":
        data_dir_path += "_inc_with_corr_sentences/"
        app_name = "SentimentClassification-{}-inc-corr".format(dataset_name)
        app_description = "Sentiment Recognition App for Original and Corrected Sentences"
    data_dir_path_test = data_dir_path + 'test_luis.json'
    data_dir_path += 'train_luis.json'

    # ======= Dataset =========
    # Load LUIS json data
    datastore = read_json(data_dir_path)

    # Change "luis_schema_version" to "2.2.0"
    datastore["luis_schema_version"] = "2.2.0"

    # Save in train_luis_v2.json
    data_dir_path = data_dir_path.split('.json')[0] + '_v2.json'
    write_json(data_dir_path, datastore)

    # project_name = "{}{}".format(prefix_comp, prefix_spacy)
    project_name = ""

    # ======= Train =========
    # Training RASA model
    if do_train:
        training_data = load_data(data_dir_path)
        trainer = Trainer(config.load(config_dir + "rasa_nlu_config{}.yml".format(prefix_spacy)))
        trainer.train(training_data)

        model_dir = config_dir + save_dir + '/'
        ensure_dir(model_dir)
        fixed_model_name = dataset_name.lower()+"_"+data_type+prefix_spacy
        model_directory = trainer.persist(model_dir, project_name=project_name, fixed_model_name=fixed_model_name)  # Returns the directory the model is stored in
        print(model_directory)

    # ======= Test =========
    if do_eval:
        # Load trained model
        if len(project_name) == 0:
            model_directory = config_dir + "{}/{}".format(save_dir, fixed_model_name)
        else:
            model_directory = config_dir + "{}/{}/{}".format(save_dir, project_name, fixed_model_name)
        interpreter = Interpreter.load(model_directory)

        # Get test samples
        datastore_test = read_json(data_dir_path_test)
        predicted, target = [], []
        for d in datastore_test:
            target_name = d["intent"]  # string
            # print(target_name)
            target_label = get_label(dataset_name, target_name, dict_type="sentiment")  # int
            target.append(target_label)

            prediction = interpreter.parse(d["text"])  # dict
            # print(prediction)
            predicted_intent = str(prediction['intent']['name'])  # string
            predicted_label = get_label(dataset_name, predicted_intent, dict_type="sentiment")  # int
            predicted.append(predicted_label)

        # Calculate: precision, recall and F1
        labels = LABELS_ARRAY_INT[dataset_name.lower()]
        result = {}
        result['precision_macro'], result['recall_macro'], result['f1_macro'], support =\
            precision_recall_fscore_support(target, predicted, average='macro', labels=labels)
        result['precision_micro'], result['recall_micro'], result['f1_micro'], support =\
            precision_recall_fscore_support(target, predicted, average='micro', labels=labels)
        result['precision_weighted'], result['recall_weighted'], result['f1_weighted'], support =\
            precision_recall_fscore_support(target, predicted, average='weighted', labels=labels)
        result['confusion_matrix'] = confusion_matrix(target, predicted, labels=labels).tolist()

        output_eval_filename = "eval_results"

        target_asarray = np.asarray(target)
        predicted_asarray = np.asarray(predicted)
        classes = np.asarray(LABELS_ARRAY[dataset_name.lower()])
        classes_idx = None

        ax, fig = plot_confusion_matrix(target_asarray, predicted_asarray, classes=classes,
                                        normalize=True, title='Normalized confusion matrix', rotate=False,
                                        classes_idx=classes_idx)
        fig.savefig(os.path.join(model_directory, output_eval_filename + "_confusion.png"))

        output_eval_file = os.path.join(model_directory, output_eval_filename + ".json")
        with open(output_eval_file, "w") as writer:
            json.dump(result, writer, indent=2)
