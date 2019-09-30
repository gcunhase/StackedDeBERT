import dialogflow
# import dialogflow_v2 as dialogflow
from google.cloud import storage
import argparse
import uuid
from base_utils import SENTIMENT_TAGS
import csv
from sklearn.metrics import precision_recall_fscore_support
import json
import os

# https://dialogflow-python-client-v2.readthedocs.io/en/latest/

GOOGLE_APPLICATION_CREDENTIALS = '[INCLUDE PATH TO AGENT JSON FILE HERE]'


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def delete_intent(project_id, intent_id):
    """Delete intent with the given intent type and intent value."""
    intents_client = dialogflow.IntentsClient.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

    intent_path = intents_client.intent_path(project_id, intent_id)

    intents_client.delete_intent(intent_path)


def detect_intent_texts(project_id, session_id, texts, language_code, do_print=False):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    # session_client = dialogflow.SessionsClient()
    session_client = dialogflow.SessionsClient.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    detected_intent = []
    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        if do_print:
            print('=' * 20)
            print('Query text: {}'.format(response.query_result.query_text))
            print('Detected intent: {} (confidence: {})\n'.format(
                response.query_result.intent.display_name,
                response.query_result.intent_detection_confidence))
            print('Fulfillment text: {}\n'.format(
                response.query_result.fulfillment_text))

        detected_intent.append(response.query_result.intent.display_name)
    return detected_intent


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--project-id',
        default='newagent-4a8cc',
        help='Project/agent id.  Required.')
    parser.add_argument(
        '--session-id',
        help='Identifier of the DetectIntent session. '
        'Defaults to a random UUID.',
        default=str(uuid.uuid4()))
    parser.add_argument(
        '--language-code',
        help='Language code of the query. Defaults to "en-US".',
        default='en-US')
    parser.add_argument(
        '--dataset_name',
        help='Options: [sentiment140]',
        default='sentiment140')
    parser.add_argument(
        '--results_dir',
        help='Results directory',
        default='./results/')
    parser.add_argument(
        '--intent_session_ids_file',
        help='JSON file containing intent session IDs',
        default='./intent_session_ids.json')
    parser.add_argument(
        '--data_type',
        help='Data type: corr, inc, inc_with_corr',
        default="inc_with_corr")
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    dataset_arr = [args.dataset_name]

    data_dir_path = "../../data/twitter_sentiment_data/sentiment140"
    if args.data_type == "corr":
        data_dir_path += "_corrected_sentences/"
        app_name = "IntentClassification-{}-corr".format(args.dataset_name)
        app_description = "Intent Recognition App for Corrected Sentences"
    elif args.data_type == "inc":
        data_dir_path += "/"
        app_name = "IntentClassification-{}-inc".format(args.dataset_name)
        app_description = "Intent Recognition App for Original, Incorrect Sentences"
    else:
        data_dir_path += "_inc_with_corr_sentences/"
        app_name = "IntentClassification-{}-inc-corr".format(args.dataset_name)
        app_description = "Intent Recognition App for Original and Corrected Sentences"
    # data_dir_path += type + '.tsv'
    scores_file_root = args.results_dir + '{}/'.format(args.data_type)
    ensure_dir(scores_file_root)

    for dataset in dataset_arr:
        tags = SENTIMENT_TAGS[dataset]

        scores_file = scores_file_root + dataset + ".json"

        test_intents_labels_arr = []
        test_intents_arr = []
        intent_session_ids = []
        for intent_id, intent_name in tags.items():
            print("{}: {}".format(intent_id, intent_name))

            # Data dir path
            test_data_dir_path = data_dir_path + "test_dialogflow_{}.csv".format(intent_name)

            # ============= Test =============
            try:
                tsv_file = open(test_data_dir_path)
                reader = csv.reader(tsv_file, delimiter='\t')

                for row in reader:
                    test_intents_arr.append(row[0])
                    test_intents_labels_arr.append(intent_name)
            except:
                continue

        # Detect intent from texts
        detected_intent = detect_intent_texts(args.project_id, args.session_id, test_intents_arr, args.language_code)
        print(detected_intent)

        [precision, recall, fscore, support] = precision_recall_fscore_support(test_intents_labels_arr, detected_intent,
                                                                               average='micro')
        txt_print = "Results for {} dataset with {} data".format(dataset, args.data_type)
        print(txt_print)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1 score: ", fscore)
        print("support: ", support)

        result = {'precision': precision, 'recall': recall, 'f1': fscore}
        with open(scores_file, "w") as writer:
            json.dump(result, writer, indent=2)

        # Delete intents
        print("Deleting intents")
        with open(args.intent_session_ids_file) as parmFile:
            params = json.load(parmFile)
            intent_session_ids = params['intent_session_ids']
            for i in intent_session_ids:
                delete_intent(args.project_id, i)
