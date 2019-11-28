import dialogflow
# import dialogflow_v2 as dialogflow
from google.cloud import storage
import argparse
import uuid
from baseline.base_utils import INTENTION_TAGS
import csv
from sklearn.metrics import precision_recall_fscore_support
from utils import ensure_dir
import json

# os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', '/mnt/gwena/Gwena/IntentionClassifier/baseline_models/dialoglow/IntentRecognition-f82bbba966e2.json')
# https://dialogflow-python-client-v2.readthedocs.io/en/latest/

GOOGLE_APPLICATION_CREDENTIALS = 'newagent-4a8cc-64b72d6a5adb.json'  # 'IntentRecognition-f82bbba966e2.json'


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
        help='Options: [chatbot, askubuntu, webapplications, sentiment140, snips]',
        default='chatbot')
    parser.add_argument(
        '--dataset_fullname',
        help='Options: [ChatbotCorpus, AskUbuntuCorpus, WebApplicationsCorpus, sentiment140, snips]',
        default='ChatbotCorpus')
    parser.add_argument(
        '--results_dir',
        help='Results directory',
        default='./results/intent_stterror/')
    parser.add_argument(
        '--intent_session_ids_file',
        help='JSON file containing intent session IDs',
        default='./intent_session_ids.json')
    parser.add_argument(
        '--tts_stt',
        help='gtts_witai, macsay_witai',
        default="gtts_witai")
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    # dataset_arr = ['ChatbotCorpus']  # , 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']
    dataset_arr = [args.dataset_name]
    dataset_fullname_arr = [args.dataset_fullname]
    tts_stt = args.tts_stt

    complete = False

    data_dir_path = "../../data/"
    data_dir_path += "stterror_data/"
    scores_file_root = args.results_dir + 'inc_{}/'.format(tts_stt)
    ensure_dir(scores_file_root)

    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        scores_file = scores_file_root + dataset + ".json"

        test_intents_labels_arr = []
        test_intents_arr = []
        intent_session_ids = []
        for intent_id, intent_name in tags.items():
            print("{}: {}".format(intent_id, intent_name))

            # Data dir path
            test_data_dir_path = data_dir_path + "{}/{}/test_dialogflow_{}.csv".format(dataset.lower(), tts_stt,
                                                                                       intent_name)

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
        txt_print = "Results for {} dataset".format(dataset)
        txt_print += " with comp and incomplete data with {} perc".format(tts_stt)

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
