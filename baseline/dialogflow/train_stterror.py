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

GOOGLE_APPLICATION_CREDENTIALS = 'newagent-4a8cc-64b72d6a5adb.json'


def explicit_auth():
    storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

    buckets = list(storage_client.list_buckets())
    print(buckets)


def create_intent(project_id, display_name, training_phrases_parts,
                  message_texts):
    """Create an intent of the given intent type."""
    intents_client = dialogflow.IntentsClient.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

    parent = intents_client.project_agent_path(project_id)
    training_phrases = []
    for training_phrases_part in training_phrases_parts:
        part = dialogflow.types.Intent.TrainingPhrase.Part(
            text=training_phrases_part)
        # Here we create a new training phrase for each provided part.
        training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
        training_phrases.append(training_phrase)

    text = dialogflow.types.Intent.Message.Text(text=message_texts)
    message = dialogflow.types.Intent.Message(text=text)

    intent = dialogflow.types.Intent(
        display_name=display_name,
        training_phrases=training_phrases,
        messages=[message])

    response = intents_client.create_intent(parent, intent)

    print('Intent created: {}'.format(response))
    return response


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
        '--tts_stt',
        help='TTS-STT combination: [gtts_witai, macsay_witai]',
        default="gtts_witai")
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    # Authenticate
    explicit_auth()

    # dataset_arr = ['ChatbotCorpus']  # , 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']
    dataset_arr = [args.dataset_name]
    dataset_fullname_arr = [args.dataset_fullname]

    complete = False
    tts_stt = args.tts_stt

    data_dir_path = "../../data/"
    data_dir_path += "stterror_data/"
    scores_file_root = args.results_dir + 'inc_{}/'.format(tts_stt)
    ensure_dir(scores_file_root)

    for dataset, dataset_fullname in zip(dataset_arr, dataset_fullname_arr):
        tags = INTENTION_TAGS[dataset_fullname]

        scores_file = scores_file_root + dataset + ".json"

        test_intents_labels_arr = []
        test_intents_arr = []
        print("Creating intents")
        intent_session_ids = []
        for intent_id, intent_name in tags.items():
            print("{}: {}".format(intent_id, intent_name))

            # Data dir path
            train_data_dir_path = data_dir_path + "{}/{}/train_dialogflow_{}.csv".format(dataset.lower(), tts_stt, intent_name)

            # ============= Train =============
            tsv_file = open(train_data_dir_path)
            reader = csv.reader(tsv_file, delimiter='\t')

            train_intents_arr = []
            for row in reader:
                train_intents_arr.append(row[0])

            # Create intent
            intent_response = create_intent(project_id=args.project_id, display_name=intent_name,
                                            training_phrases_parts=train_intents_arr, message_texts="")
            intent_session_ids.append(intent_response.name.split('/')[-1])
        result = {'intent_session_ids': intent_session_ids}
        with open('./intent_session_ids.json', "w") as writer:
            json.dump(result, writer, indent=2)