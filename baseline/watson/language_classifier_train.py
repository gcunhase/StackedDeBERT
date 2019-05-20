from __future__ import print_function
import json
import os

# from os.path import join, dirname
from ibm_watson import NaturalLanguageClassifierV1

# Source: https://github.com/watson-developer-cloud/python-sdk

complete = True
perc = 0.8

# Load params
if complete:
    nlcParamsFile = './watson_params_complete.json'
else:
    nlcParamsFile = './watson_params_incomplete_{}.json'.format(perc)
params = ''
with open(nlcParamsFile) as parmFile:
    params = json.load(parmFile)
url = params['url']
api_key = params['api_key']

dataset_arr = ['ChatbotCorpus', 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']
for dataset in dataset_arr:
    params_d = params[dataset]
    train_data_path = params_d['train_csv_file']

    # If service instance provides API key authentication
    service = NaturalLanguageClassifierV1(url=url, iam_apikey=api_key)

    classifiers = service.list_classifiers().get_result()
    print(json.dumps(classifiers, indent=2))

    # create a classifier
    with open(os.path.join(os.path.dirname(__file__), train_data_path), 'rb') as training_data:
        if complete:
            metadata = json.dumps({'name': dataset, 'language': 'en'})
        else:
            metadata = json.dumps({'name': "{} Incomplete {}".format(dataset, perc), 'language': 'en'})
        classifier = service.create_classifier(
            metadata=metadata, training_data=training_data).get_result()
        classifier_id = classifier['classifier_id']
        print(json.dumps(classifier, indent=2))

    status = service.get_classifier(classifier_id).get_result()
    print(json.dumps(status, indent=2))

    # Delete classifier
    # delete = service.delete_classifier(classifier_id).get_result()
    # print(json.dumps(delete, indent=2))

    # example of raising a ValueError
    # print(json.dumps(
    #     service.create_classifier(training_data='', name='weather3', metadata='metadata'),
    #     indent=2))

