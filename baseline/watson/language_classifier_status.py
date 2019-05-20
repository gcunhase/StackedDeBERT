from __future__ import print_function
import json

# from os.path import join, dirname
from ibm_watson import NaturalLanguageClassifierV1

# Source: https://github.com/watson-developer-cloud/python-sdk

complete = True
perc = 0.2

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

service = NaturalLanguageClassifierV1(url=url, iam_apikey=api_key)

# Get list of classifiers
classifiers = service.list_classifiers().get_result()
# print(json.dumps(classifiers, indent=2))

dataset_arr = ['ChatbotCorpus', 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']
for dataset in dataset_arr:
    params_d = params[dataset]
    nlc_id = params_d['nlc_id']
    # If service instance provides API key authentication

    # classifier
    # classifier_id = classifiers["classifiers"][0]
    # classifier_id = classifier_id['classifier_id']
    classifier_id = nlc_id
    status = service.get_classifier(classifier_id).get_result()
    print("Classifier for '{}' with ID {} is {}".format(status['name'], status['classifier_id'], status['status']))
    # print(json.dumps(status, indent=2))
