import json
import sys
import unicodecsv as csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from ibm_watson import NaturalLanguageClassifierV1
from utils import ensure_dir

# Source: https://github.com/joe4k/wdcutils/blob/master/notebooks/NLCPerformanceEval.ipynb

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
results_dir = params['results_dir']
ensure_dir(results_dir)


# Classifier methods
# Given a text string and a pointer to NLC instance and classifierID, get back NLC response
def getNLCresponse(nlc_instance, classifierID, string):
    # remove newlines from input text as that causes WCS to return an error
    string = string.replace("\n", "")
    print(string)
    classes = nlc_instance.classify(classifierID, string)
    return classes


# Process multiple text utterances (provided via csv file) in batch. Effectively, read the csv file and for each text
# utterance, get NLC response. Aggregate and return results.
def batchNLC(nlc_instance, classifierID, csvfile):
    test_classes = []
    nlcpredict_classes = []
    nlcpredict_confidence = []
    text = []
    i = 0
    print('reading csv file: ', csvfile)
    with open(csvfile, 'rb') as csvfile:
        # For better handling of utf8 encoded text
        csvReader = csv.reader(csvfile, encoding="utf-8-sig")
        for row in csvReader:
            #print(row)
            # Assume input text is 2 column csv file, first column is text
            # and second column is the label/class/intent
            # Sometimes, the text string includes commas which may split
            # the text across multiple colmns. The following code handles that.
            if len(row) > 2:
                qelements = row[0:len(row) - 1]
                utterance = ",".join(qelements)
                test_classes.append(row[len(row) - 1])
            else:
                utterance = row[0]
                test_classes.append(row[1])
            utterance = utterance.replace('\r', ' ')
            # print('i: ', i, ' testing row: ', utterance)

            # test_classes.append(row['class'])
            # print 'analyzing row: ', i, ' text: ', row['text']
            nlc_response = getNLCresponse(nlc_instance, classifierID, utterance)
            nlc_response = nlc_response.result
            if nlc_response['classes']:
                nlcpredict_classes.append(nlc_response['classes'][0]['class_name'])
                nlcpredict_confidence.append(nlc_response['classes'][0]['confidence'])
            else:
                nlcpredict_classes.append('')
                nlcpredict_confidence.append(0)
            text.append(utterance)

            i = i + 1
            if (i % 250 == 0):
                print("")
                print("Processed ", i, " records")
            if (i % 10 == 0):
                sys.stdout.write('.')
        print("")
        print("Finished processing ", i, " records")
    return test_classes, nlcpredict_classes, nlcpredict_confidence, text


# Plot confusion matrix as an image
def plot_conf_matrix(conf_matrix):
    plt.figure()
    plt.imshow(conf_matrix)
    plt.show()


# Print confusion matrix to a csv file
def confmatrix2csv(conf_matrix, labels, csvfile):
    with open(csvfile, 'wb') as csvfile:
        csvWriter = csv.writer(csvfile)
        row = list(labels)
        row.insert(0, "")
        csvWriter.writerow(row)
        for i in range(conf_matrix.shape[0]):
            row = list(conf_matrix[i])
            row.insert(0, labels[i])
            csvWriter.writerow(row)


# Create an object for your NLC instance
natural_language_classifier = NaturalLanguageClassifierV1(url=url, iam_apikey=api_key)

# Test performance
dataset_arr = ['ChatbotCorpus', 'AskUbuntuCorpus', 'WebApplicationsCorpus', 'snips']
for dataset in dataset_arr:
    print("======== {} ========".format(dataset))
    params_d = params[dataset]
    nlc_id = params_d['nlc_id']
    test_csv_file = params_d['test_csv_file']
    results_csv_file = results_dir + params_d['results_csv_file']
    scores_file = results_dir + params_d['scores_file']
    confmatrix_csv_file = results_dir + params_d['confmatrix_csv_file']

    # json.dumps(params)
    test_classes, nlcpredict_classes, nlcpredict_conf, text = batchNLC(natural_language_classifier, nlc_id,
                                                                       test_csv_file)

    # print results to csv file including original text, the correct label,
    # the predicted label and the confidence reported by NLC.
    csvfileOut = results_csv_file
    with open(csvfileOut, 'wb') as csvOut:
        outrow=['text', 'true class', 'NLC Predicted class', 'Confidence']
        csvWriter = csv.writer(csvOut, dialect='excel')
        csvWriter.writerow(outrow)
        for i in range(len(text)):
            outrow=[text[i], test_classes[i], nlcpredict_classes[i], str(nlcpredict_conf[i])]
            csvWriter.writerow(outrow)

    # Compute confusion matrix
    labels = list(set(test_classes))
    nlc_confusion_matrix = confusion_matrix(test_classes, nlcpredict_classes, labels)
    nlcConfMatrix = ConfusionMatrix(test_classes, nlcpredict_classes)

    # Print out confusion matrix with labels to csv file
    confmatrix2csv(nlc_confusion_matrix, labels, confmatrix_csv_file)

    nlcConfMatrix.plot()

    # Compute accuracy of classification
    acc = accuracy_score(test_classes, nlcpredict_classes)
    print('Classification Accuracy for {} dataset: {}'.format(dataset, acc))

    # print precision, recall and f1-scores for the different classes
    scores_all = classification_report(test_classes, nlcpredict_classes, labels=labels)
    print(scores_all)

    # Optional if you would like each of these metrics separately
    [precision, recall, fscore, support] = precision_recall_fscore_support(test_classes, nlcpredict_classes, labels=labels,
                                                                           average='micro')
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", fscore)
    print("support: ", support)

    result = {'precision': precision, 'recall': recall, 'f1': fscore, 'support': support, 'scores_all': scores_all}
    with open(scores_file, "w") as writer:
        json.dump(result, writer, indent=2)
