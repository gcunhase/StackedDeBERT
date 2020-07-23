from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf, clf_name, X_train, y_train, X_test, y_test, target_names,
              print_report=True, feature_names=None, print_top10=False,
              print_cm=True, params=None):

    benchmark_dataset = params['benchmark_dataset']
    oversample = params['oversample']
    synonym_extra_samples = params['synonym_extra_samples']
    augment_extra_samples = params['augment_extra_samples']
    additional_synonyms = params['additional_synonyms']
    additional_augments = params['additional_augments']
    mistake_distance = params['mistake_distance']
    results_dir = params['results_dir']
    log_csv_filename = params['log_csv_filename']
    log_txt_filename = params['log_txt_filename']
    log_f1_filename = params['log_f1_filename']
    log_f1_macro_filename = params['log_f1_macro_filename']
    log_precision_macro_filename = params['log_precision_macro_filename']
    log_recall_macro_filename = params['log_recall_macro_filename']

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    f1_score = metrics.f1_score(y_test, pred, average='weighted')
    f1_micro_score = metrics.f1_score(y_test, pred, average='micro')
    f1_macro_score = metrics.f1_score(y_test, pred, average='macro')
    precision_macro_score = metrics.precision_score(y_test, pred, average='macro')
    recall_macro_score = metrics.recall_score(y_test, pred, average='macro')

    # Accuracy and F1-micro scores are the same
    print("accuracy:   %0.3f" % score)
    print("F1-micro score:   %0.3f" % f1_micro_score)
    print("F1-macro score:   %0.3f" % f1_macro_score)
    print("Precision-macro score:   %0.3f" % precision_macro_score)
    print("Recall-macro score:   %0.3f" % recall_macro_score)
    # print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(
                    ["Make Update", "Setup Printer", "Shutdown Computer", "Software Recommendation", "None"]):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))
        print()

    classification_report = metrics.classification_report(y_test, pred, labels=range(len(target_names)),
                                                          target_names=target_names)
    if print_report:
        print("classification report:")
        print(classification_report)

    confusion_matrix = metrics.confusion_matrix(y_test, pred)
    if print_cm:
        print("confusion matrix:")
        print(confusion_matrix)

    with open("{}/{}".format(results_dir, log_csv_filename), 'a', encoding='utf8') as csvFile:
        fileWriter = csv.writer(csvFile, delimiter='\t')
        fileWriter.writerow(
            [benchmark_dataset, str(clf), str(oversample), str(synonym_extra_samples), str(augment_extra_samples),
             str(additional_synonyms), str(additional_augments), str(mistake_distance), str(score), str(f1_score)])

    with open("{}/{}".format(results_dir, log_txt_filename), 'a', encoding='utf8') as txtFile:
        txtFile.write("===================\n"
                      "Training: {clf}\n"
                      "confusion_matrix: {confusion_matrix}\n"
                      "classification report: {classification_report}\n"
                      "f1 micro: {f1_micro}\n".
                      format(clf=clf, confusion_matrix=confusion_matrix, classification_report=classification_report,
                             f1_micro=f1_micro_score))

    with open("{}/{}.txt".format(results_dir, log_f1_filename), 'a', encoding='utf8') as txtFile:
        txtFile.write("{clf_name}: {f1_micro}\n".format(clf_name=clf_name, f1_micro=f1_micro_score))

    with open("{}/{}.txt".format(results_dir, log_f1_macro_filename), 'a', encoding='utf8') as txtFile:
        txtFile.write("{clf_name}: {f1_macro}\n".format(clf_name=clf_name, f1_macro=f1_macro_score))
    with open("{}/{}.txt".format(results_dir, log_precision_macro_filename), 'a', encoding='utf8') as txtFile:
         txtFile.write("{clf_name}: {precision_macro}\n".format(clf_name=clf_name, precision_macro=precision_macro_score))
    with open("{}/{}.txt".format(results_dir, log_recall_macro_filename), 'a', encoding='utf8') as txtFile:
        txtFile.write("{clf_name}: {recall_macro}\n".format(clf_name=clf_name, recall_macro=recall_macro_score))

    clf_descr = clf_name  # str(clf).split('(')[0]
    return clf_descr, score, f1_micro_score, train_time, test_time, f1_score, f1_macro_score, precision_macro_score, recall_macro_score


def plot_results(results, benchmark_dataset, results_dir, log_f1_filename, plot=True):
    # make some plots
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(9)]

    clf_names, score, f1_micro_score, training_time, test_time, f1_score, f1_macro_score, precision_macro_score, recall_macro_score = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score for {} Corpus".format(benchmark_dataset))
    plt.barh(indices, f1_micro_score, .15, label="F1 score - micro", color='navy')
    plt.barh(indices + .2, f1_macro_score, .15, label="F1 score - macro", color='#1f77b4')
    plt.barh(indices + .4, training_time, .15, label="Training time",
             color='c')
    plt.barh(indices + .6, test_time, .15, label="Test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    if plot:
        plt.show()
    else:
        plt.savefig("{}/{}".format(results_dir, log_f1_filename))
