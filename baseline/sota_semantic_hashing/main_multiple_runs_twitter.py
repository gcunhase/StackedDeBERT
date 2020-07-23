#coding: utf-8
from __future__ import unicode_literals
import sys
import re
import os
import codecs
import json
import csv
import spacy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn import model_selection
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid
import math
import random
from tqdm import tqdm
import timeit
import gc
import spacy

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

from baseline.sota_semantic_hashing.text_utils import MeraDataset, semhash_corpus, preprocess, tokenize, ngram_encode

from baseline.sota_semantic_hashing.classifiers import benchmark, plot_results

# Classifiers
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
from utils import ensure_dir

'''
activate venv37_bert

Paper: Subword Semantic Hashing for Intent Classification on Small Datasets
Paper ArXiv: https://arxiv.org/abs/1810.07150
Code Source: https://github.com/kumar-shridhar/Know-Your-Intent/blob/master/updated_semhash_pipeline.ipynb
'''

print(sys.path)
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('/mnt/gwena/anaconda3/lib/python3.7/site-packages/spacy/data/en/en_core_web_sm-2.1.0')


def read_CSV_datafile(filename):
    X = []
    y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            X.append(row[0])
            y.append(intent_dict[row[1]])
    return X, y


def get_vectorizer(corpus):
    vectorizer = CountVectorizer(ngram_range=(2, 4), analyzer='char')
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()


def data_for_training():
    vectorizer, feature_names = get_vectorizer(X_train_raw)

    X_train_no_HD = vectorizer.transform(X_train_raw).toarray()
    X_test_no_HD = vectorizer.transform(X_test_raw).toarray()

    return X_train_no_HD, y_train_raw, X_test_no_HD, y_test_raw, feature_names


data_type_arr = ["corr", "inc", "inc_with_corr"]
run_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataset_name = 'sentiment140'
results_root_dir_tmp = './results/results_trigram_hash_twitter_{}_768_10runs_macro2'.format(dataset_name)
ensure_dir(results_root_dir_tmp)

for data_type in data_type_arr:
    for run in run_arr:
        data_dir_path = "../../data/twitter_sentiment_data/sentiment140"
        if data_type == "corr":
            data_dir_path += "_corrected_sentences/"
        elif data_type == "inc":
            data_dir_path += "/"
        else:  # data_type == "inc_with_corr":
            data_dir_path += "_inc_with_corr_sentences/"

        data_name = dataset_name
        data_dir_name = data_dir_path

        results_root_dir = results_root_dir_tmp + '/{}_run{}'.format(data_type, run)
        ensure_dir(results_root_dir)

        # Hyperparameters
        params = {'benchmark_dataset': data_name,  # Choose from ['sentiment140']
                  'oversample': True,  # Whether to oversample small classes or not. True in the paper
                  'synonym_extra_samples': False,
                  # Whether to replace words by synonyms in the oversampled samples. True in the paper
                  'augment_extra_samples': False,
                  # Whether to add random spelling mistakes in the oversampled samples. False in the paper
                  'additional_synonyms': 0,
                  # How many extra synonym augmented sentences to add for each sentence. 0 in the paper
                  'additional_augments': 0,
                  # How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
                  'mistake_distance': 2.1,  # How far away on the keyboard a mistake can be
                  'N': 768,  # desired dimensionality of HD vectors
                  'n_size': 3,  # n-gram size
                  'alphabet': 'abcdefghijklmnopqrstuvwxyz#',
                  # fix the alphabet. Note, we assume that capital letters are not in use
                  'seed': run,  # 1, for reproducibility
                  'results_dir': results_root_dir,
                  'data_dir': data_dir_name,
                  'log_csv_filename': data_name + '_log.csv',
                  'log_txt_filename': data_name + '_log.txt',
                  'png_plot_filename': data_name + '_plot.png',
                  'log_f1_filename': data_name + '_f1.txt',
                  'log_f1_macro_filename': data_name + '_f1_macro.txt',
                  'log_precision_macro_filename': data_name + '_precision_macro.txt',
                  'log_recall_macro_filename': data_name + '_recall_macro.txt'
                  }

        benchmark_dataset = params['benchmark_dataset']
        oversample = params['oversample']
        synonym_extra_samples = params['synonym_extra_samples']
        augment_extra_samples = params['augment_extra_samples']
        additional_synonyms = params['additional_synonyms']
        additional_augments = params['additional_augments']
        mistake_distance = params['mistake_distance']
        N = params['N']
        n_size = params['n_size']
        aphabet = params['alphabet']
        data_dir = params['data_dir']
        results_dir = params['results_dir']
        ensure_dir(results_dir)

        #np.random.seed()  # params['seed'])
        np.random.seed(seed=params['seed'])
        HD_aphabet = 2 * (np.random.randn(len(aphabet), N) < 0) - 1  # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet

        if benchmark_dataset == "sentiment140":
            intent_dict = {"Negative": 0, "Positive": 1}

        filename_train = data_dir + "/train_semantic_hashing.csv"
        filename_test = data_dir + "/test_semantic_hashing.csv"

        print(filename_train)
        dataset = MeraDataset(filename_train, nlp, params)
        print("mera****************************")
        splits = dataset.get_splits()
        xS_train = []
        yS_train = []
        for elem in splits[0]["train"]["X"]:
            xS_train.append(elem)
        print(xS_train[:5])

        for elem in splits[0]["train"]["y"]:
            yS_train.append(intent_dict[elem])

        print(len(xS_train))

        # Read CSV data
        X_train_raw, y_train_raw = read_CSV_datafile(filename=filename_train)
        X_test_raw, y_test_raw = read_CSV_datafile(filename=filename_test)
        print(y_train_raw[:5])

        X_train_raw = xS_train
        y_train_raw = yS_train

        print("Training data samples: \n", X_train_raw, "\n\n")
        print("Class Labels: \n", y_train_raw, "\n\n")
        print("Size of Training Data: {}".format(len(X_train_raw)))

        # Semantic Hashing
        X_train_raw = semhash_corpus(X_train_raw, nlp)
        X_test_raw = semhash_corpus(X_test_raw, nlp)

        # print(timeit.Timer("for x in range(100): semhash_corpus(X_train_raw,nlp)", "gc.enable()").timeit())

        # print(X_train_raw[:5])
        # print(y_train_raw[:5])

        # Data for training
        X_train_no_HD, y_train, X_test_no_HD, y_test, feature_names = data_for_training()
        # print(X_train_raw[:5])

        # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding
        for i in range(len(X_train_raw)):
            X_train_raw[i] = ngram_encode(X_train_raw[i], HD_aphabet, aphabet, n_size)

        # print(X_train_raw[:5])
        for i in range(len(X_test_raw)):
            X_test_raw[i] = ngram_encode(X_test_raw[i], HD_aphabet, aphabet, n_size)
        # print(X_test_raw[:5])

        # print(print(timeit.Timer("for x in range(100): ngram_encode(X_train_raw[i], HD_aphabet, aphabet, n_size)", "gc.enable()").timeit()))

        X_train, y_train, X_test, y_test = X_train_raw, y_train_raw, X_test_raw, y_test_raw
        # print(X_train[:5])

        # Classifiers
        for _ in enumerate(range(1)):
            i_s = 0
            split = 0
            print("Evaluating Split {}".format(i_s))
            #     X_train, y_train, X_test, y_test, feature_names = data_for_training()
            target_names = None
            if benchmark_dataset == "sentiment140":
                target_names = ["Negative", "Positive"]

            #     print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
            #     print("Train Size: {}\nTest Size: {}".format(X_train, X_test.shape[0]))
            results = []
            # alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
            parameters_mlp = {'hidden_layer_sizes': [(100, 50), (300, 100), (300, 200, 100)]}
            parameters_RF = {"n_estimators": [50, 60, 70],
                             "min_samples_leaf": [1, 11]}
            k_range = list(range(3, 7))
            parameters_knn = {'n_neighbors': k_range}
            knn = KNeighborsClassifier(n_neighbors=5)
            for clf, name in [
                # (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                # (GridSearchCV(knn, parameters_knn, cv=5), "GridSearchCV-KNN"),
                # (Perceptron(n_iter=50), "Perceptron"),
                # (GridSearchCV(MLPClassifier(activation='tanh'), parameters_mlp, cv=5),"GridSearchMLP"),
                # (MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", max_iter=300), "MLP"),
                # (MLPClassifier(hidden_layer_sizes=(300, 100, 50), activation="logistic", max_iter=500), "MLP"),
                (MLPClassifier(hidden_layer_sizes=(300, 100, 50), activation="tanh", max_iter=500), "MLP"),
                # (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                # (KNeighborsClassifier(n_neighbors=1), "kNN"),
                # (KNeighborsClassifier(n_neighbors=3), "kNN"),
                # (KNeighborsClassifier(n_neighbors=5), "kNN"),
                # (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (GridSearchCV(RandomForestClassifier(n_estimators=10000900), parameters_RF, cv=5), "GridSearchRF"),
                # (RandomForestClassifier(n_estimators=10), "Random forest"),
                (RandomForestClassifier(n_estimators=50), "Random forest")
            ]:
                print('=' * 80)
                print(name)
                result = benchmark(clf, name, X_train, y_train, X_test, y_test, target_names,
                                   feature_names=feature_names, params=params)
                results.append(result)

            # print('parameters')
            # print(clf.grid_scores_[0])
            # print('CV Validation Score')
            # print(clf.grid_scores_[0].cv_validation_scores)
            # print('Mean Validation Score')
            # print(clf.grid_scores_[0].mean_validation_score)
            # grid_mean_scores = [result.mean_validation_score for result in clf.grid_scores_]
            # print(grid_mean_scores)
            # plt.plot(k_range, grid_mean_scores)
            # plt.xlabel('Value of K for KNN')
            # plt.ylabel('Cross-Validated Accuracy')

            # parameters_Linearsvc = [{'C': [1, 10], 'gamma': [0.1,1.0]}]
            for penalty in ["l2", "l1"]:
                print('=' * 80)
                print("%s penalty" % penalty.upper())
                # Train Liblinear model
                # grid=(GridSearchCV(LinearSVC,parameters_Linearsvc, cv=10),"gridsearchSVC")
                # results.append(benchmark(LinearSVC(penalty=penalty), X_train, y_train, X_test, y_test, target_names,
                # feature_names=feature_names, params=params))

                result = benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3), "LinearSVC-" + penalty.upper(),
                                   X_train, y_train, X_test, y_test, target_names,
                                   feature_names=feature_names, params=params)
                results.append(result)

                # Train SGD model
                result = benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                                 penalty=penalty), "SGD-" + penalty.upper(),
                                   X_train, y_train, X_test, y_test, target_names,
                                   feature_names=feature_names, params=params)
                results.append(result)

            # Train SGD with Elastic Net penalty
            print('=' * 80)
            name = "Elastic-Net penalty"
            print(name)
            results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                                   penalty="elasticnet"), name,
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names, params=params))

            # Train NearestCentroid without threshold
            print('=' * 80)
            name = "Nearest Centroid"  # "(aka Rocchio classifier)"
            print(name)
            results.append(benchmark(NearestCentroid(), name,
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names, params=params))

            # Train sparse Naive Bayes classifiers
            print('=' * 80)
            name = "Naive Bayes"
            print(name)
            print('Cant do it with negatives from HD!')
            #     results.append(benchmark(MultinomialNB(alpha=.01),
            #                              X_train, y_train, X_test, y_test, target_names,
            #                              feature_names=feature_names, params=params))

            #     result = benchmark(BernoulliNB(alpha=.01),
            #                              X_train, y_train, X_test, y_test, target_names,
            #                              feature_names=feature_names, params=params)
            #     results.append(result)

            result = benchmark(BernoulliNB(alpha=.01), name, X_train, y_train, X_test, y_test, target_names,
                               feature_names=feature_names, params=params)
            results.append(result)

            print('=' * 80)
            name = "LinearSVC-L1"  # name = "LinearSVC with L1-based feature selection"
            print(name)
            # The smaller C, the stronger the regularization.
            # The more regularization, the more sparsity.

            # uncommenting more parameters will give better exploring power but will
            # increase processing time in a combinatorial way
            result = benchmark(Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                                tol=1e-3))),
                ('classification', LinearSVC(penalty="l2"))]), name,
                X_train, y_train, X_test, y_test, target_names,
                feature_names=feature_names, params=params)
            results.append(result)
            # print(grid.grid_scores_)
            # KMeans clustering algorithm
            print('=' * 80)
            name = "KMeans"
            print(name)
            results.append(benchmark(KMeans(n_clusters=2, init='k-means++', max_iter=300,
                                            verbose=0, random_state=0, tol=1e-4), name,
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names, params=params))

            print('=' * 80)
            name = "Logistic Regression"
            print(name)
            # kfold = model_selection.KFold(n_splits=2, random_state=0)
            # model = LinearDiscriminantAnalysis()
            results.append(benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False,
                                                        fit_intercept=True, intercept_scaling=1, max_iter=100,
                                                        multi_class='ovr', n_jobs=1, penalty='l2',
                                                        random_state=None,
                                                        solver='liblinear', tol=0.0001, verbose=0,
                                                        warm_start=False),
                                     name,
                                     X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names, params=params))

            plot_results(results, benchmark_dataset, results_dir, params['png_plot_filename'], plot=False)
