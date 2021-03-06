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


def get_synonyms(word, number=3):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name().lower().replace("_", " "))
    synonyms = list(OrderedDict.fromkeys(synonyms))
    return synonyms[:number]


def preprocess(doc, nlp):
    clean_tokens = []
    doc = nlp(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


def tokenize(doc, nlp):
    """
    Returns a list of strings containing each token in `sentence`
    """
    tokens = []
    doc = nlp.tokenizer(doc)
    for token in doc:
        tokens.append(token.text)
    return tokens


def ngram_encode(str_test, HD_aphabet, aphabet, n_size):
    '''Method for mapping n-gram statistics of a word to an N-dimensional HD vector'''

    # Store n-gram statistics mapped to HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1])

    # Include extra symbols to the string
    full_str = '#' + str_test + '#'

    # Loops through all n-grams
    for il, l in enumerate(full_str[:-(n_size - 1)]):
        # Picks HD vector for the first symbol in the current n-gram
        hdgram = HD_aphabet[aphabet.find(full_str[il]), :]
        # Loops through the rest of symbols in the current n-gram
        for ng in range(1, n_size):
            # Two operations simultaneously; binding via elementwise multiplication; rotation via cyclic shift
            hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[il + ng]), :], ng)

        # Increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram
        HD_ngram += hdgram

    # Normalizes HD-vector so that its norm equals sqrt(N)
    HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (HD_ngram / np.linalg.norm(HD_ngram))
    return HD_ngram_norm


# ********* Semantic Hashing **************
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens


def semhash_corpus(corpus, nlp):
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence, nlp)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str, tokens)))
    return new_corpus
# ***********************


# ********* Data augmentation part **************
class MeraDataset():
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards

        It's the actual test set as defined in the paper that we comparing against."""

    def __init__(self, dataset_path, nlp, params):
        """ Instantiate the object.
            @param: dataset_path The directory which contains the data set."""
        self.nlp = nlp

        self.oversample = params['oversample']
        self.synonym_extra_samples = params['synonym_extra_samples']
        self.augment_extra_samples = params['augment_extra_samples']
        self.additional_synonyms = params['additional_synonyms']
        self.additional_augments = params['additional_augments']
        self.mistake_distance = params['mistake_distance']

        self.dataset_path = dataset_path
        self.X_test, self.y_test, self.X_train, self.y_train = self.load()
        self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                   'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                   'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                   'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                   's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                   'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                   'm': {'x': 6, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                   'h': {'x': 5, 'y': 1}, 'j': {'x': 6, 'y': 1}, 'k': {'x': 7, 'y': 1},
                                   'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2},
                                   'ß': {'x': 10, 'y': 2}, 'ü': {'x': 10, 'y': 2}, 'ä': {'x': 10, 'y': 0},
                                   'ö': {'x': 11, 'y': 0}}
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
        self.splits = self.stratified_split()

    def get_nearest_to_i(self, keyboard_cartesian):
        """ Get the nearest key to the one read.
            @params: keyboard_cartesian The layout of the QWERTY keyboard for English

            return dictionary of eaculidean distances for the characters"""
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i, j) < self.mistake_distance:  # was > 1.2
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def _shuffle_word(self, word, cutoff=0.7):
        """ Rearange the given characters in a word simulating typos given a probability.

            @param: word A single word coming from a sentence
            @param: cutoff The cutoff probability to make a change (default 0.9)

            return The word rearranged
            """
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc] in self.keyboard_cartesian:
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)

    def _euclidean_distance(self, a, b):
        """ Calculates the euclidean between 2 points in the keyboard
            @param: a Point one
            @param: b Point two

            return The euclidean distance between the two points"""
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)

    def _get_augment_sentence(self, sentence):
        return ' '.join([self._shuffle_word(item) for item in sentence.split(' ')])

    def _augment_sentence(self, sentence, num_samples):
        """ Augment the dataset of file with a sentence shuffled
            @param: sentence The sentence from the set
            @param: num_samples The number of sentences to genererate

            return A set of augmented sentences"""
        sentences = []
        for _ in range(num_samples):
            sentences.append(self._get_augment_sentence(sentence))
        sentences = list(set(sentences))
        # print("sentences", sentences)
        return sentences + [sentence]

    def _augment_split(self, X_train, y_train, num_samples=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)

            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            tmp_x = self._augment_sentence(X, num_samples)
            sample = [[Xs.append(item), ys.append(y)] for item in tmp_x]
        #             print(X, y)
        #             print(self.augmentedFile+str(self.nSamples)+".csv")

        with open(self.augmentedFile + str(self.nSamples) + "Train.csv", 'w', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            for i in range(0, len(Xs) - 1):
                fileWriter.writerow([Xs[i] + '\t' + ys[i]])
                # print(Xs[i], "\t", ys[i])
                # print(Xs[i])
            # fileWriter.writerows(Xs + ['\t'] + ys)
        return Xs, ys

    # Randomly replaces the nouns and verbs by synonyms
    def _synonym_word(self, word, cutoff=0.5):
        if random.uniform(0, 1.0) > cutoff and len(get_synonyms(word)) > 0 and word in nouns and word in verbs:
            return random.choice(get_synonyms(word))
        return word

    # Randomly replace words (nouns and verbs) in sentence by synonyms
    def _get_synonym_sentence(self, sentence, cutoff=0.5):
        return ' '.join([self._synonym_word(item, cutoff) for item in sentence.split(' ')])

    # For all classes except the largest ones; add duplicate (possibly augmented) samples until all classes have the same size
    def _oversample_split(self, X_train, y_train, synonym_extra_samples=False, augment_extra_samples=False):
        """ Split the oversampled train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset

            return Oversampled training dataset"""

        classes = {}
        for X, y in zip(X_train, y_train):
            if y not in classes:
                classes[y] = []
            classes[y].append(X)

        max_class_size = max([len(entries) for entries in classes.values()])

        Xs, ys = [], []
        for y in classes.keys():
            for i in range(max_class_size):
                sentence = classes[y][i % len(classes[y])]
                if i >= len(classes[y]):
                    if synonym_extra_samples:
                        sentence = self._get_synonym_sentence(sentence)
                    if augment_extra_samples:
                        sentence = self._get_augment_sentence(sentence)
                Xs.append(sentence)
                ys.append(y)

        #with open("./datasets/KL/Chatbot/train_augmented.csv", 'w', encoding='utf8') as csvFile:
        #    fileWriter = csv.writer(csvFile, delimiter='\t')
        #    for i in range(0, len(Xs) - 1):
        #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])

        return Xs, ys

    def _synonym_split(self, X_train, y_train, num_samples=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)

            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            sample = [[Xs.append(self._get_synonym_sentence(X)), ys.append(y)] for item in range(self.additional_synonyms)]
        #             print(X, y)

        with open("./datasets/KL/Chatbot/train_augmented.csv", 'w', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            for i in range(0, len(Xs) - 1):
                fileWriter.writerow([Xs[i] + '\t' + ys[i]])
        return Xs, ys

    def load(self):
        """ Load the file for now only the test.csv, train.csv files hardcoded

            return The vector separated in test, train and the labels for each one"""
        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='	')
            all_rows = list(readCSV)
            #             for i in all_rows:
            #                 if i ==  28823:
            #                     print(all_rows[i])
            X_test = [a[0] for a in all_rows]
            y_test = [a[1] for a in all_rows]

        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            X_train = [a[0] for a in all_rows]
            y_train = [a[1] for a in all_rows]
        return X_test, y_test, X_train, y_train

    def process_sentence(self, x):
        """ Clean the tokens from stop words in a sentence.
            @param x Sentence to get rid of stop words.

            returns clean string sentence"""
        clean_tokens = []
        doc = self.nlp.tokenizer(x)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    def process_batch(self, X):
        """See the progress as is coming along.

            return list[] of clean sentences"""
        return [self.process_sentence(a) for a in tqdm(X)]

    def stratified_split(self):
        """ Split data whole into stratified test and training sets, then remove stop word from sentences

            return list of dictionaries with keys train,test and values the x and y for each one"""
        self.X_train, self.X_test = (
        [preprocess(sentence, self.nlp) for sentence in self.X_train], [preprocess(sentence, self.nlp) for sentence in self.X_test])
        print(self.X_train)
        if self.oversample:
            self.X_train, self.y_train = self._oversample_split(self.X_train, self.y_train, self.synonym_extra_samples,
                                                                self.augment_extra_samples)
        if self.additional_synonyms > 0:
            self.X_train, self.y_train = self._synonym_split(self.X_train, self.y_train, self.additional_synonyms)
        if self.additional_augments > 0:
            self.X_train, self.y_train = self._augment_split(self.X_train, self.y_train, self.additional_augments)

        splits = [{"train": {"X": self.X_train, "y": self.y_train},
                   "test": {"X": self.X_test, "y": self.y_test}}]
        return splits

    def get_splits(self):
        """ Get the splitted sentences

            return splitted list of dictionaries"""
        return self.splits
# ****************************************************
