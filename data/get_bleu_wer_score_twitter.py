import os.path
from timeit import default_timer as timer

import utils
import numpy as np
import nltk
import os
from natsort import natsorted
import glob
from collections import defaultdict
import nltk.translate.bleu_score as bleu
import csv


__author__ = "Gwena Cunha"


class HandlerTwitter:

    def __init__(self, data_dir="", text_filename="test.tsv"):
        """ Initialize handler

            Args:
                text_filename (str): name of text file to read sentences from, where each line is a CONTENT (news/abstract)
        """
        print("Initializing Text Handler for Intent Corpus")
        self.project_dir = utils.get_project_path()
        self.data_dir = data_dir
        self.text_filename = text_filename

        # Needed to separate sentences in CONTENT
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')  # downloaded to /home/gwena/nltk_data

    def bleu_score(self, scores_filename="bleu_score.txt"):
        tsv_dict = read_tsv(os.path.join(self.project_dir, self.data_dir, self.text_filename))
        sentences_ref = tsv_dict['target']
        sentences_hyp = tsv_dict['sentence']
        ref, hyp = [], []
        for r, h in zip(sentences_ref, sentences_hyp):
            ref.append([r.lower().split()])
            hyp.append(h.lower().split())
        corpus_score = bleu.corpus_bleu(ref, hyp)
        with open(os.path.join(self.project_dir, self.data_dir, scores_filename), 'w') as f:
            f.write("{}".format(corpus_score))
        return corpus_score

    def wer_score(self, scores_filename="wer_score.txt"):
        tsv_dict = read_tsv(os.path.join(self.project_dir, self.data_dir, self.text_filename))
        sentences_ref = tsv_dict['target']
        sentences_hyp = tsv_dict['sentence']
        ref, hyp = [], []
        for r, h in zip(sentences_ref, sentences_hyp):
            ref.append([r.lower().split()])
            hyp.append(h.lower().split())
        corpus_score = 0
        len_ref = len(ref)
        for r, h in zip(ref, hyp):
            corpus_score += self.wer_score_sentence(r[0], h)  # Assumes only 1 reference
        corpus_score /= len_ref
        with open(os.path.join(self.project_dir, self.data_dir, scores_filename), 'w') as f:
            f.write("{}".format(corpus_score))
        return corpus_score

    def wer_score_sentence(self, ref, hyp):
        """ Calculation of WER with Levenshtein distance.

        Time/space complexity: O(nm)

        Source: https://martin-thoma.com/word-error-rate-calculation/

        :param ref: reference text (separated into words)
        :param hyp: hypotheses text (separated into words)
        :return: WER score
        """

        # Initialization
        d = np.zeros([len(ref) + 1, len(hyp) + 1], dtype=np.uint8)
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # print(d)

        # Computation
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # print(d)
        return d[len(ref)][len(hyp)]


def read_tsv(filename):
    tsv_dict = defaultdict(lambda: [])
    with open(filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            for k, v in row.items():
                tsv_dict[k].append(v)
        return tsv_dict


def main():

    # 1. Settings
    text_filename = "test.tsv"
    #textHandler = HandlerTwitter(data_dir="data/twitter_sentiment_data/sentiment140_inc_with_corr_sentences/", text_filename=text_filename)  # Initialize TextHandler
    textHandler = HandlerTwitter(data_dir="data/intent_data/stterror_data/chatbot/macsay_witai/", text_filename=text_filename)  # Initialize TextHandler

    # 4. BLEU scores
    scores_filename = "{}_bleu.txt".format(text_filename.split('.tsv')[0])
    textHandler.bleu_score(scores_filename=scores_filename)
    scores_filename = "{}_wer.txt".format(text_filename.split('.tsv')[0])
    textHandler.wer_score(scores_filename=scores_filename)


if __name__ == '__main__':
    time = timer()
    main()
    print("Program ran for %.2f minutes" % ((timer()-time)/60))
