# -*- coding: utf-8 -*-

import data.make_stterror_data.utils as utils
import nltk
import os
from data.make_stterror_data.tts import TTS
from data.make_stterror_data.stt import STT
from natsort import natsorted
import glob
from collections import defaultdict
import nltk.translate.bleu_score as bleu

__author__ = 'Gwena Cunha'

""" Module to handle Intent NLU Corpora (Snips, AskUbuntu, WebApplications, Chatbot)
    Snips NLU Intent Corpus: https://github.com/snipsco/nlu-benchmark
    Others: https://github.com/sebischair/NLU-Evaluation-Corpora

    * Handles text before TTS and after STT (read and write)
    * Reads from text file where each line is a sentence to perform TTS on and saves the audio with filename
        corresponding to the line number
"""


class HandlerIntent:

    def __init__(self, data_dir="", text_filename="test.tsv"):
        """ Initialize handler

            Args:
                text_filename (str): name of text file to read sentences from, where each line is a CONTENT (news/abstract)
        """
        print("Initializing Text Handler for Intent Corpus")
        self.project_dir = utils.project_dir_name()
        self.data_dir = data_dir

        # Open file
        print("Opening tsv file")
        self.sentences, self.labels = self.set_sentence_labels(text_filename)

        # Needed to separate sentences in CONTENT
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')  # downloaded to /home/gwena/nltk_data

    def text2audio(self, audio_files_dir="audio_results/", audio_type=".wav", tts_type="gtts",
                   initial_idx=1, final_idx=float("inf")):
        """ Separate sentences in text

            Args:
                audio_type (str): default is .wav, but could be .mp3, .flac, anything
                num_lines (int): if float("inf"), then entire text file is read, otherwise, specified first number of lines is read
                initial_idx (int): line number to start reading from
                final_idx (int): line number to finish reading at
        """

        # Get audio from text file
        print("Get audio from text file with TTS type {}".format(tts_type))

        # Create folder to save all audios and initialize TTS object
        text_data_dir = self.data_dir
        audio_files_dir = self.data_dir + audio_files_dir
        tts = TTS(text_data_dir, audio_files_dir, audio_type=audio_type)

        print("Reading lines in file")
        # j = 1  # initial_idx
        for j, line in enumerate(self.sentences):
            print("Line %d" % j)
            if j >= initial_idx-1:
                if j <= final_idx-1:
                    # Clean string
                    print(line)
                    sent = utils.clean_string(line)
                    print(sent)
                    tts.read_text(sent, str(j), tts_type=tts_type)
                else:
                    break

    def set_sentence_labels(self, text_filename):
        text_file_dir = self.project_dir + self.data_dir + text_filename
        tsv_dict = utils.read_tsv(text_file_dir)

        # Extract sentences in tsv file
        return tsv_dict['sentence'], tsv_dict['label']

    def audio2text(self, audio_files_dir="audio_results/", audio_type=".wav", stt_type="sphinx",
                   recovered_texts_dir="recovered_text_results/", stt_out_text_filename="stt_out.txt"):
        """ Get text from audio files
            Included initial and final indexes option to limit the set that needs STT applied to

        :param text_filename: filename of original text file to save in final tsv
        :param audio_files_dir: string, directory containing all audio files
        :param audio_type:
        :param stt_type:
        :param stt_out_text_filename: string, name of file containing sentences after STT
        :param initial_idx:
        :param final_idx:
        :param stt_out_text_filename2:
        :return:
        """

        # Get text file from audio files
        print("Get text file from audio files with STT {}".format(stt_type))

        # Get audio files (full path)
        audio_files = glob.glob(os.path.join(self.project_dir, self.data_dir, audio_files_dir, "*"+audio_type))
        print(audio_files)

        # Natsort: sorts directories in natural ascending order
        audio_files_sorted = natsorted(audio_files)
        print(audio_files_sorted)

        # Initialize STT object
        stt = STT(self.data_dir + audio_files_dir)

        # Open file to write results
        print("Opening text file to write sentences")
        stt_out_text_file = os.path.join(self.project_dir, self.data_dir, recovered_texts_dir, stt_out_text_filename)
        # f_text = open(stt_out_text_file, 'wt')

        print("Applying STT")
        # Saves .tsv with labels, missing and target
        tsv_array = []
        count = 1
        for i, audio_file in enumerate(audio_files_sorted):
            print("Audio file: {}".format(audio_file))
            # get audio files in dir directory
            audio = stt.read_audio(audio_file.split('/')[-1])
            text_from_audio = stt.text_from_stt_audio(audio, stt_type=stt_type, print_enabled=False)
            print("Text from audio: {}".format(text_from_audio))
            if text_from_audio is not None:
                content_sent = text_from_audio + ". "
            else:
                content_sent = " "
            tsv_array.append([content_sent, self.labels[i], "", self.sentences[i]])
            # f_text.write(content_sent+"\n")
            count += 1

        utils.write_tsv(['sentence', 'label', 'missing', 'target'], tsv_array, stt_out_text_file)

        print("Closing STT output file")
        # f_text.close()

    def bleu_score(self, recovered_texts_dir="recovered_text_results/", stt_out_text_filename="stt_out.txt",
                   scores_dir="bleu_scores_results/", scores_filename="stt_out.txt"):
        utils.ensure_dir(os.path.join(self.project_dir, self.data_dir, scores_dir))
        tsv_dict = utils.read_tsv(os.path.join(self.project_dir, self.data_dir, recovered_texts_dir, stt_out_text_filename))
        sentences_ref = tsv_dict['target']
        sentences_hyp = tsv_dict['sentence']
        ref, hyp = [], []
        for r, h in zip(sentences_ref, sentences_hyp):
            ref.append([r.lower().split()])
            hyp.append(h.lower().split())
        corpus_score = bleu.corpus_bleu(ref, hyp)
        with open(os.path.join(self.project_dir, self.data_dir, scores_dir, scores_filename), 'w') as f:
            f.write("{}".format(corpus_score))
        return corpus_score