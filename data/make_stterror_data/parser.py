import os
import argparse
from data.make_stterror_data import utils
import platform

FILENAME = 'train'


def snips_parser():

    if platform.system() == 'Darwin':
        tts_type_arr = ["gtts", "macsay"]
    elif platform.system() == 'Linux':
        tts_type_arr = ["gtts"]
    else:  # 'Windows'
        tts_type_arr = ["gtts"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/intent_snips/", type=str,
                        help="The input data dir relative to PROJECT ROOT DIRECTORY. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--audios_dir", default="results_{}_tts_audios/".format(FILENAME), type=str,
                        help="Path where audio should be saved relative to DATA DIRECTORY.")
    parser.add_argument("--recovered_texts_dir", default="results_{}_stt_recovered_texts/".format(FILENAME), type=str,
                        help="Path where recovered texts should be saved relative to DATA DIRECTORY.")
    parser.add_argument("--scores_dir", default="results_{}_bleu_score/".format(FILENAME), type=str,
                        help="Path where BLEU scores should be saved relative to DATA DIRECTORY.")
    parser.add_argument("--filename", default="{}.tsv".format(FILENAME), type=str,
                        help="File for Text-to-Speech and Speech-to-Text.")
    parser.add_argument("--tts_types", default=tts_type_arr, type=list,
                        help="List with TTS to perform. Options=[gtts, macsay (Mac)]")
    parser.add_argument("--stt_types", default=["witai"], type=list,
                        help="List with TTS to perform. Options=[witai]")

    return parser.parse_args()
