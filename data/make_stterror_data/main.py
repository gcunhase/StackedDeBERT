import os.path
from timeit import default_timer as timer

import data.make_stterror_data.utils as utils
from data.make_stterror_data.handler import HandlerIntent
from data.make_stterror_data.parser import snips_parser

__author__ = "Gwena Cunha"

""" Main module for Snips
    text -> TTS -> STT -> wrong text
"""


def main():

    # 1. Settings
    args = snips_parser()
    audio_file_dir = args.data_dir  # "data/intent_snips/"
    audios_relative_dir = args.audios_dir  # "results_tts_audios/"
    recovered_texts_relative_dir = args.recovered_texts_dir  # "results_stt_recovered_texts/"
    scores_dir = args.scores_dir  # "results_bleu_score/"
    text_filename = args.filename  # "test.tsv"
    tts_type_arr = args.tts_types  # ["gtts", "macsay"]
    stt_type_arr = args.stt_types  # ["witai"]
    audio_type = ".wav"
    textHandler = HandlerIntent(audio_file_dir, text_filename)  # Initialize TextHandler

    # 2. TTS from single file
    audios_dir = os.path.join(utils.project_dir_name(), audio_file_dir, audios_relative_dir)
    utils.ensure_dir(audios_dir)
    for tts_type in tts_type_arr:
        text_results_dir = "{}/{}/".format(audios_relative_dir, tts_type)
        textHandler.text2audio(audio_files_dir=text_results_dir, audio_type=audio_type, tts_type=tts_type)

    # 3. Apply STT to directory and get audio referring to that line
    recovered_texts_dir = os.path.join(utils.project_dir_name(), audio_file_dir, recovered_texts_relative_dir)
    utils.ensure_dir(recovered_texts_dir)
    for tts_type in tts_type_arr:
        text_results_dir = "{}/{}/".format(audios_relative_dir, tts_type)
        for stt_type in stt_type_arr:
            textHandler.audio2text(audio_files_dir=text_results_dir, audio_type=audio_type,
                                   stt_type=stt_type, recovered_texts_dir=recovered_texts_relative_dir,
                                   stt_out_text_filename="{}_{}_{}.tsv".format(text_filename.split('.tsv')[0], tts_type, stt_type))

    # 4. BLEU scores
    for tts_type in tts_type_arr:
        for stt_type in stt_type_arr:
            stt_out_text_filename = "{}_{}_{}.tsv".format(text_filename.split('.tsv')[0], tts_type, stt_type)
            scores_filename = "{}_{}_{}.txt".format(text_filename.split('.tsv')[0], tts_type, stt_type)
            textHandler.bleu_score(recovered_texts_dir=recovered_texts_relative_dir,
                                   stt_out_text_filename=stt_out_text_filename, scores_dir=scores_dir,
                                   scores_filename=scores_filename)


if __name__ == '__main__':
    time = timer()
    main()
    print("Program ran for %.2f minutes" % ((timer()-time)/60))