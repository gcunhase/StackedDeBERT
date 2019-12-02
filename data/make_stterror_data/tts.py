import data.make_stterror_data.utils as utils
import os
import sys
import subprocess

# TTS imports
from gtts import gTTS
import pyttsx3

# sys.path.append("~/PycharmProjects/pyfestival")  # https://github.com/techiaith/pyfestival/pull/4
# import festival

__author__ = 'Gwena Cunha'

"""
    Text-To-Speech Module
"""


class TTS:

    def __init__(self, data_dir="", result_dir="", audio_type=".wav"):
        print("Initializing TTS Module")
        self.audio_type = audio_type

        self.project_dir = utils.project_dir_name()
        self.data_dir = data_dir
        self.result_dir = result_dir
        utils.ensure_dir(self.project_dir + self.result_dir)

    def set_data_dir(self, data_dir):
        self.data_dir = data_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def read_text(self, text, sentence_id, tts_type="gtts"):
        if "macsay" in tts_type:  # Mac
            self.audio_from_mac_say(text, sentence_id)
        else:  # google gtts
            self.audio_from_google(text, sentence_id)

    def audio_from_google(self, text, sentence_id):
        gtts = gTTS(text=text, lang='en')  # , slow=False)
        partial_saved_audio_filename = self.project_dir + self.result_dir + "gtts_" + sentence_id
        tmp_saved_audio_filename = partial_saved_audio_filename + "_tmp" + self.audio_type
        final_saved_audio_filename = partial_saved_audio_filename + self.audio_type
        gtts.save(tmp_saved_audio_filename)
        # fix_missing_riff_header = "ffmpeg - i "+tmp_saved_audio_filename+" -y "+final_saved_audio_filename
        # Making ffmpeg quieter (less verbose): ffmpeg -nostats -loglevel 0 -i 2.mp3 ~/PycharmProjects/STTError/assets/2.mp3
        subprocess.call(
            ["ffmpeg", "-nostats", "-loglevel", "0", "-i", tmp_saved_audio_filename, "-y", final_saved_audio_filename])
        # os.system("ffmpeg -nostats -loglevel 0 -i {} -y {}".format(tmp_saved_audio_filename, final_saved_audio_filename))
        # Remove tmp_saved_audio_filename
        subprocess.call(["rm", tmp_saved_audio_filename])
        return final_saved_audio_filename

    def audio_from_mac_say(self, text, sentence_id):
        """ Mac's say command: Mac Systems
        """
        for voice in ['Fred']:  # ['Alex', 'Fred', 'Victoria']
            partial_saved_audio_filename = self.project_dir + self.result_dir + "macsay_" + sentence_id
            tmp_saved_audio_filename = partial_saved_audio_filename + "_tmp.aiff"
            final_saved_audio_filename = partial_saved_audio_filename + self.audio_type

            subprocess.call(["say", "-o", tmp_saved_audio_filename, "-v", voice, text])
            subprocess.call(["ffmpeg", "-nostats", "-loglevel", "0", "-i", tmp_saved_audio_filename, "-y",
                             final_saved_audio_filename])
            # os.system("say -o {} -v {} {}".format(tmp_saved_audio_filename, voice, text))
            # os.system("ffmpeg -nostats -loglevel 0 -i {} -y {}".format(tmp_saved_audio_filename, final_saved_audio_filename))

            # Remove tmp_saved_audio_filename
            subprocess.call(["rm", tmp_saved_audio_filename])
        return final_saved_audio_filename
