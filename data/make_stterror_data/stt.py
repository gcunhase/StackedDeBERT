
import speech_recognition as sr
import data.make_stterror_data.utils as utils

__author__ = 'Gwena Cunha'

"""
    Speech-To-Text Module
    Based on SpeechRecognition Github Repository: https://github.com/Uberi/speech_recognition/blob/master/examples/audio_transcribe.py
"""


class STT:

    def __init__(self, data_dir=""):
        print("Initializing STT Module")
        self.project_dir = utils.project_dir_name()
        self.data_dir = data_dir

        # Save global speech recognizer
        self.r = sr.Recognizer()

    def set_data_dir(self, data_dir):
        self.data_dir = data_dir

    def read_audio(self, audio_filename):
        # print("Reading audio")
        audio_file = self.project_dir + self.data_dir + audio_filename
        # use the audio file as the audio source
        with sr.AudioFile(audio_file) as source:
            audio = self.r.record(source)  # read the entire audio file
            return audio

    def text_from_stt_audio(self, audio, stt_type="sphinx", print_enabled=True):
        if "witai" in stt_type:
            return self.text_from_wit_ai(audio, print_enabled)

    def text_from_wit_ai(self, audio, print_enabled=True):
        # recognize speech using Wit.ai
        WIT_AI_KEY = "XHYECBHHVRWWWOUTBVF7QJ2TE6WJDYLH"  # Wit.ai keys are 32-character uppercase alphanumeric strings
        audio_text = None

        try:
            audio_text = self.r.recognize_wit(audio, key=WIT_AI_KEY)
            if print_enabled: print("STT with Wit.ai: " + audio_text)
        except sr.UnknownValueError:
            if print_enabled: print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            if print_enabled: print("Could not request results from Wit.ai service; {0}".format(e))

        return audio_text
