import os
import sys
import io
import torch
import time
import numpy as np
from collections import OrderedDict

import librosa

from TTS.models.tacotron import Tacotron
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config
from TTS.utils.text import text_to_sequence
from TTS.utils.synthesis import synthesis
from utils.text.symbols import symbols, phonemes
from TTS.utils.visual import visualize

class TTS_mod():
    def __init__(self, message):
        self.message = message
        self.MODEL_PATH = './stt_models/best_model.pth.tar'
        self.CONFIG_PATH = './stt_models/config.json'
        self.OUT_FOLDER = '/output'
        self.CONFIG = load_config(self.CONFIG_PATH)
        self.use_cuda = False


    def tts(self, model, text, CONFIG, use_cuda, ap):
        waveform, alignment, spectrogram, mel_spectrogram, stop_tokens = synthesis(model, text, CONFIG, use_cuda, ap)
        ap.save_wav(waveform, 'out.wav')
        return alignment, spectrogram, stop_tokens

    def load_model(self):
	    # load the model
        self.num_chars = len(phonemes) if self.CONFIG.use_phonemes else len(symbols)
        self.model = Tacotron(self.num_chars, self.CONFIG.embedding_size, self.CONFIG.audio['num_freq'], self.CONFIG.audio['num_mels'], self.CONFIG.r, attn_windowing=False)

        self.CONFIG.audio["preemphasis"] = 0.97
        self.ap = AudioProcessor(**self.CONFIG.audio)

	    # load model state
        if self.use_cuda:
            self.cp = torch.load(self.MODEL_PATH)
        else:
            self.cp = torch.load(self.MODEL_PATH, map_location=lambda storage, loc: storage)

        # load the model
        self.model.load_state_dict(self.cp['model'])
        if self.use_cuda:
            self.model.cuda()
        self.model.decoder.max_decoder_steps = 1000


        self.sentence = self.message
        align, spec, stop_tokens = self.tts(self.model, self.sentence, self.CONFIG, self.use_cuda, self.ap)

#r = TTS_mod('Hello, how are you doing?').load_model()
