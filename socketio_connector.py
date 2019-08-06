import logging
import uuid
from sanic import Blueprint, response
from sanic.request import Request
from socketio import AsyncServer
from typing import Optional, Text, Any, List, Dict, Iterable

from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import UserMessage, OutputChannel

import deepspeech
from deepspeech import Model
import scipy.io.wavfile as wav
import io
import soundfile as sf
import wave
import numpy


import os
import sys
import io
import torch
import time
import numpy as np
from collections import OrderedDict
import urllib

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
from tts_func import TTS_mod
import pyaudio
import wave
import os
import pygame
from pygame import *


logger = logging.getLogger(__name__)


class SocketBlueprint(Blueprint):
    def __init__(self, sio: AsyncServer, socketio_path, *args, **kwargs):
        self.sio = sio
        self.socketio_path = socketio_path
        super(SocketBlueprint, self).__init__(*args, **kwargs)

    def register(self, app, options):
        self.sio.attach(app, self.socketio_path)
        super(SocketBlueprint, self).register(app, options)


class SocketIOOutput(OutputChannel):

    @classmethod
    def name(cls):
        return "socketio"

    def __init__(self, sio, sid, bot_message_evt, message):
        self.sio = sio
        self.sid = sid
        self.bot_message_evt = bot_message_evt
        self.message = message


    def tts(self, model, text, CONFIG, use_cuda, ap, OUT_FILE):
        import numpy as np
        waveform, alignment, spectrogram, mel_spectrogram, stop_tokens = synthesis(model, text, CONFIG, use_cuda, ap)
        ap.save_wav(waveform, OUT_FILE)
        wav_norm = waveform * (32767 / max(0.01, np.max(np.abs(waveform))))
        return alignment, spectrogram, stop_tokens, wav_norm


    def load_model(self, MODEL_PATH, sentence, CONFIG, use_cuda, OUT_FILE):
        # load the model
        num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
        model = Tacotron(num_chars, CONFIG.embedding_size, CONFIG.audio['num_freq'], CONFIG.audio['num_mels'], CONFIG.r, attn_windowing=False)

        # load the audio processor
        # CONFIG.audio["power"] = 1.3
        CONFIG.audio["preemphasis"] = 0.97
        ap = AudioProcessor(**CONFIG.audio)


        # load model state
        if use_cuda:
            cp = torch.load(MODEL_PATH)
        else:
            cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

        # load the model
        model.load_state_dict(cp['model'])
        if use_cuda:
            model.cuda()


        model.eval()
        model.decoder.max_decoder_steps = 1000
        align, spec, stop_tokens, wav_norm = self.tts(model, sentence, CONFIG, use_cuda, ap, OUT_FILE)
        return wav_norm

    async def _send_message(self, socket_id, response,  **kwargs: Any):
        # type: (Text, Any) -> None
        """Sends a message to the recipient using the bot event."""

        #tts_out = TTS_mod(response).load_model()
        #await self.sio.emit(self.bot_message_evt, response, room=socket_id)

        # Set constants

        MODEL_PATH = './tts_model/best_model.pth.tar'
        CONFIG_PATH = './tts_model/config.json'
        OUT_FILE = 'tts_out.wav'
        CONFIG = load_config(CONFIG_PATH)
        use_cuda = False


        wav_norm = self.load_model(MODEL_PATH, response['text'], CONFIG, use_cuda, OUT_FILE)




        #await self.sio.emit(self.bot_message_evt, {'text':response['text'], "user_utterance":"Hello", "link":"file://local/Users/juste/Desktop/rasa-demo/tts_out.wav"}, room=socket_id)
        await self.sio.emit(self.bot_message_evt, {'text':response['text'], "link":"https://file-examples.com/wp-content/uploads/2017/11/file_example_WAV_1MG.wav"}, room=socket_id)

    async def send_text_message(self, recipient_id: Text, message: Text, **kwargs: Any) -> None:
        """Send a message through this channel."""

        await self._send_message(self.sid, {"text": message})
        #await self._send_message(self.sid, {"attachment":self.mssage})

    async def send_image_url(self, recipient_id: Text, image_url: Text,  **kwargs: Any) -> None:
        """Sends an image. Default will just post the url as a string."""
        message = {"attachment": {"type": "image", "payload": {"src": image}}}
        await self._send_message(self.sid, message)

    async def send_text_with_buttons(self, recipient_id: Text, text: Text,
                               buttons: List[Dict[Text, Any]],
                               **kwargs: Any) -> None:
        """Sends buttons to the output."""

        message = {
            "text": text,
            "quick_replies": []
        }

        for button in buttons:
            message["quick_replies"].append({
                "content_type": "text",
                "title": button['title'],
                "payload": button['payload']
            })

        await self._send_message(self.sid, message)

    async def send_custom_message(self, recipient_id: Text,
                            elements: List[Dict[Text, Any]],  **kwargs: Any) -> None:
        """Sends elements to the output."""

        message = {"attachment": {
            "type": "template",
            "payload": {
                "template_type": "generic",
                "elements": elements[0]
            }}}

        await self._send_message(self.sid, message)


class SocketIOInput(InputChannel):
    """A socket.io input channel."""

    @classmethod
    def name(cls):
        return "socketio"

    @classmethod
    def from_credentials(cls, credentials):
        credentials = credentials or {}
        return cls(credentials.get("user_message_evt", "user_uttered"),
                   credentials.get("bot_message_evt", "bot_uttered"),
                   credentials.get("namespace"),
                   credentials.get("session_persistence", False),
                   credentials.get("socketio_path", "/socket.io"),
                   )

    def __init__(self,
                 user_message_evt: Text = "user_uttered",
                 bot_message_evt: Text = "bot_uttered",
                 namespace: Optional[Text] = None,
                 session_persistence: bool = False,
                 socketio_path: Optional[Text] = '/socket.io'
                 ):
        self.bot_message_evt = bot_message_evt
        self.session_persistence = session_persistence
        self.user_message_evt = user_message_evt
        self.namespace = namespace
        self.socketio_path = socketio_path

    def blueprint(self, on_new_message):
        sio = AsyncServer(async_mode="sanic")
        socketio_webhook = SocketBlueprint(
            sio, self.socketio_path, "socketio_webhook", __name__
        )

        @socketio_webhook.route("/", methods=['GET'])
        async def health(request):
            return response.json({"status": "ok"})

        @sio.on('connect', namespace=self.namespace)
        async def connect(sid, environ):
            logger.debug("User {} connected to socketIO endpoint.".format(sid))
            print('Connected!')

        @sio.on('disconnect', namespace=self.namespace)
        async def disconnect(sid):
            logger.debug("User {} disconnected from socketIO endpoint."
                         "".format(sid))

        @sio.on('session_request', namespace=self.namespace)
        async def session_request(sid, data):
            print('This is sessioin request')
            #print(data)
            #print(data['session_id'])
            if data is None:
                data = {}
            if 'session_id' not in data or data['session_id'] is None:
                data['session_id'] = uuid.uuid4().hex
            await sio.emit("session_confirm", data['session_id'], room=sid)
            logger.debug("User {} connected to socketIO endpoint."
                         "".format(sid))

        #@sio.on('recorder stopped', namespace=self.namespace)
        #async def get_audio(sid, data):
        #    print('This is what I got')
        #    print(data)

        @sio.on('user_uttered', namespace=self.namespace)
        async def handle_message(sid, data):

            output_channel = SocketIOOutput(sio, sid, self.bot_message_evt, data['message'])
            if data['message'] == "/get_started":
                message = data['message']
            else:
                ##receive audio as .ogg
                received_file = sid+'.wav'

                urllib.request.urlretrieve(data['message'], received_file)
                path = os.path.dirname(__file__)
                #print(path)
                #print(sid)
                # convert .ogg file into int16 wave file by ffmpeg
                #-ar 44100
                os.system("ffmpeg -y -i {0} -ar 16000 output_{1}.wav".format(received_file,sid))
                #os.system("ffmpeg -y -i {0} -c:a pcm_s161e output_{1}.wav".format(received_file,sid))
                N_FEATURES = 25
                N_CONTEXT = 9
                BEAM_WIDTH = 500
                LM_ALPHA = 0.75
                LM_BETA = 1.85


                ds = Model('deepspeech-0.5.1-models/output_graph.pbmm', N_FEATURES, N_CONTEXT, 'deepspeech-0.5.1-models/alphabet.txt', BEAM_WIDTH)
                fs, audio = wav.read("output_{0}.wav".format(sid))
                message = ds.stt(audio, fs)

                #await self.sio.emit(self.bot_message_evt, response, room=socket_id)
                await sio.emit("user_uttered", {"text":message}, room=sid)
                #ffmpeg -i input.flv -f s16le -acodec pcm_s16le output.raw


            

            if self.session_persistence:
                #if not data.get("session_id"):
                #    logger.warning("A message without a valid sender_id "
                #                   "was received. This message will be "
                #                   "ignored. Make sure to set a proper "
                #                   "session id using the "
                #                   "`session_request` socketIO event.")
                #    return
                #sender_id = data['session_id']
            #else:
                sender_id = sid


            message_rasa = UserMessage(message, output_channel, sender_id,
                                  input_channel=self.name())
            await on_new_message(message_rasa)

        return socketio_webhook
