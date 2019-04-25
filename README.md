# Sara - the Rasa Demo Bot (Voice)
This repo contains the code for the Sara voice project. To run this example you will have to install the following:  
- Sara, the Rasa demo bot (installation instructions below)
- [Mozilla TTS](https://github.com/mozilla/TTS/tree/db7f3d3)  
- [Mozilla TTS model](https://drive.google.com/drive/folders/1GU8WGix98WrR3ayjoiirmmbLUZzwg4n0) (place the downloaded files in a tts_models directory)
- [Mozilla Deep Speech](https://github.com/mozilla/DeepSpeech)  
- [Mozilla Deep Speech model](https://drive.google.com/drive/folders/1GU8WGix98WrR3ayjoiirmmbLUZzwg4n0) (place the downloaded files in a models_stt directory)

# How to use this demo  
Right now the demo is pretty dumb - when the user sends an assistant a message, the bot runs STT on a example audio file called LDC93S1.wav and passes the result text the Rasa Stack. Once the stack makes a prediction on how to respond, the output is converted into a out.wav file and exported locally.  

The main part where the integration between the Rasa Stack and Mozilla happens is a connector (socketio_mod.py file). For the sake of demonstration this is the same socetio file as a regular Rasa socketio connector, but to integrate it with Mozilla we modified the input channel method handle_message() and output channel method _send_message(). 

To run this example run:  
- python run_bot.py (this should load the Rasa bot on a server)
- cd bot_ui (got to a directory which contains the example bot ui)
- php -S localhost:8080 (start the UI on a local server and talk to you bot after navigating to http://localhost:8080)

## 🤖 How to install and run Sara

To install Sara, please clone the repo and run:

```
cd rasa-demo
pip install -e .
```
This will install the bot and all of its requirements.
Note that it was written in Python 3 so might not work with PY2.

To train the core model: `make train-core` (this will take 2h+ and a significant amount of memory to train,
if you want to train it faster, try the training command with
`--augmentation 0`)

To train the NLU model: `make train-nlu`

To run Sara with both these models:
```
docker run -p 8000:8000 rasa/duckling
make run-cmdline
```

There are some custom actions that require connections to external services,
specifically `ActionSubscribeNewsletter` and `ActionStoreSalesInfo`. For these
to run you would need to have your own MailChimp newsletter and a Google sheet
to connect to.

If you would like to run Sara on your website, follow the instructions
[here](https://github.com/mrbot-ai/rasa-webchat) to place the chat widget on
your website.