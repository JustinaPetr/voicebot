from socketio_mod import *
from rasa_core.agent import Agent
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.utils import EndpointConfig, read_endpoint_config

interpreter = NaturalLanguageInterpreter.create('./models/nlu/current')
action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
agent = Agent.load('./models/dialogue', interpreter = interpreter, action_endpoint= action_endpoint)

input_channel = SocketIOInput(
			user_message_evt="user_uttered",
			bot_message_evt="bot_uttered",
			namespace=None)


s = agent.handle_channels([input_channel], 5005, serve_forever=True)