import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from itertools import count
from C4State import C4State
import typing
from tensorflow import keras
from Agent import *
from ModelTools import twohead_evaluate

model = typing.cast(
    Model, keras.models.load_model("az_models/model_4.keras"))

agent = AZAgent(model, "model_4", 500)

game = C4State()

while not game.is_terminal():
    print(game)
    agent_move = agent.get_action(game)
    print(agent_move)
    print(f"network evaluation of this position: {twohead_evaluate(game, agent.model)}")
    game.push(agent_move)
    if game.is_terminal():
        break
    print(game)
    print(game.legal_moves())
    user_move = int(input("Your move: "))
    while user_move not in game.legal_moves():
        user_move = int(input("Your move: "))
    game.push(user_move)

print(game)
