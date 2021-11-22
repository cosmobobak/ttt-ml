import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from itertools import count
from C4State import C4State
import typing
from tensorflow import keras
from Agent import *
from tqdm import tqdm

model = typing.cast(
    Model, keras.models.load_model("az_models/small_it41.keras"))

agent = AZAgent(model, "model_it41", 500)

game = C4State()

while not game.is_terminal():
    print(game)
    agent_move = agent.get_action(game)
    print(agent_move)
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