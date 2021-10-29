from itertools import count
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import typing

from tensorflow import keras
from Agent import *
from tqdm import tqdm

sup_model = typing.cast(
    Model, keras.models.load_model("supervised_models/supervised_model.h5"))
random_model = typing.cast(
    Model, keras.models.load_model("az_models/random_model.keras"))
az_model0 = typing.cast(
    Model, keras.models.load_model("az_models/model_it0.keras"))

AGENTS = [
    RandomAgent(), 
    MinimaxAgent(), 
    SupervisedAgent(sup_model),
    RawAZAgent(random_model, "random_model"),
    AZAgent(random_model, "random_model"),
    RawAZAgent(az_model0, "model0"), 
    AZAgent(az_model0, "model0"),
]

all_pairs = [(a1, a2) for a1 in AGENTS for a2 in AGENTS if a1 != a2]
matchups = []
for a1, a2 in all_pairs:
    if (a2, a1) not in matchups:
        matchups.append((a1, a2))

for a1, a2 in matchups:
    print(f"{a1} vs {a2}")
    results = [ play_game(a1, a2, i & 1 == 0) for i in tqdm(range(100)) ]
    x_wins = sum(1 for result in results if result == 1)
    o_wins = sum(1 for result in results if result == -1)
    draws =  sum(1 for result in results if result == 0)
    print(f"{a1} wins {x_wins} times, {a2} wins {o_wins} times, and there are {draws} draws.")
