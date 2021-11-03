import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from itertools import count
from C4State import C4State
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
az_model5 = typing.cast(
    Model, keras.models.load_model("az_models/model_it5.keras"))
az_model10 = typing.cast(
    Model, keras.models.load_model("az_models/model_it10.keras"))
az_model15 = typing.cast(
    Model, keras.models.load_model("az_models/model_it15.keras"))
# az_model20 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it20.keras"))
# az_model25 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it25.keras"))
# az_model30 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it30.keras"))
# az_model35 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it35.keras"))
# az_model40 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it40.keras"))
# az_model45 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it45.keras"))
# az_model50 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it50.keras"))
# az_model55 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it55.keras"))
# az_model60 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it60.keras"))
# az_model65 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it65.keras"))
# az_model70 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it70.keras"))
# az_model71 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it71.keras"))
# az_model72 = typing.cast(
#     Model, keras.models.load_model("az_models/model_it72.keras"))

AGENTS = [
    RandomAgent(),
    RawAZAgent(random_model, "random"),
    RawAZAgent(az_model0, "az_model0"),
    RawAZAgent(az_model5, "az_model5"),
    RawAZAgent(az_model10, "az_model10"),
    RawAZAgent(az_model15, "az_model15"),
]

all_pairs = [(a1, a2) for a1 in AGENTS for a2 in AGENTS if a1 != a2]
matchups = []
for a1, a2 in all_pairs:
    if (a2, a1) not in matchups:
        matchups.append((a1, a2))

for a1, a2 in matchups:
    print(f"{a1} vs {a2}")
    results = [ play_game(a1, a2, i & 1 == 0, game=C4State) for i in tqdm(range(500)) ]
    x_wins = sum(1 for result in results if result == 1)
    o_wins = sum(1 for result in results if result == -1)
    draws =  sum(1 for result in results if result == 0)
    print(f"{a1} wins {x_wins} times, {a2} wins {o_wins} times, and there are {draws} draws.")
