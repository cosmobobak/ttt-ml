import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from itertools import count
from C4State import C4State
import typing
from tensorflow import keras
from Agent import *
from ModelTools import twohead_evaluate
import time

model = typing.cast(
    Model, keras.models.load_model("az_models/random_model.keras"))
print("Loaded model!")

agent10 = AZAgent(model, "model_6", 10)
agent20 = AZAgent(model, "model_6", 20)
agent30 = AZAgent(model, "model_6", 30)
agent100 = AZAgent(model, "model_6", 100)
agent200 = AZAgent(model, "model_6", 200)
agent300 = AZAgent(model, "model_6", 300)

game = C4State()

# benchmark search

start = time.time()
_ = agent10.get_next_state(game)
end = time.time()
print(f"10-rollouts: {round(end - start, 2)}, rollouts/s: {round(10/(end - start), 2)}")

start = time.time()
_ = agent20.get_next_state(game)
end = time.time()
print(f"20-rollouts: {round(end - start, 2)}, rollouts/s: {round(20/(end - start), 2)}")

start = time.time()
_ = agent30.get_next_state(game)
end = time.time()
print(f"30-rollouts: {round(end - start, 2)}, rollouts/s: {round(30/(end - start), 2)}")

start = time.time()
_ = agent100.get_next_state(game)
end = time.time()
print(f"100-rollouts: {round(end - start, 2)}, rollouts/s: {round(100/(end - start), 2)}")

start = time.time()
_ = agent200.get_next_state(game)
end = time.time()
print(f"200-rollouts: {round(end - start, 2)}, rollouts/s: {round(200/(end - start), 2)}")

start = time.time()
_ = agent300.get_next_state(game)
end = time.time()
print(f"300-rollouts: {round(end - start, 2)}, rollouts/s: {round(300/(end - start), 2)}")



