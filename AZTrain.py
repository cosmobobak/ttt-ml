import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import typing
import tensorflow.keras as keras
from az_method.MonteCarloTreeSearch import MCTS
from az_method.ReinfLearn import ReinfLearn
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Model

model: "Model" = typing.cast(
    Model, keras.models.load_model("az_models/random_model.keras"))

mcts_seacher = MCTS(model)

learner = ReinfLearn(model)

for training_run in range(0, 100):
    print(f"Training run {training_run}")

    all_pos = []
    all_move_probs = []
    all_values = []

    for j in tqdm(range(0, 500)):
        pos, move_probs, values = learner.play_game()

        all_pos += pos
        all_move_probs += move_probs
        all_values += values

    np_pos = np.array(all_pos)
    np_probs = np.array(all_move_probs)
    np_values = np.array(all_values)

    # print(f"num. x: {len(np_pos)}")
    # print(f"num. y1: {len(np_probs)}")
    # print(f"num. y2: {len(np_values)}")

    # print(np_pos[0])
    # print(np_probs[0])
    # print(np_values[0])

    model.fit(
        x=np_pos, 
        y=[np_probs, np_values], 
        epochs=256, 
        batch_size=16
    )

    if training_run % 5 == 0:
        model.save(f"az_models/model_it{training_run}.keras")
