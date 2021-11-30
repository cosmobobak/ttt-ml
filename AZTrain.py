import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import typing
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from az_method.MonteCarloTreeSearch import MCTS
from az_method.ReinfLearn import ReinfLearn
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Model

GAMES_PER_RUN = 1000
ROLLOUTS = 50

CURRENT_ITERATION = 4
# starting_model_path = "az_models/random_model.keras"
starting_model_path = f"az_models/model_{CURRENT_ITERATION}.keras"
CURRENT_ITERATION += 1

model: "Model" = typing.cast(
    Model, keras.models.load_model(starting_model_path))

mcts_seacher = MCTS(model)

learner = ReinfLearn(model)


for training_run in range(0, 100):
    print(f"Training run {CURRENT_ITERATION+training_run}")
    print(f"Running with {ROLLOUTS} rollouts per move, ")
    print(f"and {GAMES_PER_RUN} games per training run.")
    print(f"Starting model: {starting_model_path}")
    

    all_states = []
    all_actions = []
    all_values = []

    for j in tqdm(range(0, GAMES_PER_RUN)):
        states, actions, values = learner.play_game(ROLLOUTS)

        all_states += states
        all_actions += actions
        all_values += values

    np_states = np.array(all_states)
    np_actions = np.array(all_actions)
    np_values = np.array(all_values)

    model.fit(
        x=np_states, 
        y=[np_actions, np_values], 
        epochs=256, 
        batch_size=16
    )

    # if (CURRENT_ITERATION+training_run) % 5 == 0:
    model.save(f"az_models/model_{CURRENT_ITERATION+training_run}.keras")
