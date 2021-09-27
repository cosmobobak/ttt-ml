from State import State
import numpy as np
from random import random, choice
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model

PLAYER_TWO = -1

class Agent:
    def __init__(self, input_model, side) -> None:
        self.model: Model = input_model
        self.side = side
        self.epsilon = 0.1

    def take_best_action(self, state) -> None:
        # evaluate all future states (generate them and model.__call__())
        evals: np.ndarray = self.model(np.array([child for child in state.children()])).numpy()

        if self.side == PLAYER_TWO: 
            evals = -evals # if we are player 2, we are looking to minimise score instead.

        chosen_move = state.legal_moves()[np.argmax(evals)]
        state.play(chosen_move)

    def take_random_action(self, state: State) -> None:
        state.play(choice(state.legal_moves()))

    def take_epsilon_action(self, state: State) -> None:
        # generate random value between 0 and 1
        # if value > epsilon, take argmax, else move randomly
        
        if random() > self.epsilon:
            self.take_best_action(state)
        else:
            self.take_random_action(state)
