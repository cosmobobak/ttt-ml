from State import State
import numpy as np
from random import random, choice
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model

PLAYER_TWO = -1

class Agent:
    def __init__(self, currentModel, side) -> None:
        self.model: Model = currentModel
        self.side = side
        self.epsilon = 0.1

    def takeBestAction(self, state) -> None:
        # evaluate all future states (generate them and model.__call__())
        evals: np.ndarray = self.model(np.array([child for child in state.children()])).numpy()

        if self.side == PLAYER_TWO: 
            evals = -evals # if we are player 2, we are looking to minimise score instead.

        chosen_move = state.legal_moves()[np.argmax(evals)]
        state.play(chosen_move)

    def takeRandomAction(self, state: State) -> None:
        state.play(choice(state.legal_moves()))

    def takeLearningAction(self, state: State) -> None:
        # generate random value between 0 and 1
        # if value > epsilon, take argmax, else move randomly
        
        if random() > self.epsilon:
            self.takeBestAction(state)
        else:
            self.takeRandomAction(state)
