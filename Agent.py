from C4State import C4State
from Oracle import perspective_value
from State import State
import numpy as np
from random import random, choice
import os
import copy
from ModelTools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model

PLAYER_TWO = -1

class RandomAgent:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "RandomAgent"

    def get_action(self, state: State) -> int:
        return choice(state.legal_moves())

    def get_next_state(self, state: State) -> State:
        out = state.clone()
        out.push(self.get_action(state))
        return out

class MinimaxAgent:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "MinimaxAgent"

    def get_action(self, state: State) -> int:
        if type(state) == C4State:
            return 0
        sa_pairs = state.state_action_pairs()
        values = [oracle_value(sa_pair[0]) for sa_pair in sa_pairs]
        if state.get_turn() == State.X:
            index = np.argmax(values)
        else:
            index = np.argmin(values)
        return sa_pairs[index][1]

    def get_next_state(self, state: State) -> State:
        out = state.clone()
        out.push(self.get_action(state))
        return out

class SupervisedAgent:
    def __init__(self, model: Model) -> None:
        self.model = model

    def __repr__(self) -> str:
        return "SupervisedAgent"

    def get_action(self, state: State) -> int:
        sa_pairs = state.state_action_pairs()
        best_state = best_state_given_model([s[0] for s in sa_pairs], self.model)
        return max(sa_pairs, key=lambda x: x[0] == best_state)[1]

    def get_next_state(self, state: State) -> State:
        return best_state_given_model(state.children(), self.model)

class RawAZAgent:
    def __init__(self, model: Model, model_name: str) -> None:
        self.model = model
        self.model_name = model_name

    def __repr__(self) -> str:
        return f"RawAZAgent-{self.model_name}"

    def get_action(self, state: State) -> int:
        return twohead_policy(state, self.model)

    def get_next_state(self, state: State) -> State:
        out = state.clone()
        move = self.get_action(state)
        out.push(move)
        assert out != state, f"RawAZAgent: get_next_state: state is not changed, move: {move}, state: {state}"
        return out


class AZAgent:
    def __init__(self, model: Model, model_name: str, rollouts: int) -> None:
        self.model = model
        self.model_name = model_name
        self.rollouts = rollouts

    def __repr__(self) -> str:
        return f"AZAgent-{self.model_name} ({self.rollouts} rollouts/move)"

    def get_action(self, state: State) -> int:
        return mcts_policy(state, self.model, self.rollouts)

    def get_next_state(self, state: State) -> State:
        return mcts_new_state(state, self.model, self.rollouts)

def play_game(agent1, agent2, start=True, game=State) -> int:
    state = game()
    turn_target = State.X if start else State.O
    while not state.is_terminal():
        # print(state)
        if state.get_turn() == turn_target:
            state = agent1.get_next_state(state)
        else:
            state = agent2.get_next_state(state)
    # print(state)
    return state.evaluate() if turn_target == State.X else -state.evaluate()
