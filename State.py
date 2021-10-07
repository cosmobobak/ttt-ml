import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from random import choice
from typing import List
import numpy as np


from tensorflow.python.keras.engine.training import Model

class State:
    X, O = 1, -1

    def __init__(self, input_node=None) -> None:
        if input_node is not None:
            self.node = input_node
        else:
            self.node: np.ndarray = np.array(
                [
                    np.zeros(9, dtype=int),
                    np.zeros(9, dtype=int)
                ]
            )
        self.move_count = self.node.sum()
        self.stack = []

    def __eq__(self, other: "State") -> bool:
        return (self.node == other.node).all()

    def __hash__(self) -> int:
        return hash(self.node.tostring())

    def reset(self) -> None:
        self.move_count = 0
        self.stack = []
        self.node: np.ndarray = np.zeros((2, 9))

    def get_turn(self) -> int:
        return self.O if (self.move_count & 1) else self.X

    def get_turn_as_str(self) -> str:
        return 'O' if (self.move_count & 1) else 'X'

    def get_move_counter(self) -> int:
        return self.move_count

    def pos_filled(self, i) -> bool:
        return self.node[0][i] != 0 or self.node[1][i] != 0

    # only valid to use if self.pos_filled() returns True:
    def player_at(self, i) -> bool:
        return self.node[0][i] != 0

    def probe_spot(self, i: int) -> bool:
        # tests the bit of the most recently played side
        return self.node[(self.move_count + 1) & 1][i] == 1

    def is_full(self) -> bool:
        return all((self.pos_filled(i) for i in range(9)))

    def symbol(self, xy: "tuple[int, int]") -> str:
        x, y = xy
        return ('X' if self.player_at(x * 3 + y) else 'O') if self.pos_filled(x * 3 + y) else '.'

    def __repr__(self) -> str:
        gs = lambda x: self.symbol(x)
        pairs = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)]
        ]
        return '\n'.join([' '.join(map(gs, pairline)) for pairline in pairs])

    def play(self, i) -> None:
        self.node[self.move_count & 1][i] = 1
        self.move_count += 1
        self.stack.append(i)

    def unplay(self) -> None:
        assert self.move_count > 0
        i = self.stack.pop()
        self.move_count -= 1
        self.node[self.move_count & 1][i] = 0

    def push(self, i) -> None:
        self.play(i)

    def push_ret(self, i) -> "State":
        self.push(i)
        return self

    def pop(self) -> None:
        self.unplay()

    def evaluate(self) -> int:
        # check first diagonal
        if (self.probe_spot(0) and self.probe_spot(4) and self.probe_spot(8)):
            return -self.get_turn()
        
        # check second diagonal
        if (self.probe_spot(2) and self.probe_spot(4) and self.probe_spot(6)):
            return -self.get_turn()
        
        # check rows
        for i in range(3):
            if (self.probe_spot(i * 3) and self.probe_spot(i * 3 + 1) and self.probe_spot(i * 3 + 2)):
                return -self.get_turn()
            
        # check columns
        for i in range(3):
            if (self.probe_spot(i) and self.probe_spot(i + 3) and self.probe_spot(i + 6)):
                return -self.get_turn()
            
        return 0

    def is_terminal(self) -> bool:
        return self.is_full() or (self.evaluate() != 0)

    def legal_moves(self) -> "list[int]":
        return [m for m in range(9) if not self.pos_filled(m)]

    def children(self) -> "list[State]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append(self.clone())
            self.unplay()

        return cs

    def random_play(self) -> None:
        self.play(choice(self.legal_moves()))

    def vectorise(self) -> np.ndarray:
        return np.reshape(self.node.copy(), (2, 3, 3))

    def flatten(self) -> np.ndarray:
        return np.reshape(self.node.copy(), (18))

    def clone(self) -> "State":
        return State(self.node.copy())


def perft(state: State, ss: "set[State]"):
    ss.add(state.clone())
    if state.is_terminal():
        return
    for move in state.legal_moves():
        state.push(move)
        perft(state, ss)
        state.pop()

FIRST_9_STATES: "list[State]" = [
    State().push_ret(0),
    State().push_ret(1),
    State().push_ret(2),
    State().push_ret(3),
    State().push_ret(4),
    State().push_ret(5),
    State().push_ret(6),
    State().push_ret(7),
    State().push_ret(8)
]
