from random import choice
from typing import List
import numpy as np
from tensorflow.python.keras.engine.training import Model

class State:
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
        self.move_count = 0
        self.stack = []

    def __eq__(self, other) -> bool:
        return self.node[0] == other.node[0] and self.node[1] == other.node[1]

    def reset(self) -> None:
        self.move_count = 0
        self.stack = []
        self.node: np.ndarray = np.zeros((2, 9))

    def get_turn(self) -> int:
        return -1 if (self.move_count & 1) else 1

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

    def is_game_over(self) -> bool:
        return self.is_full() or (self.evaluate() != 0)

    def legal_moves(self) -> "list[int]":
        return [m for m in range(9) if not self.pos_filled(m)]

    def children(self) -> "list[State]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append(self.vectorise().copy())
            self.unplay()

        return cs

    def random_play(self) -> None:
        self.play(choice(self.legal_moves()))

    def vectorise(self) -> np.ndarray:
        return np.reshape(self.node, (2, 3, 3))

    def flatten(self) -> np.ndarray:
        return np.reshape(self.node, (18))
