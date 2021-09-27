from random import choice
from typing import List
import numpy as np
from tensorflow.python.keras.engine.training import Model

class State:
    def __init__(self, input_node=None):
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

    def __eq__(self, other):
        return self.node[0] == other.node[0] and self.node[1] == other.node[1]

    def reset(self):
        self.move_count = 0
        self.stack = []
        self.node: np.ndarray = np.array(
            [
                np.zeros(9, dtype=int), 
                np.zeros(9, dtype=int)
            ]
        )

    def get_turn(self) -> int:
        return -1 if (self.move_count % 2) else 1

    def get_move_counter(self) -> int:
        return self.move_count

    def pos_filled(self, i):
        return self.node[0][i] != 0 or self.node[1][i] != 0

    # only valid to use if self.pos_filled() returns True:
    def player_at(self, i):
        return self.node[0][i] != 0

    def probe_spot(self, i: int) -> bool:
        # tests the bit of the most recently played side
        return self.node[(self.move_count + 1) % 2][i] == 1

    def is_full(self):
        return all((self.pos_filled(i) for i in range(9)))

    def __repr__(self):
        # print(self.node)
        builder = ""
        for x in range(3):
            for y in range(3):
                if (self.pos_filled(x * 3 + y)):
                    if (self.player_at(x * 3 + y)):
                        builder += "X "
                    else:
                        builder += "0 "
                else:
                    builder += ". "
            builder += '\n'
        builder += '\n'
        return builder

    def play(self, i):
        self.node[self.move_count % 2][i] = 1
        self.move_count += 1
        self.stack.append(i)

    def unplay(self):
        assert self.move_count > 0
        i = self.stack.pop()
        self.move_count -= 1
        self.node[self.move_count % 2][i] = 0

    def push(self, i):
        self.play(i)

    def pop(self):
        self.unplay()

    def evaluate(self):
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

    def is_game_over(self):
        return self.is_full() or (self.evaluate() != 0)

    def legal_moves(self):
        return [m for m in range(9) if not self.pos_filled(m)]

    def children(self):
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append(self.vectorise().copy())
            self.unplay()

        return cs

    def random_play(self):
        self.play(choice(self.legal_moves()))

    def vectorise(self):
        out = np.zeros((2, 3, 3), dtype=int)
        for idx in range(18):
            side = idx // 9
            sqidx = idx - (side * 9)
            row = sqidx // 3
            col = sqidx % 3
            out[side][row][col] = self.node[side][sqidx]

        return out

