from tensorflow.python.keras.engine.training import Model
import numpy as np
from random import choice
from Hyperparameters import DEBUG
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class C4State:
    X, O = 1, -1
    ACTION_SPACE_SIZE = 7

    def __init__(self, input_node=None, heights=None) -> None:
        if input_node is not None:
            self.node = input_node
        else:
            self.node: np.ndarray = np.zeros((6, 7, 2))
        self.move_count = int(self.node.sum())
        self.stack = []
        if heights is not None:
            self.heights = [h for h in heights]
        else:
            self.heights = [0 for _ in range(7)]

    def __eq__(self, other: "C4State") -> bool:
        return (self.node == other.node).all()

    def __hash__(self) -> int:
        return hash(self.node.tostring())

    def reset(self) -> None:
        self.node: np.ndarray = np.zeros((6, 7, 2))
        self.move_count = 0
        self.stack = []
        self.heights = [0 for _ in range(7)]

    def set_starting_position(self) -> None:
        self.reset()

    def get_turn(self) -> int:
        return self.O if (self.move_count & 1) else self.X

    def get_turn_as_str(self) -> str:
        return 'O' if (self.move_count & 1) else 'X'

    def get_move_counter(self) -> int:
        return self.move_count

    def pos_filled(self, row: int, col: int) -> bool:
        return self.node[row][col][0] != 0 or self.node[row][col][0] != 0

    # only valid to use if self.pos_filled() returns True:
    def player_at(self, row: int, col: int) -> bool:
        return self.node[row][col][0] != 0

    def probe_spot(self, row: int, col: int) -> bool:
        # tests the bit of the most recently played side
        return self.node[row][col][(self.move_count + 1) & 1] == 1

    def is_full(self) -> bool:
        for col in range(7):
            if not self.heights[col] == 6:
                return False
        return True

    def __repr__(self) -> str:
        sym = lambda r, c: 'X' if self.node[r][c][0] == 1 else ('O' if self.node[r][c][1] == 1 else '.')
        return "\n".join((" ".join(sym(row, col) for col in range(7))) for row in reversed(range(6)))

    def play(self, col) -> None:
        row = self.heights[col]
        self.node[row][col][self.move_count & 1] = 1
        self.heights[col] += 1
        self.move_count += 1
        self.stack.append(col)

    def unplay(self) -> None:
        assert self.move_count > 0
        self.move_count -= 1
        col = self.stack.pop()
        self.heights[col] -= 1
        row = self.heights[col]
        self.node[row][col][self.move_count & 1] = 0

    def push(self, i) -> None:
        assert i in self.legal_moves(), f"illegal move: {i}"
        self.play(i)

    def push_ret(self, i) -> "C4State":
        self.push(i)
        return self

    def pop(self) -> None:
        self.unplay()

    def evaluate(self) -> int:
        if len(self.stack) == 0:
            return 0
        last_col = self.stack[-1]
        last_row = self.heights[last_col] - 1
        for direction in range(4):
            if self.check_direction(last_row, last_col, direction):
                return - self.get_turn()

        return 0

    def check_direction(self, row: int, col: int, direction: int) -> bool:
        row_origin = row
        col_origin = col
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for _ in range(4):
            if row < 0 or row >= 6 or col < 0 or col >= 7:
                break
            if self.probe_spot(row, col):
                count += 1
                row += directions[direction][0]
                col += directions[direction][1]
            else:
                break
        row = row_origin - directions[direction][0]
        col = col_origin - directions[direction][1]
        for _ in range(3):
            if row < 0 or row >= 6 or col < 0 or col >= 7:
                break
            if self.probe_spot(row, col):
                count += 1
                row -= directions[direction][0]
                col -= directions[direction][1]
            else:
                break
        return count >= 4 

    def is_terminal(self) -> bool:
        return self.is_full() or (self.evaluate() != 0)

    def legal_moves(self) -> "list[int]":
        return [i for i in range(7) if not self.heights[i] == 6]

    def children(self) -> "list[C4State]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append(self.clone())
            self.unplay()

        return cs

    def state_action_pairs(self) -> "list[tuple[C4State, int]]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append((self.clone(), move))
            self.unplay()

        return cs

    def random_play(self) -> None:
        self.play(choice(self.legal_moves()))

    def vectorise_chlast(self) -> np.ndarray:
        out = self.node.copy()
        return out

    def flatten(self) -> np.ndarray:
        return np.reshape(self.node.copy(), (18))

    def clone(self) -> "C4State":
        return C4State(self.node.copy(), heights=self.heights)

    @classmethod
    def _perft(cls, state: "C4State", ss: "set[C4State]"):
        if state in ss:
            return
        ss.add(state.clone())
        if state.is_terminal():
            return
        for move in state.legal_moves():
            state.push(move)
            cls._perft(state, ss)
            state.pop()

    @classmethod
    def get_every_state(cls) -> "set[C4State]":
        ss = set()
        state = C4State()
        cls._perft(state, ss)
        return ss

    @classmethod
    def state_space(cls) -> int:
        return len(cls.get_every_state())


FIRST_7_STATES: "list[C4State]" = [
    C4State().push_ret(0),
    C4State().push_ret(1),
    C4State().push_ret(2),
    C4State().push_ret(3),
    C4State().push_ret(4),
    C4State().push_ret(5),
    C4State().push_ret(6),
]
