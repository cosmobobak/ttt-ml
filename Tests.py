from State import State
from Oracle import oracle_value
import numpy as np
import ValidationData

def test_move_make():
    board = State()
    board.push(4)
    expected = np.zeros((2, 3, 3), dtype=int)
    expected[0, 1, 1] = 1

    assert np.array_equal(board.vectorise_chlast(), expected), f"Error: {board.vectorise_chlast()} != {expected}"


def test_make_unmake():
    board = State()
    moves = [0, 1, 2, 3, 4]
    for move in moves:
        board.push(move)
        board.pop()

    for move in moves:
        board.push(move)

    for move in moves:
        board.pop()

    expected = np.zeros((2, 3, 3), dtype=int)

    assert np.array_equal(board.vectorise_chlast(), expected), f"Error: {board.vectorise_chlast()} != {expected}"

def test_oracle():
    board = State()

    if not oracle_value(board) == 0:
        print("Error: value(b) != 0")
        print(f"{oracle_value(board) = }")
        print(board)
        exit(1)

    board.play(0)
    board.play(1)
    if not oracle_value(board) == 1:
        print("Error: value(b) != 1")
        print(f"{oracle_value(board) = }")
        print(board)
        exit(1)

    board.play(2)
    board.play(4)
    board.play(3)

    if not oracle_value(board) == -1:
        print("Error: value(b) != -1")
        print(f"{oracle_value(board) = }")
        print(board)
        exit(1)

def more_oracle_tests():
    states = map(State, ValidationData.x_val)
    values = ValidationData.y_val
    assert all(v == oracle_value(s) for v, s in zip(values, states))

if __name__ == "__main__":
    test_move_make()
    test_make_unmake()
    test_oracle()
