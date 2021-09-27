from State import State
from Oracle import oracle_value
import numpy as np

def test_move_make():
    board = State()
    board.push(4)
    expected = np.zeros((2, 3, 3), dtype=int)
    expected[0, 1, 1] = 1

    assert np.array_equal(board.vectorise(), expected), f"Error: {board.vectorise()} != {expected}"


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

    assert np.array_equal(board.vectorise(), expected), f"Error: {board.vectorise()} != {expected}"

def test_oracle():
    if not oracle_value(State()) == 0:
        print("Error: value(State()) != 0")
        print(f"{oracle_value(State()) = }")
        print(State())
        exit(1)

    b = State()
    b.play(0)
    b.play(1)
    if not oracle_value(b) == 1:
        print("Error: value(b) != 1")
        print(f"{oracle_value(b) = }")
        print(b)
        exit(1)

    b.play(2)
    b.play(4)
    b.play(3)

    if not oracle_value(b) == -1:
        print("Error: value(b) != -1")
        print(f"{oracle_value(b) = }")
        print(b)
        exit(1)

if __name__ == "__main__":
    test_move_make()
    test_make_unmake()
