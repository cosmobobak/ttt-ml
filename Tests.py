from State import State
import numpy as np

def test_move_make():
    board = State()
    board.push(4)
    expected = np.array([
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=int),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int),
    ])

    assert np.array_equal(board.vectorise(), expected)


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

    expected = np.array(
        [
            np.zeros(9, dtype=int),
            np.zeros(9, dtype=int)
        ]
    )

    assert np.array_equal(board.vectorise(), expected)

if __name__ == "__main__":
    test_move_make()
    test_make_unmake()