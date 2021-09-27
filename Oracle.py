from State import State
from copy import copy


def oracle_value(state: State) -> float:
    mutator = copy(state)
    return negamax(mutator, a=float('-inf'), b=float('inf'), depth=10)


def negamax(state: State, a: float, b: float, depth: int) -> float:
    if depth == 0 or state.is_game_over():
        return state.evaluate() * state.get_turn()
    score = 0
    for move in state.legal_moves():
        state.play(move)
        score = -negamax(state, -b, -a, depth - 1)
        state.unplay()
        a = max(a, score)
        if a >= b:
            break
    return a
