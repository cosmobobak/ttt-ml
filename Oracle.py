from State import State

def oracle_value(state: State) -> float:
    mutator = state.clone()
    return negamax(mutator, a=float('-inf'), b=float('inf'), depth=10) * state.get_turn()

def perspective_value(state: State) -> float:
    mutator = state.clone()
    return negamax(mutator, a=float('-inf'), b=float('inf'), depth=10)

def negamax(state: State, a: float, b: float, depth: int) -> float:
    if depth == 0 or state.is_terminal():
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

if __name__ == '__main__':
    state = State()
    assert oracle_value(state) == 0, "Incorrect root eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(0)
    state.play(1)
    assert oracle_value(state) == 1, "Incorrect forced X win eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(4)
    state.play(8)
    state.play(6)
    state.play(2)
    assert oracle_value(state) == 1, "Incorrect immediate X win eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(3)
    assert oracle_value(state) == 1, "Incorrect gameover X win eval."
    print(state)
    print(f"{oracle_value(state)=}")

    state = State()
    state.play(1)
    state.play(4)
    state.play(7)
    assert oracle_value(state) == -1, "Incorrect forced O win eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(2)
    state.play(6)
    state.play(8)
    assert oracle_value(state) == -1, "Incorrect forced O win eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(0)
    assert oracle_value(state) == -1, "Incorrect immediate O win eval."
    print(state)
    print(f"{oracle_value(state)=}")
    state.play(5)
    assert oracle_value(state) == -1, "Incorrect gameover O win eval."
    print(state)
    print(f"{oracle_value(state)=}")
