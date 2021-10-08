import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from State import FIRST_9_STATES, State
from NetMaker import MultilayerPerceptron
from Oracle import oracle_value
import numpy as np
from tensorflow.keras.models import Model

def map_wdl_to_eval(wdl: np.ndarray) -> float:
    """
    Maps a WDL array to a value.
    """
    top = np.argmax(wdl)
    # 0, 1, 2 -> 1, 0, -1
    return (2 - top) - 1

def evaluate_states_with_model(ss: "list[State]", m: "Model") -> "list[float]":
    """
    Evaluates a list of states with a model and returns a list of the values.
    """
    predictions = m.predict(np.array([s.vectorise_chlast() for s in ss]))
    return [p[0] for p in predictions]

def best_state_given_model(ss: "list[State]", m: "Model") -> "State":
    """
    Returns the best state from a list of states given a model.
    """
    assert len(ss) > 0
    assert all(s.get_turn() == -1 for s in ss) or all(s.get_turn() == 1 for s in ss), f"All states must be of the same turn. state turns: {[s.get_turn() for s in ss]}"
    turn = ss[0].get_turn()
    values = evaluate_states_with_model(ss, m)

    is_x_to_move = turn == State.O # comparing against O as these are child positions.

    return ss[np.argmax(values)] if is_x_to_move else ss[np.argmin(values)]

def model_evaluate(s: State, m: "Model") -> float:
    """
    Evaluates a single state with a model and returns the value.
    """
    return m.predict(np.array([s.vectorise_chlast()]))[0][0]

if __name__ == "__main__":
    # Create a random untrained model
    model = MultilayerPerceptron().get_model()
    # Load some example data
    positions = FIRST_9_STATES

    # Test state eval
    for p in positions:
        print(p, f"{oracle_value(p)=}")
        print()
    print("\n".join(map(str, enumerate(evaluate_states_with_model(positions, model)))))
    print("\nchoice:")
    print(best_state_given_model(positions, model))
