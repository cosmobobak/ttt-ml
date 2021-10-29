import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from State import FIRST_9_STATES, State
from NetMaker import MultilayerPerceptron
from Oracle import oracle_value
import numpy as np
from tensorflow.keras.models import Model
from az_method.MonteCarloTreeSearch import MCTS
from az_method.NodeEdge import Edge, Node
import copy

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
    predictions = m(np.array([s.vectorise_chlast() for s in ss]))
    return [p[0] for p in predictions]

def evaluate_states_with_twohead(ss: "list[State]", m: "Model") -> "list[float]":
    """
    Evaluates a list of states with a two-head model and returns a list of the values.
    """
    predictions = m(np.array([s.vectorise_chlast() for s in ss]))
    return [p for p in predictions[1]]

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

def best_state_given_twohead(ss: "list[State]", m: "Model") -> "State":
    """
    Returns the best state from a list of states given a two-head model.
    """
    assert len(ss) > 0
    assert all(s.get_turn() == -1 for s in ss) or all(s.get_turn() == 1 for s in ss), f"All states must be of the same turn. state turns: {[s.get_turn() for s in ss]}"
    turn = ss[0].get_turn()
    values = evaluate_states_with_twohead(ss, m)

    is_x_to_move = turn == State.O # comparing against O as these are child positions.

    return ss[np.argmax(values)] if is_x_to_move else ss[np.argmin(values)]

def model_evaluate(s: State, m: "Model") -> float:
    """
    Evaluates a single state with a model and returns the value.
    """
    return m(np.array([s.vectorise_chlast()]))[0][0]

def twohead_evaluate(s: State, m: "Model") -> float:
    """
    Evaluates a single state with a two-head model and returns the value.
    """
    return m(np.array([s.vectorise_chlast()]))[1][0][0]

def twohead_policy(s: State, m: "Model") -> np.ndarray:
    """
    Returns the policy of a single state with a two-head model.
    """
    return np.argmax(m(np.array([s.vectorise_chlast()]))[0])

def twohead_new_state(s: State, m: "Model") -> State:
    """
    Returns the new state of a single state with a two-head model.
    """
    out = copy.deepcopy(s)
    out.push(twohead_policy(s, m))
    return out

def mcts_new_state(s: State, m: "Model") -> State:
    """
    Returns the new state of a single state with a MCTS model.
    """
    out = copy.deepcopy(s)
    agent = MCTS(m)
    r_edge = Edge(None, None)
    r_node = Node(out, r_edge)
    probs = agent.search(r_node, 100)
    move = max(probs, key=lambda x: x[1])[0]
    out.push(move)
    return out 

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
