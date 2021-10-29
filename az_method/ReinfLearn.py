import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from az_method.MonteCarloTreeSearch import MCTS
from az_method.NodeEdge import Edge, Node
from State import State
from tensorflow.keras.models import Model

class ReinfLearn:
    def __init__(self, model: "Model") -> None:
        self.model = model
    
    def play_game(self) -> "tuple[list[np.ndarray], list[np.ndarray], list[float]]":
        positions_data: "list[np.ndarray]" = []
        move_probs_data: "list[np.ndarray]" = []
        values_data: "list[float]" = []

        g = State()
        g.set_starting_position()

        while not g.is_terminal():
            positions_data.append(g.vectorise_chlast()) 

            root_edge = Edge(None, None)
            root_edge.N = 1
            root_node = Node(g, root_edge)
            mcts_seacher = MCTS(self.model)

            move_probs = mcts_seacher.search(root_node)
            output_vec = np.zeros(State.ACTION_SPACE_SIZE)

            for move, prob, _, _ in move_probs:
                move_idx = move
                output_vec[move_idx] = prob

            rand_idx = np.random.multinomial(1, output_vec)
            idx = np.where(rand_idx == 1)[0][0]
            next_move = None

            for move, _, _, _ in move_probs:
                move_idx = move
                if idx == move_idx:
                    next_move = move
            move_probs_data.append(output_vec)
            g.push(next_move)

        winner = g.evaluate()
        for _ in move_probs_data:
            if winner == State.X:
                values_data.append(1.0)
            elif winner == State.O:
                values_data.append(-1.0)
            else:
                values_data.append(0.0)
        return positions_data, move_probs_data, values_data


