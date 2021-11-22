import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from az_method.MonteCarloTreeSearch import MCTS
from az_method.NodeEdge import Edge, Node
from C4State import C4State
from tensorflow.keras.models import Model

class ReinfLearn:
    def __init__(self, model: "Model") -> None:
        self.model = model
    
    def play_game(self, rollouts: int) -> "tuple[list[np.ndarray], list[np.ndarray], list[float]]":
        positions_data: "list[np.ndarray]" = []
        move_probs_data: "list[np.ndarray]" = []
        values_data: "list[float]" = []

        g = C4State()
        g.set_starting_position()

        while not g.is_terminal():
            positions_data.append(g.vectorise_chlast()) 

            root_edge = Edge(None, None)
            root_edge.N = 1
            root_node = Node(g, root_edge)
            mcts_seacher = MCTS(self.model)

            move_probs = mcts_seacher.search(root_node, sims=rollouts)
            output_vec = np.zeros(C4State.ACTION_SPACE_SIZE)

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
            if winner == C4State.X:
                values_data.append(1.0)
            elif winner == C4State.O:
                values_data.append(-1.0)
            else:
                values_data.append(0.0)
        return positions_data, move_probs_data, values_data


