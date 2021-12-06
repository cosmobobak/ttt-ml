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
            if type(g) == C4State: # if we're training for Connect 4, we can use mirrored boards
                vect = g.vectorise_chlast()
                np.flip(vect, axis=1)
                positions_data.append(vect)

            root_edge = Edge(None, None)
            root_edge.N = 1
            root_node = Node(g, root_edge)
            mcts_seacher = MCTS(self.model)

            move_probs = mcts_seacher.search(root_node, sims=rollouts)
            output_vec = np.zeros(C4State.ACTION_SPACE_SIZE)

            for move, prob, _, _ in move_probs:
                move_idx = move
                output_vec[move_idx] = prob

            # add dirichlet noise so we don't always just pick the best move, even when it's [1,0,0,0]
            dirch_vec = add_dirichlet_noise(output_vec)
            # zero-out illegal moves and normalise
            learning_vec = np.zeros(C4State.ACTION_SPACE_SIZE)
            for move, prob, _, _ in move_probs:
                move_idx = move
                learning_vec[move_idx] = dirch_vec[move_idx]
            learning_vec = learning_vec / np.sum(learning_vec)
            # select a move
            rand_idx = np.random.multinomial(1, learning_vec)
            # convert the move to an index
            idx = np.where(rand_idx == 1)[0][0]
            next_move = None

            for move, _, _, _ in move_probs:
                move_idx = move
                if idx == move_idx:
                    next_move = move
                    break
            else:
                print("Error: couldn't find move")
                print(f"{output_vec = }")
                print(f"{dirch_vec = }")
                print(f"{rand_idx = }")
                print(f"{idx = }")
                print(f"{g.legal_moves() = }")
                print(f"{next_move = }")
                raise ValueError("No move found")
            
            move_probs_data.append(output_vec)
            if type(g) == C4State: # if we're training for Connect 4, we can use mirrored move probs
                vect = output_vec.copy()
                np.flip(vect, axis=0)
                move_probs_data.append(vect)
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

def add_dirichlet_noise(vec: "np.ndarray") -> "np.ndarray":
    """
    Add dirichlet noise to a vector
    """
    alpha = 1
    noise = np.random.dirichlet(np.ones(len(vec)) * alpha)
    return vec + noise