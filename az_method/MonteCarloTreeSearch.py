import math
import random
from az_method.NodeEdge import Node, Edge
from C4State import C4State
from tqdm import tqdm

class MCTS:
    def __init__(self, network) -> None:
        self.network = network
        self.root_node = None
        self.tau = 1.0
        self.c_puct = 1.0

    def uct_value(self, edge: Edge, parent_n: int) -> float:
        return self.c_puct * edge.P * (math.sqrt(parent_n) / (1 + edge.N))

    def select(self, node: Node) -> Node:
        if node.is_leaf():
            return node
        else:
            max_uct_child = None
            max_uct_value = -100000000
            for edge, child_node in node.child_edge_node:
                uct_value = self.uct_value(edge, edge.parent_node.parent_edge.N)
                val = edge.Q
                if edge.parent_node.board.get_turn() == C4State.O:
                    val = -val
                uct_val_child = val + uct_value
                if uct_val_child > max_uct_value:
                    max_uct_child = child_node
                    max_uct_value = uct_val_child
            all_best_children = []
            for edge, child_node in node.child_edge_node:
                uct_value = self.uct_value(edge, edge.parent_node.parent_edge.N)
                val = edge.Q
                if edge.parent_node.board.get_turn() == C4State.O:
                    val = -val
                uct_val_child = val + uct_value
                if uct_val_child == max_uct_value:
                    all_best_children.append(child_node)
            if max_uct_child is None:
                raise ValueError("could not identify best child")
            else:
                if len(all_best_children) > 1:
                    idx = random.randint(0, len(all_best_children) - 1)
                    return self.select(all_best_children[idx])
                else:
                    return self.select(max_uct_child)

    def expand_and_evaluate(self, node: Node) -> None:
        terminal = node.board.is_terminal()
        winner = node.board.evaluate()
        if terminal:
            v = 0.0
            if winner == C4State.X:
                v = 1.0
            elif winner == C4State.O:
                v = -1.0
            self.backpropagate(v, node.parent_edge)
            return
        v = node.expand(self.network)
        self.backpropagate(v, node.parent_edge)
        
    def backpropagate(self, v: float, edge: Edge) -> None:
        edge.N += 1
        edge.W = edge.W + v
        edge.Q = edge.W / edge.N
        if edge.parent_node is not None:
            if edge.parent_node.parent_edge is not None:
                self.backpropagate(v, edge.parent_node.parent_edge)

    def search(self, root_node: Node, sims: int = 50) -> "list[tuple[int, float, int, float]]":
        self.root_node = root_node
        self.root_node.expand(self.network)
        for _ in range(sims):
            selected_node = self.select(root_node)
            self.expand_and_evaluate(selected_node)
        N_sum: int = 0
        move_probabilities: "list[tuple[int, float, int, float]]" = []
        for edge, _ in root_node.child_edge_node:
            N_sum += edge.N
        for edge, _ in root_node.child_edge_node:
            prob = (edge.N ** (1 / self.tau)) / (N_sum ** (1 / self.tau))
            move_probabilities.append((edge.move, prob, edge.N, edge.Q))
        return move_probabilities
