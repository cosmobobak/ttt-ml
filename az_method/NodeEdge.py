import copy
import numpy as np
from State import State

class Edge:
    def __init__(self, move: "int | None", parent_node: "Node | None") -> None:
        self.parent_node = parent_node
        self.move = move
        self.N: int = 0
        self.W: float = 0
        self.Q: float = 0
        self.P: float = 0

class Node:
    def __init__(self, board: "State", parent_edge: "Edge") -> None:
        self.board = board
        self.parent_edge = parent_edge
        self.child_edge_node: "list[tuple[Edge, Node]]" = []

    def expand(self, network) -> float:
        moves = self.board.legal_moves()
        for m in moves:
            child_board = copy.deepcopy(self.board)
            child_board.push(m)
            child_edge = Edge(m, self)
            child_node = Node(child_board, child_edge)
            self.child_edge_node.append((child_edge, child_node))
        q = network(np.array([self.board.vectorise_chlast()]))
        ps = q[0][0]
        prob_sum = 0.0
        for edge, _ in self.child_edge_node:
            m_idx = edge.move # directly works as an index
            edge.P = ps[m_idx]
            prob_sum += edge.P
        for edge, _ in self.child_edge_node:
            edge.P /= prob_sum
        v = q[1][0][0]
        return v

    def is_leaf(self) -> bool:
        return len(self.child_edge_node) == 0
