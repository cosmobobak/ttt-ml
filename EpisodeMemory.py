from typing import List, Tuple
import numpy as np

class EpisodeMemory:
    def __init__(self) -> None:
        self.states_encountered: List[np.ndarray] = []
        self.evaluations: List[float] = []
        self.results: List[int] = []

    def push(self, state: np.ndarray, eval: float) -> None:
        self.states_encountered.append(state)
        self.evaluations.append(eval)

    def push_x(self, state: np.ndarray) -> None:
        self.states_encountered.append(state)

    def push_y(self, eval: float) -> None:
        self.evaluations.append(eval)

    def push_result(self, res: int) -> None:
        self.results.append(res)

    def get_wdl(self):
        l = len(self.results)
        return self.results.count(1) / l, self.results.count(0) / l, self.results.count(-1) / l

    def get_xy_train(self) -> Tuple[np.ndarray, np.ndarray]:
        # out_x = np.array(self.states_encountered[:-1])
        # out_y = np.array(self.evaluations[1:])
        # print(f"{out_y=}")
        # return out_x, out_y
        return np.array(self.states_encountered), np.array(self.evaluations)
