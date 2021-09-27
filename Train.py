import os
from time import sleep, time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from Agent import Agent
from NetMaker import BATCH_SIZE, MLPMaker
from EpisodeMemory import EpisodeMemory
from State import State
from ValidationData import x_val, y_val
from random import choices
from progress import bar
import tensorflow as tf
import numpy as np

NUM_SIMULATIONS = int(input("Self-play games per training step: "))
NUM_TRAINING_STEPS = int(input("Training steps: "))

MODEL = MLPMaker()()
try:
    MODEL.load_weights("td_agent_for_ttt")
except Exception:
    print("starting fresh!")


def evaluate_state(s: State) -> float:
    if s.is_game_over():
        return float(s.evaluate())
    else:
        return float(MODEL.predict(np.array([s.vectorise()]))[0][0])

def run_game(memory: EpisodeMemory):
    # create a fresh game, and have the agent play against itself
    # use a sampling policy (epsilon-greedy)
    # training data = all the games
    # x_train = states from state 0 -> state n-1
    # y_train = predictions for all states from state 1 -> state n - 1 AND the actual evaluation of state n
    # this way the terminal states get included too! (that's the training signal ig)

    a1 = Agent(currentModel=MODEL, side=1)
    a2 = Agent(currentModel=MODEL, side=-1)

    game = State()

    turn = 0
    while not game.is_game_over():
        # sleep(1)
        # print(game)

        # add states 0 -> N-1
        memory.push_x(game.vectorise())
        if turn % 2 == 0:
            a1.takeLearningAction(game)
        else:
            a2.takeLearningAction(game)
        turn += 1
        memory.push_y(evaluate_state(game))
        assert turn == game.get_move_counter()
    memory.push_result(game.evaluate())
    # add last state
    # memory.push(game.vectorise(), evaluate_state(game))
    # print("game over!")
    # print(game)
    # sentinel values for final game states - we want to evaluate a terminal position correctly, and the 
    # starting position as the minimax value for the game, so we add an additional x-y pair, 
    # y_tracker.append(game.evaluate())
    # game.reset()
    # x_tracker.append(game.vectorise())

def learn():
    memory = EpisodeMemory()
    
    loading_bar = bar.Bar("Playing against myself", max=NUM_SIMULATIONS)
    for _ in range(NUM_SIMULATIONS):
        run_game(memory)
        loading_bar.next()

    x_train: np.ndarray
    y_train: np.ndarray
    x_train, y_train = memory.get_xy_train()

    # print(f"{x_train=}")
    # print(f"{y_train=}")

    w, d, l = memory.get_wdl()
    print(f" W/D/L = {w:.2f}/{d:.2f}/{l:.2f}")

    MODEL.fit(
        x_train,
        y_train,
        BATCH_SIZE,
        epochs=10,
        validation_split=0.1
    )

    MODEL.save("td_agent_for_ttt")

if __name__ == "__main__":
    for _ in range(NUM_TRAINING_STEPS):
        learn()
