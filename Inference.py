from State import State
from Agent import Agent
from NetMaker import MLPMaker

MODEL = MLPMaker(dims=(2, 9))()
MODEL.load_weights("td_agent_for_ttt")

if __name__ == "__main__":
    game = State()
    a1 = Agent(currentModel=MODEL, side=1)
    a2 = Agent(currentModel=MODEL, side=-1)
    turn = 1
    while not game.is_game_over():
        print(game)
        if turn == 1:
            a1.takeBestAction(game)
        else:
            a2.takeBestAction(game)
        turn = -turn
    print(game)
