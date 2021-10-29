import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from State import State
from NetMaker import ConvolutionalNeuralNetwork, MultilayerPerceptron
from Oracle import oracle_value
from ModelTools import best_state_given_model, best_state_given_twohead, mcts_new_state, model_evaluate, twohead_evaluate, twohead_new_state, twohead_policy
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import typing

if __name__ == '__main__':
    # create a model to predict evaluations of a given state
    print(f"Creating model!")
    # model = ConvolutionalNeuralNetwork().get_model()
    model: "Model" = typing.cast(
        Model, keras.models.load_model("az_models/model_it60.keras"))

    # load the model
    print(f"Loading model!")
    # model_name = f"{input('Enter model name (XXXX.h5) ==> ')}"
    # model.load_weights(model_name)

    game = State()

    while not game.is_terminal():
        print(f"\nCurrent board: \n{game}\n")
        print(f"Current player: {game.get_turn_as_str()}")
        print(f"Current move: {game.get_move_counter()}")
        print(f"Current oracle value: {oracle_value(game)}")
        # print(f"Current NN eval: {model_evaluate(game, model)}")
        print(f"Current NN eval: {twohead_evaluate(game, model)}")
        print(f"Current model policy: {twohead_policy(game, model)}")

        # next_states = game.children()
        # choice = best_state_given_model(next_states, model)
        # choice = best_state_given_twohead(next_states, model)
        choice = mcts_new_state(game, model)

        game = choice

    print(f"\nCurrent board: \n{game}\n")
    print(f"Current player: {game.get_turn_as_str()}")
    print(f"Current move: {game.get_move_counter()}")
    print(f"Current oracle value: {oracle_value(game)}")
    # print(f"Current NN eval: {model_evaluate(game, model)}")
    print(f"Current NN eval: {twohead_evaluate(game, model)}")
    print(f"Current model policy: {twohead_policy(game, model)}")

    if game.evaluate() != 0:
        print(f"\n{game.get_turn_as_str()} wins!")
    else:
        print(f"\nIt's a draw!")
