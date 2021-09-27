import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape
import tensorflow as tf

BATCH_SIZE = 1

class MLPMaker:
    def __init__(self, xbatch_size: int = BATCH_SIZE) -> None:
        inputLayer = Input(
            shape=(2, 3, 3), batch_size=xbatch_size, name="input")

        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(inputLayer)
        # x = Dense(2048, activation="relu", name="Dense0")(x)
        # x = Dense(2048, activation="relu", name="Dense1")(x)
        # x = Dense(64, activation="relu", name="Dense2")(x)
        x = Dense(64, activation="relu", name="Dense3")(x)
        x = Dense(64, activation="relu", name="Dense4")(x)
        
        outputLayer = Dense(1, activation="tanh", name="eval")(x)
        self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

        self.evalModel.compile(
            optimizer="sgd",
            loss="mse",
            metrics=[],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel
