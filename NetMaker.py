import os
from Hyperparameters import DEBUG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, Add

class MultilayerPerceptron:
    def __init__(self) -> None:
        input_layer = Input(
            shape=(3, 3, 2), name="input")

        x = Flatten()(input_layer)
        x = Dense(16, activation="relu", name="Dense0")(x)
        x = Dense(8, activation="relu", name="Dense1")(x)
        
        output_layer = Dense(1, activation="tanh", name="eval")(x)
        
        self.evaluation_model = Model(inputs=input_layer, outputs=output_layer)

        losstype = "mse"

        self.evaluation_model.compile(
            optimizer="sgd",
            loss=losstype,
            metrics=[],
        )

        self.evaluation_model.summary()

    def get_model(self) -> Model:
        return self.evaluation_model


class ConvolutionalNeuralNetwork:
    def __init__(self) -> None:
        input_layer = Input(
            shape=(3, 3, 2), name="input")

        x = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv0",
            input_shape=(3, 3, 2),
        )(input_layer)
        x = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv1",
        )(x)
        x = Flatten()(x)
        x = Dense(3, activation="relu", name="Dense1")(x)

        output_layer = Dense(1, activation="tanh", name="eval")(x)

        self.evaluation_model = Model(inputs=input_layer, outputs=output_layer)

        losstype = "mse"

        self.evaluation_model.compile(
            optimizer="sgd",
            loss=losstype,
            metrics=[],
        )

        if DEBUG:
            self.evaluation_model.summary()

    def get_model(self) -> Model:
        return self.evaluation_model
