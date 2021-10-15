import os
from Hyperparameters import DEBUG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, Add
from keras.regularizers import l2

class MultilayerPerceptron:
    def __init__(self) -> None:
        input_layer = Input(
            shape=(3, 3, 2), name="input")

        x = Flatten()(input_layer)
        x = Dense(128, activation="relu", name="Dense0")(x)
        x = Dense(64, activation="relu", name="Dense1")(x)
        x = Dense(32, activation="relu", name="Dense2")(x)
        x = Dense(16, activation="relu", name="Dense3")(x)
        x = Dense(8, activation="relu", name="Dense4")(x)
        
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
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv0",
            input_shape=(3, 3, 2),
        )(input_layer)
        x = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv1",
        )(x)
        preconv = x
        postconv = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv2",
        )(preconv)
        preconv = Add()([preconv, postconv])
        postconv = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv3",
        )(preconv)
        preconv = Add()([preconv, postconv])
        postconv = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv4",
        )(preconv)
        preconv = Add()([preconv, postconv])
        x = preconv
        x = Flatten()(x)
        x = Dense(32, activation="relu", name="Dense0")(x)

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
