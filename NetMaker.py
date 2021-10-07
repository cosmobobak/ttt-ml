import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape

class MultilayerPerceptron:
    def __init__(self) -> None:
        input_layer = Input(
            shape=(2, 3, 3), name="input")

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
