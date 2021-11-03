import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, Add
from tensorflow.keras.models import Model
import tensorflow as tf

TTT_DIM = (3, 3, 2)
TTT_ACTION_SPACE = 9
C4_DIM = (6, 7, 2)
C4_ACTION_SPACE = 7

input_layer = Input(
    shape=C4_DIM, name="input")

x = Conv2D(
    filters=256,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu",
    name="Conv0",
    input_shape=C4_DIM,
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

policy_head = x
policy_head = Dense(256, activation="relu")(policy_head)
policy_head = Dense(28, activation="relu")(policy_head)

policy_out = Dense(C4_ACTION_SPACE, activation="softmax",
                   name="policy_head")(policy_head)

value_head = x
value_head = Dense(64, activation="relu")(value_head)

value_out = Dense(1, activation="tanh", name="value_head")(value_head)

bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model = Model(inputs=input_layer, outputs=[policy_out, value_out])

model.compile(
    optimizer="SGD",
    loss={"policy_head": bce, "value_head": "mse"}
)

model.summary()

model.save("az_models/random_model.keras")

print("Random model created!")
