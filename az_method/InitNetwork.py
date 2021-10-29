import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, Add
from tensorflow.keras.models import Model
import tensorflow as tf

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

policy_out = Dense(9, activation="softmax", name="policy_head")(x)
value_out = Dense(1, activation="tanh", name="value_head")(x)

bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model = Model(inputs=input_layer, outputs=[policy_out, value_out])

model.compile(
    optimizer="SGD",
    loss={"policy_head": bce, "value_head": "mse"}
)

model.save("az_models/random_model.keras")

print("Random model created!")
