import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, Add
from tensorflow.keras.models import Model
import tensorflow as tf

inp = Input((3, 3, 2))

l0 = Flatten()(inp)

l1 = Dense(128, activation="relu")(l0)
l2 = Dense(128, activation="relu")(l1)
l3 = Dense(128, activation="relu")(l2)
l4 = Dense(128, activation="relu")(l3)
l5 = Dense(128, activation="relu")(l4)

policy_out = Dense(9, activation="softmax", name="policy_head")(l5)
value_out = Dense(1, activation="tanh", name="value_head")(l5)

bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model = Model(inputs=inp, outputs=[policy_out, value_out])

model.compile(
    optimizer="SGD",
    loss={"policy_head": bce, "value_head": "mse"}
)

model.save("random_model.keras")

print("Random model created!")