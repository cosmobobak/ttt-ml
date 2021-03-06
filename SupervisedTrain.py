import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from random import shuffle
from ModelTools import model_evaluate
from Hyperparameters import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, DEBUG
from State import State
from NetMaker import ConvolutionalNeuralNetwork, MultilayerPerceptron
from Oracle import oracle_value
import numpy as np
import tensorflow as tf

# get a list of every position
print("Getting positions...")
positions_set = State.get_every_state()

# Make the positions a list
print("Making positions a list...")
positions = list(positions_set)

# shuffle the positions
print("Shuffling positions...")
shuffle(positions)

# create xs
print(f"Vectorising...")
xs = np.array([s.vectorise_chlast() for s in positions])

# evaluate all the positions
print(f"Evaluating...")
ys = np.array([oracle_value(s) for s in positions])

if DEBUG:
    print(f"first few training examples:")
    slice_xy = list(zip(xs, ys[:10]))
    for x, y in slice_xy:
        print(f"{x} -> {y}")
        print()

# create a model to predict evaluations of a given state
print(f"Creating model!")
model = ConvolutionalNeuralNetwork().get_model()

# create TensorBoard callback
print(f"Creating TensorBoard callback...")
logdir = os.path.join(os.path.curdir, "logs")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# train the model
print(f"Training... ({BATCH_SIZE = })")
model.fit(xs, ys, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[tb_callback])

# save the model
print(f"Saving...")
mname = "supervised_model.h5"
model.save(mname)
print(f"Saved as {mname}")

# validate the model
print(f"Validating...")
print(model.evaluate(xs, ys))

print(f"Value of start-position:")
print(model_evaluate(State(), model))
