import os
from random import shuffle

from ModelTools import model_evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Hyperparameters import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from State import State, perft
from NetMaker import MultilayerPerceptron
from Oracle import oracle_value
import numpy as np

debug = True

# create a model to predict evaluations of a given state 
print(f"Creating model!")  
model = MultilayerPerceptron().get_model()

# get a list of every position
print("Getting positions...")
positions_set: "set[State]" = set()
perft(State(), positions_set)

# Make the positions a list
print("Making positions a list...")
positions = list(positions_set)

# shuffle the positions
print("Shuffling positions...")
shuffle(positions)

# create xs
print(f"Vectorising...")
xs = np.array([s.vectorise() for s in positions])

# evaluate all the positions
print(f"Evaluating...")
ys = np.array([oracle_value(s) for s in positions])

if debug:
    print(f"first few training examples:")
    slice_xy = list(zip(xs, ys[:10]))
    for x, y in slice_xy:
        print(f"{x} -> {y}")
        print()

# train the model
print(f"Training... ({BATCH_SIZE = })")
model.fit(xs, ys, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

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
