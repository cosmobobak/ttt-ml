# ttt-ml
## AI for Tic-Tac-Toe

This repository contains implementations of various AI techniques for playing Tic-Tac-Toe (or Noughts and Crosses)

## Currently implemented AI algorithms include:
### Minimax (with alpha-beta pruning)
Recursive brute-force search of the solution space, but using a pruning method that ignores parts of the tree that cannot effect the outcome of a search.
### Supervised learning for a policy network, using an oracle.
Using a training set of (state, value) pairs provided by an oracle (currently set to 10% of the state space), we train a neural network (either MLP or ConvNet) that generalises to the rest of the state space.
