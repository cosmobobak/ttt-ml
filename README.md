# ttt-ml
## AI for Tic-Tac-Toe

This repository contains implementations of various AI techniques for playing Tic-Tac-Toe (or Noughts and Crosses)

## Currently implemented AI algorithms include:
### Minimax (with alpha-beta pruning)
Recursive brute-force search of the solution space, but using a pruning method that ignores parts of the tree that cannot effect the outcome of a search.
### Supervised learning for a value network, using an oracle.
Using a training set of (state, value) pairs provided by an oracle (currently set to 10% of the state space), we train a neural network (either MLP or ConvNet) that generalises to the rest of the state space.
### CPUCT Tree Search with a two-headed policy-value network.
AlphaZero / Lc0 style self-play reinforcement learning. Starting from a randomly initialised network, an agent plays itself over and over, updating both move predictions and state-value estimations. 
#### Good resources for this include
- [Iterated Distillation and Amplification](https://youtu.be/v9M2Ho9I9Qo)
- [AlphaZero Paper on ArXiv](https://arxiv.org/abs/1712.01815)
- [Dominik Klein's Neural Networks for Chess](https://github.com/asdfjkl/neural_network_chess) (from which our implementation was essentially cloned, with some stylistic changes)

