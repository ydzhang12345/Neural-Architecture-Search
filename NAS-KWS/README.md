# Neural Architecture Search with Controller RNN for the task of Keyword-spotting

Basic implementation of Controller RNN from [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) and [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012).

- Define a state space by using `StateSpace`, a manager which adds states and handles communication between the Controller RNN and the user.
- `Controller` manages the training and evaluation of the Controller RNN
- `NetworkManager` handles the training and reward computation of a Keras model

# Usage
At a high level : For full training details, please see `train.py`.
```python
# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])

# create the managers
controller = Controller(tf_session, num_layers, state_space)
manager = NetworkManager(dataset, epochs=max_epochs, batchsize=batchsize)

# For number of trials
  sample_state = ...
  actions = controller.get_actions(sample_state)
  reward = manager.get_reward(actions)
  controller.train()
```


Implementation details were found from:
- http://rll.berkeley.edu/deeprlcoursesp17/docs/quoc_barret.pdf

# Result
I tried a toy CNN model with 4 CNN layers with different filter sizes (16, 32, 64) and kernel sizes (1, 3) to maximise score in 10 epochs of training on CIFAR-10.


# Acknowledgements
Code heavily inspired by https://github.com/titu1994/neural-architecture-search

