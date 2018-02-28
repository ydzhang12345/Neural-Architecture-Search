from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn

import pdb
import argparse
import os.path
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin

import input_data
import models
from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim

FLAGS = None


def main(_):
    # create a shared session between Keras and Tensorflow
    policy_sess = tf.Session()
    K.set_session(policy_sess)

    NUM_LAYERS = 3  # number of layers of the state space
    MAX_TRIALS = 250  # maximum number of models generated

    MAX_EPOCHS = 60  # maximum number of epochs to train
    BATCHSIZE = 100  # batchsize
    EXPLORATION = 0.8  # high exploration for the first 1000 steps
    REGULARIZATION = 1e-3  # regularization strength
    CONTROLLER_CELLS = 32  # number of cells in RNN controller
    CLIP_REWARDS = False  # clip rewards in the [-0.05, 0.05] range
    RESTORE_CONTROLLER = True  # restore controller to continue training

    # construct a state space
    state_space = StateSpace()

    # add states
    state_space.add_state(name='kernel', values=[3])
    state_space.add_state(name='filters', values=[64])
    state_space.add_state(name='stride', values=[1])

    # print the state space being searched
    state_space.print_state_space()

    previous_acc = 0.0
    total_reward = 0.0

    with policy_sess.as_default():
        # create the Controller and build the internal policy network
        controller = Controller(policy_sess, NUM_LAYERS, state_space,
                                reg_param=REGULARIZATION,
                                exploration=EXPLORATION,
                                controller_cells=CONTROLLER_CELLS,
                                restore_controller=RESTORE_CONTROLLER)

    # create the Network Manager
    manager = NetworkManager(FLAGS, clip_rewards=CLIP_REWARDS)

    # get an initial random state space if controller needs to predict an
    # action from the initial state
    state = state_space.get_random_state_space(NUM_LAYERS)
    print("Initial Random State : ", state_space.parse_state_space_list(state))
    print()

    # train for number of trails
    for trial in range(MAX_TRIALS):
        with policy_sess.as_default():
            K.set_session(policy_sess)
            actions = controller.get_action(state)  # get an action for the previous state

        # print the action probabilities
        state_space.print_actions(actions)
        print("Predicted actions : ", state_space.parse_state_space_list(actions))

        # build a model, train and get reward and accuracy from the network manager
        reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
        print("Rewards : ", reward, "Accuracy : ", previous_acc)

        with policy_sess.as_default():
            K.set_session(policy_sess)

            total_reward += reward
            print("Total reward : ", total_reward)

            # actions and states are equivalent, save the state and reward
            state = actions
            controller.store_rollout(state, reward)

            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step()
            print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

            # write the results of this trial into a file
            with open('train_history.csv', mode='a+') as f:
                data = [previous_acc, reward]
                data.extend(state_space.parse_state_space_list(state))
                writer = csv.writer(f)
                writer.writerow(data)
        print()

    print("Total Reward : ", total_reward)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=10,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)