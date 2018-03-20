from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
from manager import NetworkManager
import tensorflow as tf
from model import model_fn
from random import choice
import itertools
import tensorflow as tf
from collections import defaultdict

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn, model_fn_cnn
import json
import argparse
import os.path
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import input_data
import models
from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim
FLAGS = None
states = []
'''
filter_val = [50, 100, 200]
stride_val = [1,2]
kernel_val = [3]
filter_space = [v for v in itertools.product(filter_val, repeat=3)]
stride_space = [v for v in itertools.product(stride_val, repeat=3)]
kernal_space = [v for v in itertools.product(kernel_val, repeat=3)]
for itr in xrange(3):
    states.append([random.choice(filter_val), random.choice(kernel_val), random.choice(stride_val),
                      random.choice(filter_val), random.choice(kernel_val), random.choice(stride_val),
                      random.choice(filter_val), random.choice(kernel_val), random.choice(stride_val)])
'''
states = []
with open('Main_case.txt', 'r') as fin:
    for line in fin:
        line = line.strip('\n')
        line = line.strip()
        line = line.split(' ')
        s = []
        for l in line:
            s.append(int(l))
        states.append(s)

def main(_):
    CLIP_REWARDS = False
    data = defaultdict(list)
    with open('main_result.txt', 'w') as out:
        for ite in xrange(3):
            ite += 1
            print ('outter iteration:',ite)
            iteration = 0
            for state in states:
                iteration +=1
                print (iteration,state)
                manager = NetworkManager(FLAGS, clip_rewards=CLIP_REWARDS)
                reward, previous_acc = manager.get_rewards(model_fn_cnn, state)
                previous_acc = round(previous_acc*100000,2)
                print (previous_acc)
                data[str(state)].append(previous_acc)
                out.write("{} {}\n".format(state, previous_acc))
    with open('main_sample.json', 'w') as outfile:
        json.dump(data, outfile)
           

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
      default='1000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=100,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001',
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
      default='yes,no', #default='yes,no,up,down,left,right,on,off,stop,go'
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


