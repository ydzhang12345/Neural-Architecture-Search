import numpy as np
import tensorflow as tf
#from keras import backend as K
import argparse
#import csv
import os
import sys
import json

import input_data
from model import model_fn
from manager import NetworkManager

from config import *
# LAYER_SIZES = [30, 60, 100, 144]
# NUM_LAYERS = 3
# layer_sizes = [30, 60]
# num_layers = 3
# NUM_ENUM = 2
# FLAGS = None
# CLIP_REWARDS = False

NUM_ENUM = 1
def main(_):
	action_reward_dict = {}
	policy_sess = tf.Session()
	#K.set_session(policy_sess)
	manager = NetworkManager(FLAGS, clip_rewards=CLIP_REWARDS)

	size = [len(LAYER_SIZES)]*NUM_LAYERS
	reward_space = np.zeros((size))
	#print(reward_space.shape)
	for i in range(NUM_ENUM):
		for idx,val in np.ndenumerate(reward_space):
			action = [LAYER_SIZES[i] for i in idx]
			#print(action)
			with policy_sess.as_default():
				_, acc = manager.get_rewards(model_fn, action)
				print(action, acc)
				acc = round(acc*JSON_SCALE, 2)
				action = str(tuple(action))
				if action not in action_reward_dict:
					action_reward_dict[action] = [acc]
				else:
					action_reward_dict[action].append(acc)

	action_average_reward_dict = {}
	for k in action_reward_dict.keys():
		action_average_reward_dict[k] = round(np.mean(action_reward_dict[k]), 2)

	with open('action_reward_dict.json', 'w') as f:
	    json.dump(action_reward_dict, f)
	f.close()
	with open('action_average_reward_dict.json', 'w') as f:
	    json.dump(action_average_reward_dict, f)
	f.close()


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
		default='200',
		help='How many training loops to run',)
	parser.add_argument(
		'--eval_step_interval',
		type=int,
		default=200,
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
		default='yes', #default='yes,no,up,down,left,right,on,off,stop,go'
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

