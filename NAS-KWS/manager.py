from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import pdb
from six.moves import xrange  # pylint: disable=redefined-builtin
import pdb
import argparse
import os.path
import sys
import input_data
import models
from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim


def KWS_data_loader(FLAGS, sess):
    #sess = tf.Session()
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    validation_fingerprints, validation_ground_truth = (
                audio_processor.get_data(-1, 0, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    validation_fingerprints = validation_fingerprints.reshape(-1, input_time_size, input_frequency_size, 1)
    
    return audio_processor, training_steps_list, learning_rates_list, model_settings, time_shift_samples, validation_fingerprints, validation_ground_truth

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, FLAGS, acc_beta=0.8, clip_rewards=False):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: whether to clip rewards in [-0.05, 0.05] range to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.FLAGS = FLAGS
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, model_fn, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            audio_processor, training_steps_list, learning_rates_list, model_settings, time_shift_samples, X_val, y_val = KWS_data_loader(
                self.FLAGS, network_sess)


            # generate a submodel given predicted actions
            model = model_fn(actions)  # type: Model
            adam = Adam(lr=0.001)
            model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])


            # unpack the dataset
            # Training loop.
            best_accuracy = 0
            training_steps_max = np.sum(training_steps_list)
            start_step = 1 # use for checkpoint, fixed here
            steps = audio_processor.set_size('training') // self.FLAGS.batch_size


            
            #for training_step in xrange(start_step, training_steps_max + 1):
            for epochs in range(66):
                # Training loop.
                #training_steps_sum = 0
                #for i in range(len(training_steps_list)):
                #  training_steps_sum += training_steps_list[i]
                #  if training_step <= training_steps_sum:
                #    learning_rate_value = learning_rates_list[i]
                #    break
                #K.set_value(adam.lr, 0.5 * K.get_value(adam.lr))
                # Pull the audio samples we'll use for training.
                train_fingerprints, y_train = audio_processor.get_data(
                    -1, 0, model_settings, self.FLAGS.background_frequency,
                    self.FLAGS.background_volume, time_shift_samples, 'training', network_sess)
                X_train = train_fingerprints.reshape(
                    -1, model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1)

                # train the model using Keras methods
                for step in range(steps + 1):
                    start = step*self.FLAGS.batch_size
                    if (step == steps):
                        end = -1  
                    else:
                        end = (step+1)*self.FLAGS.batch_size
                    batch_loss, batch_acc = model.train_on_batch(X_train[start:end], y_train[start:end])
                    print ('training_steps:', step, 'batch_acc:', batch_acc)

                if (epochs % 2 == 0):
                    loss, acc = model.evaluate(X_val, y_val, batch_size=self.FLAGS.batch_size)
                    print ('epochs', epochs, 'validation_acc:', acc)
                    if acc > best_accuracy:
                        best_accuracy = acc
                    else:
                        K.set_value(adam.lr, 0.5 * K.get_value(adam.lr))

                '''
                is_last_step = (training_step == training_steps_max)
                if (training_step % self.FLAGS.eval_step_interval) == 0 or is_last_step:
                    loss, acc = model.evaluate(X_val, y_val, batch_size=self.FLAGS.batch_size)
                    print ('validation_acc:', acc)
                    if acc > best_accuracy:
                        best_accuracy = acc
                    else:
                        K.set_value(adam.lr, 0.5 * K.get_value(adam.lr))
                '''
            

            # load best performance epoch in this training session
            # model.load_weights('weights/temp_network.h5')

            # evaluate the model
            # loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize)

            # compute the reward
            reward = (best_accuracy - self.moving_acc)

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0

            print()
            print("Manager: EWA Accuracy = ", self.moving_acc)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc