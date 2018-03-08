from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
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
            tf.logging.set_verbosity(tf.logging.INFO)
            audio_processor, training_steps_list, learning_rates_list, model_settings, time_shift_samples, X_val, y_val = KWS_data_loader(
                self.FLAGS, network_sess)

            # generate a submodel given predicted actions
            logits, fingerprint_input, is_training = model_fn(actions, model_settings)
            ground_truth_input = tf.placeholder(tf.float32, [None, model_settings['label_count']], name='groundtruth_input')
            learning_rate = 0.001

            # Optionally we can add runtime checks to spot when NaNs or other symptoms of
            # numerical errors start occurring during training.
            control_dependencies = []
            if self.FLAGS.check_nans:
                checks = tf.add_check_numerics_ops()
                control_dependencies = [checks]

            # Create the back propagation and training evaluation machinery in the graph.
            with tf.name_scope('cross_entropy'):
                cross_entropy_mean = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=ground_truth_input, logits=logits))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.name_scope('train'), tf.control_dependencies(update_ops), tf.control_dependencies(control_dependencies):
                learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
                train_op = tf.train.AdamOptimizer(learning_rate_input)
                train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)
            predicted_indices = tf.argmax(logits, 1)
            expected_indices = tf.argmax(ground_truth_input, 1)
            correct_prediction = tf.equal(predicted_indices, expected_indices)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            # Training loop.
            best_accuracy = 0
            training_steps_max = np.sum(training_steps_list)
            start_step = 1 # use for checkpoint, fixed here
            tf.global_variables_initializer().run()
          
            for training_step in xrange(start_step, training_steps_max + 1):
                X_train, y_train = audio_processor.get_data(
                    self.FLAGS.batch_size, 0, model_settings, self.FLAGS.background_frequency,
                    self.FLAGS.background_volume, time_shift_samples, 'training', network_sess)
                train_accuracy, _ = network_sess.run(
                    [
                        evaluation_step , train_step
                    ],
                    feed_dict={
                        fingerprint_input: X_train,
                        ground_truth_input: y_train,
                        learning_rate_input: learning_rate,
                        is_training: True
                    })
                #tf.logging.info('Step #%d: accuracy %.2f%%' % (training_step, train_accuracy * 100))

                is_last_step = (training_step == training_steps_max)
                if (training_step % self.FLAGS.eval_step_interval) == 0 or is_last_step:
                    validation_accuracy = network_sess.run(
                        evaluation_step,
                        feed_dict={
                            fingerprint_input: X_val,
                            ground_truth_input: y_val,
                            is_training: False
                        })
                    tf.logging.info('Step #%d: Validation accuracy %.2f%%' % (training_step, validation_accuracy * 100))
                    if validation_accuracy > best_accuracy:
                        best_accuracy = validation_accuracy
                    else:
                        learning_rate = learning_rate / 2.0

            # compute the reward
            acc = best_accuracy
            reward = (acc - self.moving_acc) * 10
            if self.moving_acc == 0.0:
                reward = 0

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0



            #print()
            #print("Manager: EWA Accuracy = ", self.moving_acc)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc