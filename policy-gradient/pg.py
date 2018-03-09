# -*- coding: UTF-8 -*-

import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import json
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from pg_config import pg_config
#import logz
from config import *
import pdb
from scipy.stats import multivariate_normal


class PG(object):

  def __init__(self):
    self.lr = 5e-2
    self.controller_cells = 64
    self.batch_size = 500
    self.num_batches = 200
    self.observation_dim = 1000
    self.action_dim = 1000
    self.num_layers = 2

    self.action_buffer = []
    self.state_buffer = []
    self.logprob_buffer = []
    self._dict = {}
    self._used_dict = {}
    self.log_acc = []
    self.logger = get_logger('./log.txt')

    self._num_used_models = []

    self._initial_baseline = 0.05

    '''
    with open('./action_average_reward_dict.json', 'r') as f:
      self._raw_dict = json.load(f)
    temp_map = {30:0,  60:1, 100:2, 144:3}
    for key in self._raw_dict.keys():
      actions = [temp_map[int(a)] for a in key[1:-1].split(',')]
      temp = str(actions).replace(",","")
      accuracy = float(self._raw_dict[key]) / 10000
      self._dict[temp] = accuracy
      self._used_dict[temp] = 0
    '''
    self._dict = self.build_reward_function()
    self._used_dict = np.zeros_like(self._dict)
    self.build()


  
  def build_reward_function(self):
    x, y = np.mgrid[-10:10:0.02, -10:10:0.02]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv1 = multivariate_normal([5, -5], [[10, 0], [0, 10]])
    rv2 = multivariate_normal([2, -2], [[7, 2], [2, 5]])
    rv3 = multivariate_normal([7, -7], [[1, 0], [0, 1]])
    rv4 = multivariate_normal([3, -3], [[1, 0], [0, 1]])
    rv11 = multivariate_normal([-5, 5], [[3, 1], [1, 2]]) 
    rv22 = multivariate_normal([-2, 2], [[7, 2], [2, 5]])
    rv33 = multivariate_normal([-7, 7], [[1, 0], [0,1]])
    rv44 = multivariate_normal([-3, 3], [[4, 0], [0, 4]])
    rv = rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos) + rv4.pdf(pos) + rv11.pdf(pos) + rv22.pdf(pos) + rv33.pdf(pos) + rv44.pdf(pos)
    pdb.set_trace()
    return rv


  def add_placeholders_op(self):
    self.observation_placeholder = tf.placeholder(tf.float32, [self.batch_size, 1, self.observation_dim])
    self.action_placeholder = tf.placeholder(tf.int32, [self.num_layers, self.batch_size])
    self.advantage_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.num_layers])


  def build_policy_network_op(self, scope="policy_network"):
    temp_logprob_buffer = []
    with tf.variable_scope(scope):
      self.cell = tf.contrib.rnn.NASCell(self.controller_cells)
    cell_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
    cell_input = tf.zeros([self.batch_size, 1, self.observation_dim])
    for i in range(self.num_layers):
      outputs, cell_state = tf.nn.dynamic_rnn(self.cell, cell_input, initial_state=cell_state, dtype=tf.float32)
      action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim, name='rnn_fc_%d' % (i))

      sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
      cell_input = tf.one_hot(sampled_action, self.observation_dim)
      cell_input = tf.expand_dims(cell_input, 1)
      logprob = tf.negative(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_placeholder[i], logits=action_logits))
      logprob = tf.expand_dims(logprob, 1)
      
      self.action_buffer.append(sampled_action) #action
      #self.state_buffer.append(cell_input)  # state
      temp_logprob_buffer.append(logprob) #logprob
    
    self.logprob_buffer = tf.concat(temp_logprob_buffer, 1) # batch x layer



  def add_loss_op(self):
    self.loss = -tf.reduce_mean(self.logprob_buffer * self.advantage_placeholder)


  def add_optimizer_op(self):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


  def build(self):
    self.add_placeholders_op()
    self.build_policy_network_op()
    self.add_loss_op()
    self.add_optimizer_op()


  def initialize(self):
    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)


  def sample_model_reward_return(self):
    action_buffer = np.array(self.sess.run(self.action_buffer))
    returns = np.float32(np.zeros_like(action_buffer))
    #pdb.set_trace()
    for i in range(self.batch_size):
      #returns[:,i] = self._dict[str(action_buffer[:,i])] - self._initial_baseline
      returns[:, i] = self._dict[action_buffer[:, i][0], action_buffer[:, i][1]]
      self._used_dict[action_buffer[:, i][0], action_buffer[:, i][1]] = 1

    return action_buffer, np.transpose(returns)


  def train(self):
  
    for t in range(self.num_batches):
      actions, returns = self.sample_model_reward_return()

      self.sess.run(self.train_op, feed_dict={ 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : returns})
      

      avg_acc = np.mean(returns)

      #calculate number of used models:
      used = 0
      #for key in self._used_dict.keys():
        #used += self._used_dict[key]
      used = np.sum(self._used_dict)
      self._num_used_models.append(used)


      self.log_acc.append(avg_acc)
      #sigma_reward = np.sqrt(np.var(returns) / len(total_rewards))
      msg = "Average accuracy within a batch: {:04.2f}".format(avg_acc)
      self.logger.info(msg)
      print (actions)

  
    self.logger.info("- Training done.")
    #export_plot(self.log_acc, "Batch_Accuracy", 'NAS-DNN', "./batch_accuracy.png", self._num_used_models, "Sampled Model")
    export_plot(self.log_acc, "Score", 'NAS-DNN', "./batch_accuracy.png")
    export_plot(self._num_used_models, "Models Sampled", 'NAS-DNN', "./used_models.png")


  def run(self):
    self.initialize()
    self.train()



if __name__ == '__main__':
    model = PG()
    model.run()
