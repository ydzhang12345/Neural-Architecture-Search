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

data_mean = 95942.2077661
data_std = 1317.44860489

class PG(object):

  def __init__(self):
    self.lr = 10e-2
    self.batch_size = 64
    self.controller_cells = 64
    self.num_iterations = 100
    self.observation_dim = 4
    self.action_dim_1 = 4
    self.action_dim_2 = 2
    self.action_dim_3 = 2
    self.num_layers = 3
    self.num_actions_per_layer = 3

    self.hasConstraint = True
    self.hardConstraint = True
    self.reg_weight = 1e-5
    self.reg_op = 1e-8
    self.weight_limit = 8000
    self.op_limit = 1e8

    self.temp1 = []
    self.temp2 = []

    self.action_buffer = []
    self.state_buffer = []
    self.logprob_buffer = []
    self._dict = {}
    self._used_dict = {}
    self.log_acc = []
    self.logger = get_logger('./log.txt')

    self._num_used_models = []

    self._initial_baseline = 0.05

    #with open('./unormdata.json', 'r') as f:
    with open('./normalizedata.json', 'r') as f:
      self._raw_dict = json.load(f)
    f.close()
    filter_nums_map = {10:0, 50:1, 100:2, 200:3}
    kernel_sizes_map = {3:0, 5:1}
    strides_map = {1:0, 2:1}
    for key in self._raw_dict.keys():
      params = key[1:-1].split(',')
      temp = []
      for i in range(9):
        if i%3 == 0: temp.append(filter_nums_map[int(params[i])])
        elif i%3 == 1: temp.append(kernel_sizes_map[int(params[i])])
        else: temp.append(strides_map[int(params[i])])

      self._dict[str(temp)] = np.mean(self._raw_dict[key])
      self._used_dict[str(temp)] = 0
    self.build()



  def add_placeholders_op(self):
    self.observation_placeholder = tf.placeholder(tf.float32, [self.batch_size, 1, self.observation_dim])
    self.action_placeholder = tf.placeholder(tf.int32, [self.num_layers*self.num_actions_per_layer, self.batch_size])
    self.advantage_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.num_layers*self.num_actions_per_layer])



  def build_policy_network_op(self, scope="policy_network"):
    temp_logprob_buffer = []
    with tf.variable_scope(scope):
      self.cell = tf.contrib.rnn.NASCell(self.controller_cells)
      cell_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
      cell_input = tf.zeros([self.batch_size, 1, self.observation_dim])
      for i in range(self.num_layers*self.num_actions_per_layer):
        outputs, cell_state = tf.nn.dynamic_rnn(self.cell, cell_input, initial_state=cell_state, dtype=tf.float32)
        # outputs[:, -1, :].shape = (batch_size, controller_cells)
        #action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_1, name='rnn_fc', reuse=tf.AUTO_REUSE)
        if i%3 == 0: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_1, name='rnn_fc_%d' % (i))
        elif i%3 == 1: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_2, name='rnn_fc_%d' % (i))
        else: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_3, name='rnn_fc_%d' % (i))
        # if i%3 == 0: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_1, name='rnn_fc_1', reuse=tf.AUTO_REUSE)
        # elif i%3 == 1: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_2, name='rnn_fc_2', reuse=tf.AUTO_REUSE)
        # else: action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_3, name='rnn_fc_3', reuse=tf.AUTO_REUSE)

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


  def sample_model_reward_return(self, t):
    filter_nums_map = {0:10, 1:50, 2:100, 3:200}
    kernel_sizes_map = {0:3, 1:5}
    strides_map = {0:1, 1:2}

    action_buffer = np.array(self.sess.run(self.action_buffer))
    # action_buffer.shape = (num_layers, N)
    returns = np.float32(np.zeros_like(action_buffer))
    losses = np.float32(np.zeros_like(action_buffer))
    #pdb.set_trace()
    for i in range(self.batch_size):
      #returns[:,i] = self._dict[str(action_buffer[:,i])] - self._initial_baseline
      #returns[:, i] = self._dict[action_buffer[:, i][0], action_buffer[:, i][1]]
      #self._used_dict[action_buffer[:, i][0], action_buffer[:, i][1]] = 1
      temp = action_buffer[:, i]
      #print temp
      # temp[1] = 0 if temp[1]<=1 else 1
      # temp[2] = 0 if temp[2]<=1 else 1
      # temp[4] = 0 if temp[4]<=1 else 1
      # temp[5] = 0 if temp[5]<=1 else 1
      # temp[7] = 0 if temp[7]<=1 else 1
      # temp[8] = 0 if temp[8]<=1 else 1

      filter1, kernel1, stride1 = filter_nums_map[temp[0]], kernel_sizes_map[temp[1]], strides_map[temp[2]]
      filter2, kernel2, stride2 =  filter_nums_map[temp[3]], kernel_sizes_map[temp[4]], strides_map[temp[5]]
      filter3, kernel3, stride3 =  filter_nums_map[temp[6]], kernel_sizes_map[temp[7]], strides_map[temp[8]]

      if str(temp) not in self._dict.keys():
        # print 'not in buffer', [filter1, kernel1, stride1, filter2, kernel2, stride2, filter3, kernel3, stride3]
        s = str([filter1, kernel1, stride1, filter2, kernel2, stride2, filter3, kernel3, stride3])
        #print self._raw_dict[s]
        #self._dict[str(temp)] = np.mean(self._raw_dict[s]) / 100000.
        self._dict[str(temp)] = np.mean(self._raw_dict[s])
      
      returns[:, i] = self._dict[str(temp)] #- self._initial_baseline

      if self.hasConstraint:
        weights = (filter1 + 1) * (kernel1**2)
        weights += (filter2 + 1) * (kernel2**2)
        weights += (filter3 + 1) * (kernel3**2)
        
        t_in, f_in = 49, 10
        t1, f1 = np.ceil(t_in/stride1), np.ceil(f_in/stride1)
        t2, f2 = np.ceil(t1/stride2), np.ceil(f1/stride2)
        t3, f3 = np.ceil(t2/stride3), np.ceil(f2/stride3)
        ops = 2*1*t1*f1*stride1**2*filter1 + t1*f1*filter1
        ops += (2*filter1*t2*f2*stride2**2 + filter1*t2*f2) + (2*filter1*t2*f2*filter2 + t2*f2*filter2)
        ops += (2*filter2*t3*f3*stride3**2 + filter2*t3*f3) + (2*filter2*t3*f3*filter3 + t3*f3*filter3)
      
        self.temp1.append(weights); self.temp2.append(ops)

        if self.hardConstraint:
          if weights > self.weight_limit or ops > self.op_limit:
            #returns[:, i] = 0
            returns[:, i] = -data_mean
            losses[:, i] = 0
          else:
            losses[:, i] = 0
        else:
          losses[:, i] = self.reg_weight*weights + self.reg_op*ops

      self._used_dict[str(temp)] = 1
      if t==self.num_iterations-1 and i>=self.batch_size-5:
        print 'converges at:', [filter_nums_map[temp[0]], kernel_sizes_map[temp[1]], strides_map[temp[2]],\
                                filter_nums_map[temp[3]], kernel_sizes_map[temp[4]], strides_map[temp[5]],\
                                filter_nums_map[temp[6]], kernel_sizes_map[temp[7]], strides_map[temp[8]]]
      #print np.mean(losses), np.mean(returns)
    return action_buffer, np.transpose(returns), np.transpose(losses)


  def train(self):
  
    for t in range(self.num_iterations):
      actions, returns, losses = self.sample_model_reward_return(t)

      self.sess.run(self.train_op, feed_dict={ 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : returns-losses})
      
      #avg_acc = np.mean(returns)
      avg_acc = (np.mean(returns)*data_std + data_mean) / 100000.

      #calculate number of used models:
      used = 0
      for key in self._used_dict.keys():
        used += self._used_dict[key]
      #used = np.sum(self._used_dict)
      self._num_used_models.append(used)


      self.log_acc.append(avg_acc)
      #sigma_reward = np.sqrt(np.var(returns) / len(total_rewards))
      msg = "Average accuracy within a batch: {:04.2f}".format(avg_acc*100)
      self.logger.info(msg)
      #print (actions)

  
    self.logger.info("- Training done.")
    #export_plot(self.log_acc, "Batch_Accuracy", 'NAS-DNN', "./batch_accuracy.png", self._num_used_models, "Sampled Model")
    export_plot(self.log_acc, "Score", 'NAS-DNN', "./batch_accuracy.png")
    export_plot(self._num_used_models, "Models Sampled", 'NAS-DNN', "./used_models.png")

    print 'log_acc'; print self.log_acc
    print '_num_used_models'; print self._num_used_models
    # print 'weights', np.mean(self.temp1), np.var(self.temp1)
    # print 'ops', np.mean(self.temp2), np.var(self.temp2)


  def run(self):
    self.initialize()
    self.train()



if __name__ == '__main__':
    model = PG()
    model.run()
