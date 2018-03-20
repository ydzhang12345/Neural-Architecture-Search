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
import matplotlib.pyplot as plt


data_mean = 95.74566997
data_std = 1.27014307

data_max = 102.65417672555384
data_min = 64.0118744

tf.set_random_seed(0)

class PG(object):

  def __init__(self):
    self.lr = 5e-2
    self.batch_size = 500
    self.controller_cells = 128
    self.num_iterations = 5000
    self.observation_dim = 100
    self.action_dim_1 = 1
    self.action_dim_2 = 2
    self.action_dim_3 = 2
    self.num_layers = 3
    self.num_actions_per_layer = 3

    self.hasConstraint = False
    self.hardConstraint = False
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

    self._initial_baseline = 0
    self.max_filter = 100
    self._used_models = []

    with open('./norm_inter_acc.json', 'r') as f:
      self._raw_dict = json.load(f)

    self.build()


  def analyze_data(self):
    dictt = []
    for key in self._raw_dict.keys():
      if self._raw_dict[key]*2 >= 4.07:
        dictt.append(key)
    pdb.set_trace()
    a=1
    


  def interpolate_continues_reward(self, d, input_dict):
    temp_dict = {}
    all_dict = {}
    for key in input_dict.keys():
      params = key[1:-1].split(',')
      test1 = int(params[-3])
      test2 = int(params[-6])
      test3 = int(params[-9])
      if test1>100 or test2>100 or test3>100:
        continue
      keyword = list(map(int, params))
      new_key = str(keyword[:-3*d] + ['N'] + keyword[-3*d+1:])
      if new_key not in temp_dict:
        temp_dict[new_key] = {}
        temp_dict[new_key]['x'] = []
        temp_dict[new_key]['y'] = []
      temp_dict[new_key]['x'].append(keyword[-3*d])
      temp_dict[new_key]['y'].append((input_dict[key]))

    maxx = -100
    minn = 100
    # interpolate
    for key in temp_dict.keys():
      x_vec = temp_dict[key]['x']
      y_vec = temp_dict[key]['y']
      z = np.polyfit(x_vec, y_vec, 2)
      fff = np.poly1d(z)
      xp = np.linspace(1,100,100)
      yp = fff(xp)
      params = key[1:-1].split(',')
      for j, i in enumerate(xp):
        temp = params[:-3*d] + [i] + params[-3*d+1:]
        keyword = list(map(int, temp))
        all_dict[str(keyword)] =  yp[j]
      maxx = max(maxx, np.max(yp)) 
      minn = min(minn, np.min(yp))
    print ('max:', maxx)
    print ('min:', minn)
    return all_dict



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
    return rv


  def add_placeholders_op(self):
    self.action_placeholder = tf.placeholder(tf.int32, [self.num_layers*(self.num_actions_per_layer-1), self.batch_size])
    self.con_action_placeholder = tf.placeholder(tf.float32, [self.num_layers, self.batch_size])
    self.advantage_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.num_layers*self.num_actions_per_layer])



  def build_policy_network_op(self, scope="policy_network"):
    temp_logprob_buffer = []
    with tf.variable_scope(scope):
      self.cell = tf.contrib.rnn.NASCell(self.controller_cells)
      cell_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
      cell_input = tf.zeros([self.batch_size, 1, self.observation_dim])
      for i in range(self.num_layers*self.num_actions_per_layer):
        outputs, cell_state = tf.nn.dynamic_rnn(self.cell, cell_input, initial_state=cell_state, dtype=tf.float32)
        if i%3 == 0: 
          temp = tf.layers.dense(outputs[:, -1, :], units=self.observation_dim, name='rnn_fc_pre_1%d' % (i), bias_initializer=tf.constant_initializer(50.0))
          temp = tf.nn.relu(temp)
          action_means1 = tf.reduce_mean(temp, [1])
          action_means1 = tf.expand_dims(action_means1, 1)
          #action_means1 = tf.layers.dense(temp, units=self.action_dim_1, name='rnn_fc_1%d' % (i))
          log_std1 = tf.get_variable('log_std_1' + str(i), shape=[self.action_dim_1], initializer=tf.constant_initializer(2.0))
          mvn1 = tf.contrib.distributions.MultivariateNormalDiag(action_means1, tf.exp(log_std1))
          logprob = mvn1.log_prob(tf.expand_dims(self.con_action_placeholder[int(i/3)], 1))
          logprob = tf.expand_dims(logprob, 1)

          epsilon = tf.random_normal(shape=[self.action_dim_1], mean=0.0, stddev=1.0)
          sampled_action = action_means1 + epsilon * tf.exp(log_std1)
          sampled_action = tf.squeeze(sampled_action, axis=1)

          round_action = tf.cast(tf.round(sampled_action), tf.int32)
          round_action = tf.minimum(round_action, tf.ones([self.batch_size], dtype=tf.int32)*(self.observation_dim-1))
          round_action = tf.maximum(round_action, tf.zeros([self.batch_size], dtype=tf.int32))
          cell_input = tf.one_hot(round_action, self.observation_dim)
          cell_input = tf.expand_dims(cell_input, 1)


        else:
          if i%3 == 1: 
            action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_2, name='rnn_fc_%d' % (i))
          else: 
            action_logits = tf.layers.dense(outputs[:, -1, :], units=self.action_dim_3, name='rnn_fc_%d' % (i))
          index = 2*int(i/3) + i%3 - 1

          sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
          cell_input = tf.one_hot(sampled_action, self.observation_dim)
          cell_input = tf.expand_dims(cell_input, 1)
          logprob = tf.negative(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_placeholder[index], logits=action_logits))
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
    returns = np.float32(np.zeros_like(action_buffer))
    losses = np.float32(np.zeros_like(action_buffer))

    actions = np.zeros([self.num_layers*(self.num_actions_per_layer-1), self.batch_size])
    con_actions = np.zeros([self.num_layers, self.batch_size])


    for i in range(self.batch_size):
      temp = action_buffer[:, i].copy()
      actions[:,i] = np.array([temp[1], temp[2], temp[4],temp[5],temp[7],temp[8]])
      con_actions[:,i] = np.array([temp[0],temp[3],temp[6]])
      flag = 0
      for j in [0,3,6]:
        temp[j] = np.minimum(temp[j], self.observation_dim)
        temp[j] = np.round(temp[j])
        if temp[j] < 1:
          flag = 1
        temp[j] = np.maximum(temp[j], 1)

      filter1, kernel1, stride1 = temp[0], kernel_sizes_map[temp[1]], strides_map[temp[2]]
      filter2, kernel2, stride2 =  temp[3], kernel_sizes_map[temp[4]], strides_map[temp[5]]
      filter3, kernel3, stride3 =  temp[6], kernel_sizes_map[temp[7]], strides_map[temp[8]]
      keyword = [filter1, kernel1, stride1, filter2, kernel2, stride2, filter3, kernel3, stride3]
      keyword = list(map(int, keyword))
      
      returns[:, i] = self._raw_dict[str(keyword)] - flag*10

      if self.hasConstraint:
        weights = (filter1 + 1) * (kernel1**2)
        weights += (filter2 + 1) * (kernel2**2)
        weights += (filter3 + 1) * (kernel3**2)
        
        t_in, f_in = 99, 40
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

      if str(keyword) not in self._used_models:
        self._used_models.append(str(keyword))
      if t==self.num_iterations-1 and i>=self.batch_size-5:
        print ('converges at:', [temp[0], kernel_sizes_map[temp[1]], strides_map[temp[2]],\
                                temp[3], kernel_sizes_map[temp[4]], strides_map[temp[5]],\
                                temp[6], kernel_sizes_map[temp[7]], strides_map[temp[8]]])
      #print np.mean(losses), np.mean(returns)
    return actions, con_actions, np.transpose(returns), np.transpose(losses)


  def train(self):
    self.baseline = -1000.0
  
    for t in range(self.num_iterations):
      #print ('iterations:', t)
      actions, con_actions, returns, losses = self.sample_model_reward_return(t)
      returns = returns * 2
      #self.baseline = (t*self.baseline + np.mean(returns)) / (t+1)
      if self.baseline == -1000.0:
        self.baseline = np.mean(returns)
      else:
        self.baseline = 0.6 * self.baseline + 0.4 * np.mean(returns)

      self.sess.run(self.train_op, feed_dict={ 
                    self.action_placeholder : actions,
                    self.con_action_placeholder: con_actions, 
                    self.advantage_placeholder : returns - self.baseline})
      
      avg_acc = np.mean(returns)
      used = len(self._used_models)
      self._num_used_models.append(used)


      self.log_acc.append(avg_acc)
      #sigma_reward = np.sqrt(np.var(returns) / len(total_rewards))
      msg = "Average accuracy within a batch: {:04.2f}".format(avg_acc)
      self.logger.info(msg)
      #print (actions)

  
    self.logger.info("- Training done.")
    export_plot(self.log_acc, "Score", 'NAS-CNN', "./batch_accuracy.png")
    export_plot(self._num_used_models, "Number of distinct models sampled", 'NAS-CNN', "./used_models.png")



  def run(self):
    self.initialize()
    self.train()



if __name__ == '__main__':
    model = PG()
    #model.analyze_data()
    model.run()