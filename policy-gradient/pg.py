# -*- coding: UTF-8 -*-

import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
#import gym
import scipy.signal
import os
import time
import inspect
#from utils.general import get_logger, Progbar, export_plot
from pg_config import pg_config
#import logz
from config import *




class PG(object):

  def __init__(self, pg_config, logger=None):
    if not os.path.exists(pg_config.output_path):
      os.makedirs(pg_config.output_path)
    self.pg_config = pg_config
    # self.logger = logger
    # if logger is None:
    #   self.logger = get_logger(pg_config.log_path)
    self.observation_dim = num_layers  #####len(LAYER_SIZES)
    self.action_dim = len(LAYER_SIZES)
    self.lr = self.pg_config.learning_rate

    self.controller_cells = controller_cells
    self.num_layers = NUM_LAYERS
    self.build()


  def build_rnn(scope):
    with tf.variable_scope(scope):
      nas_cell = tf.contrib.rnn.NASCell(self.controller_cells)
      cell_state = nas_cell.zero_state(batch_size=1, dtype=tf.float32)
      cell_input = tf.ones([1, self.action_dim]) / self.action_dim
      output = []

      for i in range(self.num_layers):
        with tf.name_scope('rnn_output_%d' % i):
          outputs, _ = tf.nn.dynamic_rnn(nas_cell, cell_input, dtype=tf.float32)
          classifier = tf.layers.dense(outputs[:, -1, :], units=self.action_dim, name='classifier_%d' % (i))
          preds = tf.nn.softmax(classifier)
          cell_input = tf.expand_dims(preds, 0, name='cell_output_%d' % (i))
        output = tf.concat([output, cell_input], 0)
    return output


  def add_placeholders_op(self):
    self.observation_placeholder = tf.placeholder(tf.float32, [None, self.observation_dim]) #####
    self.action_placeholder = tf.placeholder(tf.int32, [None, num_layers])
    self.advantage_placeholder = tf.placeholder(tf.float32, [None,])


  def build_policy_network_op(self, scope="policy_network"):
    actions_taken = self.action_placeholder
    # action_logits = build_rnn(scope)
    # self.sampled_action = tf.squeeze(tf.multinomial(action_logits, num_samples=1), axis=1)
    # self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_taken, logits=action_logits)

    action_means = build_rnn(scope)
    log_std = tf.get_variable('log_std', self.num_layers, tf.float32)
    self.sampled_action = tf.random_normal((pg_config.batch_size, self.num_layers), action_means, tf.exp(log_std))
    self.logprob = tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std)).log_prob(actions_taken)


  def add_loss_op(self):
    self.loss = -tf.reduce_mean(self.logprob * self.advantage_placeholder)


  def add_optimizer_op(self):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


  def build(self):
    self.add_placeholders_op()
    self.build_policy_network_op()
    self.add_loss_op()
    self.add_optimizer_op()



  def initialize(self):
    self.sess = tf.Session()
    self.add_summary()
    init = tf.global_variables_initializer()
    self.sess.run(init)


  def add_summary(self):
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.pg_config.output_path,self.sess.graph)


  def init_averages(self):
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.


  def update_averages(self, rewards, scores_eval):
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]


  def record_summary(self, t):
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    self.file_writer.add_summary(summary, t)


  # sample_path and get_returns in assignment 3
  def sample_model_reward_return(self):
    action_average_reward_dict = {}
    with open(action_average_reward_dict_name, 'r') as f:
      action_average_reward_dict = json.load(f)
    f.close()

    models_rewards = []
    returns = []
    for key in action_average_reward_dict.keys():
      actions = [int(a) for a in key[1:-1].split(',')]
      accuracy = float(action_average_reward_dict[key]) / JSON_SCALE
      sample = {"observation": np.array(actions), "action": np.array(actions), "reward": accuracy}
      models_rewards.append(sample)
      returns.append(accuracy)
    return models_rewards, returns


  def calculate_advantage(self, returns, observations):
    adv = returns
    # if self.pg_config.use_baseline:
    #   temp = self.sess.run(self.baseline, feed_dict={self.observation_placeholder: observations})
    #   adv = adv - np.squeeze(temp)
    if self.pg_config.normalize_advantage:
      adv = (adv - np.mean(adv)) / np.std(adv)
    return adv


  def train(self):
    last_eval = 0 
    last_record = 0
    self.init_averages()
    scores_eval = []
  
    for t in range(self.pg_config.num_batches):
      paths, total_rewards = self.sample_model_reward_return()
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      returns = total_rewards
      advantages = self.calculate_advantage(returns, observations)

      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages})

      if (t % self.pg_config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)
  
    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", pg_config.env_name, self.pg_config.plot_output)


  def run(self):
    self.initialize()
    self.train()



if __name__ == '__main__':
    model = PG(pg_config)
    model.run()
