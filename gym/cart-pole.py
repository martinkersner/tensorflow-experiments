#/usr/bin/env python

import gym

#env = gym.make("CartPole-v0")
#obs = env.reset()
#print(obs)
## env.render()
#action = 1
#obs, reward, done, info = env.step(action)

def dicount_rewards(rewards, discount_rate):
  dicounted_rewards = np.empty(len(rewards))
  cumulative_rewards = 0

  for step in reversed(range(len(rewards))):
    cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
    discounted_rewards[step] = cumulative_rewards

  return cumulative_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
  all_discounted_rewards = [discount_rewards(rewards) for rewards in all_rewards]
  flat_rewards = np.concatenate(all_discounted_rewards)
  reward_mean = flat_rewards.mean()
  rewards_std = flat_rewards.std()

  return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs  = 4
n_hidden  = 4
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1.0 - tf.to_float(action)

learning_rate = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed   = []

for grad, variable, in grads_and_vars:
  gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
  gradient_placeholders.append(gradient_placeholder)
  grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver= tf.train.Saver()
