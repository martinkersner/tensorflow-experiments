import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

def fetch_batch(data, labels, epoch, batch_index, batch_size):
  start_idx = batch_index * batch_size
  stop_idx  = data_start_idx + batch_size

  return data[start_idx:stop_idx, ], labels[start_idx:stop_idx, ]

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data = housing.data
scaled_housing_data = (housing_data - housing_data.mean(axis=0))/housing_data.std(axis=0)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs      = 1000
learning_rate = 0.01
batch_size    = 100
n_batches     = int(np.ceil(batch_size/m))

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

#gradients = 2.0/m * tf.matmul(tf.transpose(X), error) # manual gradient computation
gradients = tf.gradients(mse, [theta])[0] # autodiff

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
#training_op = tf.assign(theta, theta - learning_rate * gradients) # manual optimization

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    for batch_index in range(n_batches):
      X_batch, y_batch = fetch_batch(X, y, epoch, batch_index, batch_size)
      if epoch % 100 == 0:
        print("Epoch", epoch, "MSE =", mse.eval())
      sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})

  best_theta = theta.eval()

print(best_theta)
