# Written by Jonathan Jouty
#
# Based on code from MLPR Lab 09b:
# > https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2016-7/coursework3-4/notebooks/09b_Music_genre_classification_with_the_Million_Song_Dataset.ipynb
#

import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider

# TODO uncomment these when running on GCloud
#train_data = MSD10GenreDataProvider('train', batch_size=50, gcloud='gs://your-bucket/path-to-data')
#valid_data = MSD10GenreDataProvider('valid', batch_size=50, gcloud='gs://your-bucket/path-to-data')
train_data = MSD10GenreDataProvider('train', batch_size=50)
valid_data = MSD10GenreDataProvider('valid', batch_size=50)

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def run_training():

  inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
  targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
  num_hidden = 200

  with tf.name_scope('fc-layer-1'):
      hidden_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], num_hidden)
  with tf.name_scope('output-layer'):
      outputs = fully_connected_layer(hidden_1, num_hidden, train_data.num_classes, tf.identity)

  with tf.name_scope('error'):
      error = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
  with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(
              tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
              tf.float32))

  with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer().minimize(error)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
      sess.run(init)
      for e in range(10):
          running_error = 0.
          running_accuracy = 0.
          for input_batch, target_batch in train_data:
              _, batch_error, batch_acc = sess.run(
                  [train_step, error, accuracy],
                  feed_dict={inputs: input_batch, targets: target_batch})
              running_error += batch_error
              running_accuracy += batch_acc
          running_error /= train_data.num_batches
          running_accuracy /= train_data.num_batches
          print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
                .format(e + 1, running_error, running_accuracy))
          if (e + 1) % 5 == 0:
              valid_error = 0.
              valid_accuracy = 0.
              for input_batch, target_batch in valid_data:
                  batch_error, batch_acc = sess.run(
                      [error, accuracy],
                      feed_dict={inputs: input_batch, targets: target_batch})
                  valid_error += batch_error
                  valid_accuracy += batch_acc
              valid_error /= valid_data.num_batches
              valid_accuracy /= valid_data.num_batches
              print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                     .format(valid_error, valid_accuracy))

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
