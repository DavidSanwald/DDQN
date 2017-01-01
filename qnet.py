import numpy as np

import tensorflow as tf
from parameters import *
from tensorflow.contrib import layers, learn, losses
from utils import prep_batch, prep_input


def _forward_pass(state_input):
    conv1 = tf.contrib.layers.conv2d(state_input,
                                     32,
                                     8,
                                     4,
                                     activation_fn=tf.nn.relu)
    conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
    conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

    # Fully connected layers
    flattened = tf.contrib.layers.flatten(conv3)
    fc1 = tf.contrib.layers.fully_connected(flattened, 256)
    predictions = tf.contrib.layers.fully_connected(fc1, 4)
    return predictions


def _image_process(raw_input):
    with tf.variable_scope("image_processing"):
        input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        output = tf.image.rgb_to_grayscale(input_state)
        output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
        output = tf.image.resize_images(
            output,
            84, 84,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.squeeze(output)
    return processed


def _build_loss(outputs, targets):
    # Huber loss
    x = outputs - targets
    clipped_error = tf.select(
        tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    loss = tf.reduce_mean(clipped_error, name='loss')
    return loss


def _build_train_op(loss, learningrate):
    optimizer = tf.train.AdamOptimizer(learningrate)
    train_op = optimizer.minimize(
        loss,
        name="minimize_loss",
        global_step=tf.contrib.framework.get_global_step())
    return train_op


def _build_update_op_soft(self, target_vars, online_vars):
    tau = 1e-2
    update_op = [
        target_vars[i].assign(tf.mul(online_vars[i], tau) + tf.mul(target_vars[
            i], (1 - tau))) for i in range(len(online_vars))
    ]
    return update_op


class Networks:
    def __init__(self, sess):
        self.learningrate = 0.002
        self.sess = sess
        self.inputs = tf.placeholder(tf.float32, [None, 4, 84, 84])
        self.target_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, 4],
                                                 name="q_targets")
        with tf.variable_scope('online') as scope:
            self.online_predictions = _forward_pass(self.inputs)
            self.online_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope.name)
            self.loss = _build_loss(self.online_predictions,
                                    self.target_placeholder)
            self.train_op = _build_train_op(self.loss, self.learningrate)

        with tf.variable_scope('target') as scope:
            self.target_predictions = _forward_pass(self.inputs)
            self.target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope.name)

        self.update_op = self._build_update_op_soft(self.target_vars,
                                                    self.online_vars)
        pass

    def train(self, X, y):
        #X = prep_batch(X)
        #y = prep_batch(y)
        feed_dict = {self.inputs: X, self.target_placeholder: y}
        _, test = self.sess.run([self.train_op, ], feed_dict)
        pass

    def predict(self, state, usetarget):
        #state = prep_input(state.flatten(), self.n_states)
        if usetarget:
            q_vals = self.sess.run(self.online_predictions,
                                   feed_dict={self.inputs: state})
            summary_str = ""
        else:
            q_vals = self.sess.run([
                self.target_predictions
            ], feed_dict={self.inputs: state})
        #    predictions = tf.summary.histogram("q_vals", q_vals[0])
        #self.writer.add_summary(test, self.step)
        return q_vals[0]

    def best_action(self, state, usetarget):
        q_vals = self.predict(state, usetarget)
        step = tf.contrib.framework.get_global_step().eval()
        #self.writer.add_summary(summary_str, step)
        best_action = np.argmax(q_vals)
        max_q = q_vals[best_action]
        return best_action, max_q

    def update_target(self):
        self.sess.run(self.update_op)
        pass

    def process_state(self, state):
        feed_dict = {self.raw_state: state}
        processed_state = self.sess.run([self.process_state], feed_dict)
        return processed_state

    def _build_update_op(self, target_vars, online_vars):
        update_op = [
            target_vars[i].assign(online_vars[i])
            for i in range(len(online_vars))
        ]
        return update_op

    def _build_update_op_soft(self, target_vars, online_vars):
        tau = 1e-2
        update_op = [
            target_vars[i].assign(tf.mul(online_vars[i], tau) + tf.mul(
                target_vars[i], (1 - tau))) for i in range(len(online_vars))
        ]
        return update_op

    def _build_image_processor(self):
        with tf.variable_scope("image_processing"):
            input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            output = tf.image.rgb_to_grayscale(input_state)
            output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
            output = tf.image.resize_images(
                output,
                84,
                84,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.squeeze(output)
        return output

    def _build_summaries_scalar(self):
        #episode_reward = tf.Variable(0.)
        #tf.scalar_summary("Reward", episode_reward)
        #episode_ave_max_q = tf.Variable(0.)
        #tf.scalar_summary("Qmax Value", episode_ave_max_q)
        loss_summary = tf.summary.scalar('loss', self.Q_o.loss)

        #summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge([loss_summary])

        return summary_ops  #, summary_vars

    def _build_summaries_histo(self):
        #episode_reward = tf.Variable(0.)
        #tf.scalar_summary("Reward", episode_reward)
        #episode_ave_max_q = tf.Variable(0.)
        #tf.scalar_summary("Qmax Value", episode_ave_max_q)
        q_vals_histo = tf.summary.histogram("q_vals", self.Q_o.outputs)

        #summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge([q_vals_histo])

        return summary_ops  #, summary_vars
