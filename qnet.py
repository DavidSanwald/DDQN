import numpy as np

import tensorflow as tf
from parameters import *
from tensorflow.contrib import layers, learn, losses
from utils import prep_batch, prep_input


class Networks:
    def __init__(self, act, n_states, n_actions, size_hidden, momentum,
                 learningrate, sess):
        self.Q_o = QModel("online", act, n_states, n_actions, size_hidden,
                          momentum, learningrate)
        self.Q_t = QModel("target", act, n_states, n_actions, size_hidden,
                          momentum, learningrate)
        self.n_states = n_states
        self.sess = sess
        self.update_op = self._build_update_op_soft(self.Q_t.var_list,
                                                    self.Q_o.var_list)
        #self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #self.predictions = tf.summary.histogram("q_vals", self.Q_o.outputs)
        self.summary_ops_scalar = self._build_summaries_scalar()
        #self.summary_ops_histo = self._build_summaries_histo()

        self.writer = tf.summary.FileWriter(LOGS_DIR, self.sess.graph)
        pass

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        feed_dict = {self.Q_o.inputs: X, self.Q_o.q_targets: y}
        _, test = self.sess.run([self.Q_o.train_op, self.summary_ops_scalar],
                                feed_dict)
        step = tf.contrib.framework.get_global_step().eval()
        self.writer.add_summary(test, step)
        pass

    def predict(self, state, usetarget):
        state = prep_input(state.flatten(), self.n_states)
        if usetarget:
            q_vals = self.sess.run(self.Q_t.outputs,
                                   feed_dict={self.Q_t.inputs: state})
            summary_str = ""
        else:
            q_vals, step = self.sess.run([
                self.Q_o.outputs, tf.train.get_global_step()
            ],
                                         feed_dict={self.Q_o.inputs: state})
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


class QModel:
    def __init__(self, scope, act, n_states, n_actions, size_hidden, momentum,
                 learningrate):
        self.scope = scope
        with tf.variable_scope(scope) as scope:
            self.inputs, self.outputs = self._build_model(act, size_hidden,
                                                          n_states, n_actions)
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope.name)
        if self.scope == "online":
            self.q_targets = tf.placeholder(tf.float32,
                                            shape=[None, n_actions],
                                            name="q_targets")
            with tf.name_scope('train'):
                self.loss = self._build_loss(self.outputs, self.q_targets)

                self.train_op = self._build_train_op(self.loss, learningrate,
                                                     momentum, self.var_list)

    def _build_model(self, act, size_hidden, n_states, n_actions):
        w_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        b_initializer = tf.constant_initializer(0.01)
        activation = tf.nn.relu
        input_ph = tf.placeholder(shape=[None, n_states],
                                  dtype=tf.float32,
                                  name="State")
        fc1 = tf.contrib.layers.fully_connected(
            inputs=input_ph,
            num_outputs=size_hidden,
            weights_initializer=w_initializer,
            biases_initializer=b_initializer,
            activation_fn=activation)
        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=size_hidden,
            weights_initializer=w_initializer,
            biases_initializer=b_initializer,
            activation_fn=activation)
        #    fc3 = tf.contrib.layers.fully_connected(
        #        inputs=fc2,
        #        num_outputs=size_hidden,
        #        weights_initializer=initializer,
        #        activation_fn=activation)
        #    fc4 = tf.contrib.layers.fully_connected(
        #        inputs=fc3,
        #        num_outputs=size_hidden,
        #        weights_initializer=initializer,
        #        activation_fn=activation)
        predictions = tf.contrib.layers.fully_connected(inputs=fc2,
                                                        num_outputs=n_actions)

        return input_ph, predictions

    def _build_loss(self, outputs, targets):
        #loss = tf.contrib.losses.mean_squared_error(outputs, targets)
        # Huber loss
        x = outputs - targets
        clipped_error = tf.select(
            tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        loss = tf.reduce_mean(clipped_error, name='loss')
        #assert_op = tf.is_numeric_tensor(loss)
        #out = tf.with_dependencies([assert_op], out)
        #with tf.control_dependencies([assert_op]):
        #    loss = tf.identity(loss, name='loss')

        return loss

    def _build_train_op(self, loss, learningrate, momentum, var_list):
        optimizer = tf.train.AdamOptimizer(learningrate)
        step = tf.contrib.framework.get_global_step()
        train_op = optimizer.minimize(
            loss,
            name="minimize_loss",
            global_step=tf.contrib.framework.get_global_step())
        return train_op
