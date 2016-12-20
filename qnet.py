import numpy as np

import tensorflow as tf
import tflearn
from utils import prep_batch, prep_input


class NN:
    def __init__(self, n_states, n_actions, size_hidden, learningrate,
                 activation, batch_size, sess):
        self.sess = sess
        self.batch_size = batch_size
        self.activation = activation
        self.n_states = n_states
        self.n_actions = n_actions
        self.size_hidden = size_hidden
        self.learningrate = learningrate
        self.inputs, self.outputs = self._make_model(self.activation,
                                                     'online_network')
        self.network_params = tf.trainable_variables()
        self.inputs_t, self.outputs_t = self._make_model(self.activation,
                                                         'target_network')
        self.target_network_params = tf.trainable_variables()[len(
            self.network_params):]
        self.predicted_q_value = tf.placeholder(
            tf.float32,
            [None, 2], )
        self.loss = tf.reduce_mean(
            tf.square(self.outputs - self.predicted_q_value),
            name='loss_function')
        self.opt = tf.train.GradientDescentOptimizer(
            self.learningrate, name='training_online_network')
        self.opt_operation = self.opt.minimize(self.loss)
        self.update_target_network_params = [
            self.target_network_params[i].assign(self.network_params[i])
            for i in range(len(self.target_network_params))
        ]

    def _make_model(self, act, network):
        with tf.variable_scope(network):
            input_layer = tflearn.input_data(
                shape=[None, self.n_states[0]], name='input')
            net = tflearn.fully_connected(
                input_layer, 16, activation=act, name='l1')
            net = tflearn.fully_connected(net, 16, activation=act, name='l2')
            net = tflearn.fully_connected(net, 16, activation=act, name='l3')
            #net = tflearn.fully_connected(net, 64, activation=act)
            #net = tflearn.fully_connected(net, 32, activation=act)
            output_layer = tflearn.fully_connected(
                net, 2, activation='linear', name='output')
        return input_layer, output_layer

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        with tf.name_scope('training_online_network'):
            loss = self.sess.run(
                self.opt_operation,
                feed_dict={self.inputs: X,
                           self.predicted_q_value: y})
        return loss

    def predict(self, state, usetarget=False):
        sstate = prep_input(state.flatten(), 4)
        if usetarget:
            q_vals = self.tf_predict_target(sstate)
        else:
            q_vals = self.tf_predict(sstate)
        return q_vals[0]

    def tf_predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})

    def tf_predict_target(self, inputs):
        return self.sess.run(self.outputs_t, feed_dict={self.inputs_t: inputs})

    def update_target(self):
        print('update')
        self.updateTarget()
        pass

    def best_action(self, state, usetarget=False):
        sstate = prep_input(state.flatten(), 4)
        if usetarget:
            q_vals = self.tf_predict_target(sstate)[0]
        else:
            q_vals = self.tf_predict(sstate)[0]
        best_action = np.argmax(q_vals)
        #print(q_vals)
        max_q = q_vals[best_action]
        return best_action, max_q

    def save(self, fname):
        self.model.save_weights(fname, overwrite=True)
        pass

    def load(self, fname):
        self.model.load_weights(fname)
        self.update()
        pass

    def updateTarget(self):
        self.sess.run(self.update_target_network_params)
