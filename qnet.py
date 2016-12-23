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
        with tf.variable_scope('online_NN') as scope:
            self.inputs, self.outputs = build_model(
                self.activation, size_hidden, n_states[0], n_actions)
            self.targets = tf.placeholder(tf.float32)
            self.loss = build_loss(self.targets, self.outputs)
            self.online_network_params = tf.get_collection(
                tf.GraphKeys.VARIABLES, scope=scope.name)
            self.train_op = build_training(self.loss, self.learningrate, 0.5)
        with tf.variable_scope('target_NN') as scope:
            self.inputs_t, self.outputs_t = build_model(
                self.activation, size_hidden, n_states[0], n_actions)
            self.target_network_params = tf.get_collection(
                tf.GraphKeys.VARIABLES, scope=scope.name)
        self.update_target_network_params_op = [
            self.target_network_params[i].assign(self.online_network_params[i])
            for i in range(len(self.target_network_params))
        ]

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        loss = self.sess.run([self.loss, self.train_op],
                             feed_dict={self.inputs: X,
                                        self.targets: y})
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
        self.sess.run(self.update_target_network_params_op)
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


def build_model(act, size_hidden, n_states, n_actions):
    input_layer = tflearn.input_data(shape=[None, n_states], name='input')
    net = tflearn.fully_connected(
        input_layer,
        size_hidden,
        activation=act,
        name='l1',
        weights_init='Xavier',
        bias_init='zeros')
    net = tflearn.fully_connected(
        net,
        size_hidden,
        activation=act,
        name='l2',
        weights_init='Xavier',
        bias_init='zeros')
    #    net = tflearn.fully_connected(
    #        net,
    #        size_hidden,
    #        activation=act,
    #        name='l3',
    #        weights_init='Xavier',
    #        bias_init='zeros')
    #net = tflearn.fully_connected(
    #    net, size_hidden, activation=act, name='l3')
    #net = tflearn.fully_connected(
    #    net, size_hidden, activation=act, name='l4')
    #net = tflearn.fully_connected(
    #    net, size_hidden, activation=act, name='l4')
    #net = tflearn.fully_connected(net, 64, activation=act)
    #net = tflearn.fully_connected(net, 32, activation=act)
    output_layer = tflearn.fully_connected(
        net, n_actions, activation='linear', name='output')
    return input_layer, output_layer


def build_loss(X, y):
    loss = tflearn.mean_square(X, y)
    return loss


def build_training(loss, learningrate, momentum):
    optimizer = tf.train.MomentumOptimizer(learningrate, momentum)
    training_op = optimizer.minimize(loss, name="minimize_loss")
    return training_op
