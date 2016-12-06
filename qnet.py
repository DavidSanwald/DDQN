import keras
import numpy as np
from keras import initializations
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop

from utils import prep_batch, prep_input


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.0001, name=name)


class NN:
    def __init__(self, n_states, n_actions, size_hidden, learning_rate,
                 activation):
        self.learning_rate = learning_rate
        self.act = activation
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = self._make_model(n_states, n_actions, size_hidden)
        self.model_t = self._make_model(n_states, n_actions, size_hidden)

    def _make_model(self, n_states, n_actions, size_hidden):
        model = Sequential()
        model.add(Dense(size_hidden, input_dim=4, activation=self.act))
        #model.add(Dense(128, activation='tanh', bias=True))
        #model.add(Dense(128, activation='tanh', bias=True))
        #model.add(Dense(128, activation='tanh', bias=True))
        #model.add(Dense(size_hidden,
        #                init='he_normal',
        #                activation=self.act,
        #            bias=True))
        model.add(Dense(size_hidden, activation=self.act))
        model.add(Dense(size_hidden, activation=self.act))
        #model.add(Dense(size_hidden, activation=self.act))
        #model.add(Dense(size_hidden, activation='tanh', bias=True))
        model.add(Dense(n_actions, activation='linear'))
        #opt = Adam(lr=0.08,
        #           beta_1=0.9,
        #           beta_2=0.999,
        #           epsilon=1e-08,
        #           decay=1e-4)
        opt = SGD(lr=self.learning_rate,
                  decay=1e-5,
                  momentum=0.5,
                  nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        loss = self.model.train_on_batch(X, y)

        return loss

    def predict(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        if usetarget:
            q_vals = self.model_t.predict(state)
        else:
            q_vals = self.model.predict(state)
        return q_vals[0]

    def update_target(self):
        weights = self.model.get_weights()
        self.model_t.set_weights(weights)
        self.save('weights.h5')
        pass

    def best_action(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        q_vals = self.predict(state, usetarget)
        best_action = np.argmax(q_vals)
        return best_action

    def save(self, fname):
        self.model.save_weights(fname, overwrite=True)
        pass

    def load(self, fname):
        self.model.load_weights(fname)
        self.update()
        pass
