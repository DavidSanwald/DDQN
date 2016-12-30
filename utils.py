from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from PIL import Image


def prep_input(data, n_dimension):
    prep = np.asarray(data)
    transformed = prep.reshape((1, n_dimension))
    return transformed


def prep_batch(to_prep):
    prep = np.vstack(to_prep)
    return prep


def prep_obs(observation):
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')
    prepped_obs = np.array(img)
    return prepped_obs.astype('uint8')


def prep_state(state):
    prepped_state = state.astype('float32') / 255.
    return prepped_state
