from collections import deque

import numpy as np

import tensorflow as tf


class AtariEnvironment:
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """

    def __init__(self, gym_env, action_repeat, sess):
        self.sess = sess
        self.env = gym_env
        self.action_repeat = 4
        self.frame_placeholder = tf.placeholder(shape=[210, 160, 3],
                                                dtype=tf.uint8)
        self.process_op = self._image_process(self.frame_placeholder)

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.state_buffer = deque([], maxlen=4)

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        single_frame = self.env.reset()
        single_frame = self.get_preprocessed_frame(single_frame)
        print(single_frame)
        #state = np.stack([single_frame for i in range(self.action_repeat)],
        #                 axis=0)
        for i in range(self.action_repeat):
            self.state_buffer.append(single_frame)
        state = np.asarray(self.state_buffer)
        return state

    def get_preprocessed_frame(self, observation):
        feed_dict = {self.frame_placeholder: observation}
        processed_image = self.sess.run([self.process_op], feed_dict)
        return processed_image

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        single_frame, r, done, info = self.env.step(self.gym_actions[
            action_index])
        single_frame = self.get_preprocessed_frame(single_frame)
        self.state_buffer.append(single_frame)

        state = np.array(self.state_buffer)
        #print(state.shape())

        return s, r, done, info

    def _image_process(self, raw_input):
        with tf.variable_scope("image_processing"):
            output = tf.image.rgb_to_grayscale(raw_input)
            output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
            output = tf.image.resize_images(
                output, [84, 84],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.squeeze(output)
            print(output.get_shape())
        return output
