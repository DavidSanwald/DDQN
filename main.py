from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

if __name__ == "__main__":
    import os
    import shutil
    import gym
    import agent
    import experiment
    import observer
    import tensorflow as tf
    from parameters import *
    import gym.wrappers
    test = 'test'
    experiment_dir = os.path.abspath("./experiments/{}".format(test))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    try:
        shutil.rmtree(LOGS_DIR)
    except (FileNotFoundError):
        pass
    new_graph = tf.Graph()

    with tf.Session(graph=new_graph) as sess:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #global_step = tf.contrib.framework.get_or_create_global_step(
        #    graph=new_graph)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        key1 = 'CartPole-v0'
        key2 = 'LunarLander-v2'
        key3 = 'Pong-v0'
        env = gym.make(key3)
        env = gym.wrappers.Monitor('./monitored', force=True)(env)
        exp = experiment.Experiment(env, sess, checkpoint_path)
        agent = agent.DQNAgent(sess, env)
        epsilon = observer.EpsilonUpdater(agent)
        agent.add_observer(epsilon)
        exp.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if latest_checkpoint:
            print(latest_checkpoint)
            print("restoring")
            test = tf.train.get_checkpoint_state(checkpoint_dir)
            #saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')

            exp.saver.restore(sess, latest_checkpoint)

        #exp.env.monitor.start('/tmp/cartpole-experiment-1')
        exp.run_experiment(agent)
        env.close()
        #exp.env.monitor.close()

        #epsilon = observer.EpsilonUpdater(agent)
        #agent.add_observer(epsilon)
