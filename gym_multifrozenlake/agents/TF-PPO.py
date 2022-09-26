import numpy as np
import tensorflow as tf
import gym

import tf_agents
import multifrozenlake
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

class TF_PPO(object):

    def __init__(self, env):

        #env_name = "FrozenLake-v1"
        #env = suite_gym.load(env_name)
        # load env
        self.train_py_env = tf_agents.environments.suite_gym.load(env)
        self.eval_py_env = tf_agents.environments.suite_gym.load(env)
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.fc_layer_params = (32, 64, 64)
        self.actor_layers = [self.dense_layer(num_units) for num_units in self.fc_layer_params]
        self.value_layers = [self.dense_layer(num_units) for num_units in self.fc_layer_params]
        self.actor_output = tf.keras.layers.Dense(4, activation = tf.nn.softmax)
        self.value_output = tf.keras.layers.Dense(1, activation = None)
        self.a_net = sequential.Sequential(self.actor_layers + [self.actor_output])
        self.v_net = sequential.Sequential(self.value_layers + [self.value_output])
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_step_counter = tf.Variable(0)

    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def collect_trajectories(self, env):
        pass # not implemented
    
    def test_run(self):
        agent = ppo_agent.PPOAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            optimizer=self.optimizer,
            actor_net = self.a_net,
            value_net = self.v_net,
            train_step_counter=self.train_step_counter)
        agent.initialize()

if __name__=="__main__":
    env = multifrozenlake.MultiFrozenLakeEnv(
        n_agents = 1)
    env.reset()
    #a = TF_PPO(env)
    #a.test_run()




