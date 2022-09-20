import numpy as np
import tensorflow as tf
import tensorflow_probabilty as tfp
import Actor
import Critic

class PPO_Agent(object):

    def __init__(number_actions: int, number_observations: int, alpha: float = 0.001, gamma: float = 0.7, epsilon: float = 0.25):
        self.n_actions = number_actions
        self.n_observations = number_observations
        self.learning_rate = alpha
        self.gamma = gamma
        self.actor = Actor(number_actions, number_observations, 256, 128, 64)
        self.critic = Critic(number_observations, 256, 128, 64)

    def get_action(self, observation):
        
        observation = np.array([observation])
        action_probs = self.actor(observation)
        action_probs = action_props.numpy()
        probs = tf.tfp.distributions.Categorical(probs=actions_probs, dtype=tf.float32)
        action = probs.sample()
        
        return int(action.numpy()[0]), probs

    def calculate_loss(self, probs, actions, advantage, old_probs, critic_loss):
        pass

    def learn(self, states, actions,  advantages, old_probs, discnt_rewards):
        pass
