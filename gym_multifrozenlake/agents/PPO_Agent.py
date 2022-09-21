import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from Actor import Actor
from Critic import Critic

class PPO_Agent(object):

    def __init__(self, number_actions: int, number_observations: int, alpha: float = 0.001, gamma: float = 0.7, epsilon: float = 0.25):
        self.n_actions = number_actions
        self.n_observations = number_observations
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actor = Actor(number_actions, number_observations, 256, 128, 64)
        self.critic = Critic(number_observations, 256, 128, 64)

    def get_action(self, observation):
        observation = np.array([[observation]])
        action_probs = self.actor(observation)
        action_probs = action_probs.numpy()
        probs = tfp.distributions.Categorical(probs=action_probs[0], dtype=tf.float32)
        action = probs.sample()
        
        return int(action.numpy()), action_probs

    def process_buffer(self, states, actions, rewards, values, dones):
        g = 0
        returns = []

        for i in range(1, len(rewards) + 1):
            delta = rewards[-i] + self.gamma * values[-i + 1] *dones[-i] - values[i]
            g = delta + self.gamma * dones[-i] * g
            returns.insert(0, g + values[-i])

        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv    



    def calculate_loss(self, probs, actions, adv, old_probs, critic_loss):
        
        probability = old_probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        sur1 = []
        sur2 = []
        
        for pb, t, op, a  in zip(probability, adv, old_probs, actions):
                        t =  tf.constant(t)
                        pb = tf.squeeze(pb)
                        op = tf.squeeze(op)
                        ratio = tf.math.divide(pb[a],op[a])
                        s1 = tf.math.multiply(ratio,t)
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon),t)
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - critic_loss + 0.001 * entropy)
        return loss

    def learn(self, states, actions, rewards, values, dones, probs):
        states, actions, returns, advantages = self.process_buffer(states, actions, rewards, values, dones)

        discnt_rewards = tf.reshape(returns, (len(returns),))
        adv = tf.reshape(advantages, (len(advantages),))

        old_p = probs

        old_p = tf.reshape(old_p, (len(old_p),4))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.calculate_loss(p, actions, adv, probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        a_v = self.actor.trainable_variables
        print(a_v)
        c_v = self.critic.trainable_variables
        print(c_v)

        self.optimizer_actor.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
        
