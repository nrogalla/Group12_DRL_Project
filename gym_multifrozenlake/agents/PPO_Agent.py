import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from Actor import Actor
from Critic import Critic

class PPO_Agent(object):

    def __init__(self, number_actions: int, map_size: int, alpha: float = 0.0001, gamma: float = 0.95, epsilon: float = 0.25):
        self.n_actions = number_actions
        #self.n_observations = number_observations
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actor = Actor(number_actions, map_size, 16, 32)
        self.critic = Critic(map_size, 16, 32)
        self.old_probs = []

    def get_action(self, observation):
        
        observation = np.array([observation])
        action_probs = self.actor(observation)
        action_probs = action_probs.numpy()
        probs = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
        action = int(probs.sample())
        
        return [action], action_probs

    def process_buffer(self, states, actions, rewards, values, dones):
       # g = 0
       # returns = []

        #for i in range(1, len(rewards) + 1):
        #    delta = rewards[-i] + self.gamma * values[-i + 1] *dones[-i] - values[i]
        #    g = delta + self.gamma * dones[-i] * g
        #    returns.insert(0, g + values[-i])

        #adv = np.array(returns, dtype=np.float32) - values[:-1]
        #adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        #states = np.array(states, dtype=np.float32)
        #actions = np.array(actions, dtype=np.int32)
        #returns = np.array(returns, dtype=np.float32)
        #return states, actions, returns, adv
        '''
        g = 0
        d_returns = []
        batch_size = len(dones)
        for t in reversed(range(batch_size)): 
            ret = 0
            for i in range(0, batch_size-t):
                ret += self.gamma**i * rewards[t+i]
            d_returns.append(ret)
        d_returns.reverse()

        d_returns = (d_returns - np.mean(d_returns)) / (np.std(d_returns) + 1e-10)
        print(d_returns)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - values[i]
            if dones[i] == 1:
                g = delta + self.gamma * 0.95 * 0 * g
            else:
                g = delta + self.gamma * 0.95 * 1 * g
        

        adv = np.array(d_returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        '''

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        #returns = np.array(d_returns, dtype=np.float32)
        return states, actions,    



    def calculate_loss(self, probs, actions, adv, old_probs, critic_loss):
        
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs,tf.math.log(probs))))
        sur1 = []
        sur2 = []
        actor_loss = []
        
        for pb, t, op, a  in zip(probs, adv, np.squeeze(old_probs), actions):
                        t =  tf.constant(t)
                        pb = np.array(pb)
                        tf.stop_gradient(pb)
                        ratio = tf.math.divide(pb[a],op[a])
                        s1 = tf.math.multiply(ratio,t)
                        s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon),t)
                        actor_loss.append(tf.math.minimum(s1,s2))
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) + critic_loss - 0.001 * entropy)
        return actor_loss, loss

    def learn(self, buffer):
        #states, actions, returns, advantages = self.process_buffer(states, actions, rewards, values, dones)
        states = np.array(buffer.states, dtype=np.float32)

        #discnt_rewards = tf.reshape(buffer.returns, (len(returns),))
        #adv = tf.reshape(advantages, (len(advantages),))        

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            c_loss = 0.5 * kls.mean_squared_error(buffer.discounted_returns,v)
            a_loss, total_loss = self.calculate_loss(p, buffer.actions, buffer.advantage, self.old_probs, c_loss)
        
        grads1 = tape1.gradient(total_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
       
        self.old_probs = buffer.probs
        

        self.optimizer_actor.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss#, total_loss
        
