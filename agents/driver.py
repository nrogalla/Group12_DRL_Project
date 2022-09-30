import numpy as np
import tensorflow as tf
from . import PPO_Agent
from . import Buffer
from .PPO import PPO

class AdversarialDriver(object):
    """Runs the environment adversary and agents to collect episodes."""

    def __init__(self, env, training_epochs, epoch_length, map_size):
            self.env = env
            self.n_training_epochs = training_epochs
            self.protagonist = PPO_Agent.PPO_Agent(number_actions = 4, map_size = map_size, epoch_length = epoch_length)
            self.antagonist = PPO_Agent.PPO_Agent(number_actions = 4, map_size = map_size,epoch_length = epoch_length)
            self.n_steps = epoch_length
            self.memory = Buffer.Buffer()
            self.gamma = 0.95
            self.action_space = 4 # 0: place agent, 1: place goal, 2: place hole, 3: do nothing
            self.adversary = PPO_Agent.PPO_Agent(number_actions = 4,map_size = map_size, epoch_length = epoch_length)
            self.n_tests = 5
            self.map_size = map_size
        

    def adversarial_epoch(self):
        # we initialize the map with a start and goal in case the adversary 
        #misses these actions in the beginning of the training
        self.env.reset()
        done = False
        # loop through all possible coordinates and let network decide on the action
        print('Creating Map')
        while done is False:
            one_hot_map = PPO.one_hot(self.env.map.copy())
            position = 0
            action, probs= self.adversary.get_action(position, one_hot_map)

            value = self.adversary.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])])
            # change coordinate according to chosen action
            _, _, done, _ = self.env.step_adversary(action[0])
            
            state = PPO.one_hot(self.env.map.copy())

            self.memory.storeTransition(state, action[0], 0, value[0][0], probs[0], done, one_hot_map)

        print('Map created. Training protagonist')
        # run protagonist through env and train, the get reward
        PPO_protagonist = PPO(4, self.map_size, self.gamma)
        for n in range(self.n_training_epochs):
            PPO_protagonist.run(self.env, 1 , self.n_steps)
        protagonist_reward = self.get_performance(self.protagonist, self.n_tests)
        protagonist_reward = np.mean(protagonist_reward)

        print('Training Antagonist')
        # run antagonist through env and train
        PPO_antagonist = PPO(4, self.map_size, self.gamma)
        for n in range(self.n_training_epochs):
            PPO_antagonist.run(self.env, 1 , self.n_steps)
        antagonist_reward = self.get_performance(self.antagonist, self.n_tests)
        antagonist_reward = np.max(antagonist_reward)

        print('Training Adversary')
        self.adversary.fit()
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor([position, self.memory.one_hot_maps], training=True)
            v = self.critic([position, self.memory.one_hot_maps],training=True)
            regret = antagonist_reward - protagonist_reward
        self.memory.clear()
        grads1 = tape1.gradient(regret, self.actor.trainable_variables)
        grads2 = tape2.gradient(regret, self.critic.trainable_variables)        

        self.optimizer_actor.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return regret

    def get_performance(self, agent, n_tests):
        total_rewards = []
        buffer = Buffer.Buffer()
        for _ in range(n_tests): 
            
            state = self.env.reset_agent(0)
            n_steps = 0
            done = False
           
            while not done:
                position = state['position'][0][0]* self.env.nrow+ state['position'][0][1]
                one_hot_map = PPO.one_hot(state['map'][0].copy())
                action, _ = agent.get_action(position, one_hot_map)
                n_steps += 1
                state, reward, done, _ = self.env.step(action)
                
            discounted = np.power(self.gamma, n_steps) * reward
            total_rewards.append(discounted)
            buffer.clear()
        return total_rewards

  