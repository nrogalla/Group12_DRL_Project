import numpy as np
import tensorflow as tf
from .PPO_Agent import PPO_Agent
from .Buffer import Buffer
from .PPO import PPO

class AdversarialDriver(object):
    """Runs the environment adversary and agents to collect episodes."""

    def __init__(self, env, training_epochs, epoch_length, map_size):
            self.env = env
            self.n_training_epochs = training_epochs
            self.protagonist = PPO_Agent(number_actions = 4, map_size = map_size, epoch_length = epoch_length)
            self.antagonist = PPO_Agent(number_actions = 4, map_size = map_size,epoch_length = epoch_length)
            self.n_steps = epoch_length
            self.memory = Buffer()
            self.gamma = 0.95
            self.action_space = 4 # 0: place agent, 1: place goal, 2: place hole, 3: do nothing
            self.adversary = PPO_Agent(number_actions = 4,map_size = map_size, epoch_length = epoch_length)
            self.n_tests = 2
            self.map_size = map_size
        

    def adversarial_epoch(self):
        '''We initialize the map with a start and goal in case the adversary misses these actions in the beginning of the training'''
        self.env.reset()
        done = False
        '''loop through all possible coordinates and let network decide on the action'''
        print('Creating Map')
        while done is False:
            one_hot_map = PPO.one_hot(self.env.map.copy())
            position = 0
            action, probs= self.adversary.get_action(position, one_hot_map)

            value = self.adversary.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])])
            # change coordinate according to chosen action
            _, _, done, _ = self.env.step_adversary(action[0])
            
            self.memory.storeTransition([position], action[0], 0, value[0][0], probs[0], done, one_hot_map)

        print('Map created. Training protagonist')
        # run protagonist through env and train, the get reward
        PPO_protagonist = PPO(4, self.map_size, self.gamma)
        PPO_protagonist.run(self.env, self.n_training_epochs , self.n_steps)
        protagonist_reward = self.get_performance(self.protagonist, self.n_tests)
        protagonist_reward = np.mean(protagonist_reward)
        
        print('Training Antagonist')
        # run antagonist through env and train
        PPO_antagonist = PPO(4, self.map_size, self.gamma)
        PPO_antagonist.run(self.env, self.n_training_epochs , self.n_steps)
        antagonist_reward = self.get_performance(self.antagonist, self.n_tests)
        antagonist_reward = np.max(antagonist_reward)
        
        print('Training Adversary')
        position = np.array(self.memory.position, dtype=np.float32)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.adversary.actor([position, self.memory.one_hot_maps], training=True)
            v = self.adversary.critic([position, self.memory.one_hot_maps],training=True)
            regret = antagonist_reward - protagonist_reward
            regret = tf.convert_to_tensor(regret)
            self.memory.clear()
        grads1 = tape1.gradient(regret, self.adversary.actor.trainable_variables)
        grads2 = tape2.gradient(regret, self.adversary.critic.trainable_variables)        

        self.adversary.optimizer_actor.apply_gradients(zip(grads1, self.adversary.actor.trainable_variables))
        self.adversary.optimizer_critic.apply_gradients(zip(grads2, self.adversary.critic.trainable_variables))
        return protagonist_reward, antagonist_reward, regret

    def get_performance(self, agent, n_tests):
        total_rewards = []
        buffer = Buffer()
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
                
            discounted = np.power(self.gamma, n_steps) * reward[0]
            total_rewards.append(discounted)
            buffer.clear()
        return total_rewards
