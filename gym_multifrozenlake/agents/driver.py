import numpy as np
import tensorflow as tf
import PPO_Agent
import Buffer
from PPO import PPO

import sys
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake\\envs')
#import envs.adversarial
from adversarial import ReparameterizedAdversarialEnv

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
            #position = self.env.agent_pos[0][0]* self.env.nrow+ self.env.agent_pos[0][1]
            position = 0
            action, probs= self.adversary.get_action(position, one_hot_map)
            

            value = self.adversary.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])])
            # change coordinate according to chosen action
            _, _, done, _ = self.env.step_adversary(action)
            
            state = PPO.one_hot(self.env.map.copy())

            self.memory.storeTransition(state, action, 0, value[0][0], probs[0], done, one_hot_map)

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
            PPO_antagonist.run(self.env, 1 , self.n_steps)#self.train_agent(env, self.antagonist, self.n_steps)
        antagonist_reward = self.get_performance(self.antagonist, self.n_tests)
        antagonist_reward = np.max(antagonist_reward)

        # calculate regret
        regret = antagonist_reward - protagonist_reward

        # use last regret as reward and put in last place of memory before calculating
        # self.memory.rewards[-1] = regret
        # self.memory.calculate_advantage
        # self.memory.calculate_advantage
        # # and learn
        # actor_loss, critic_loss = self.agent.learn(self.memory)

        print('Training Adversary')
        self.adversary.fit()
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            #p = self.adversary.actor([self.memory.position, self.memory.one_hot_maps], training=True)
            #v = self.adversary.critic([self.memory.position, self.memory.one_hot_maps],training=True)
            #v = tf.reshape(v, (len(v),))
            regret = antagonist_reward - protagonist_reward
            #c_loss = 0.5 * kls.mean_squared_error(regret,v)
            #a_loss, total_loss = self.calculate_loss(p, memory.actions, memory.advantage, self.old_probs, c_loss)
        
        grads1 = tape1.gradient(regret, self.actor.trainable_variables)
        grads2 = tape2.gradient(regret, self.critic.trainable_variables)        

        self.optimizer_actor.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return regret#, total_loss


    def choose_action(self, action):
        # 0: place start, 1: place goal, 2: place hole, 3: do nothing
        #if action == 0:
         #   return 'S'
        if action == 0:
            return 'G'
        if action == 2:
            return 'H'
        if action == 3:
            return 'F'
    '''
    def train_agent(self, env, agent, episode_length: int):
        # trains the agent one
        done = False
        self.memory.clear
        state = env.reset_agent(0)
            
        position = state['position'][0][0]* env.nrow+ state['position'][0][1]
        c = 0
        episode_values = []
        episode_dones = []
        episode_rewards = []
        while c <= episode_length:
            
            #ohs = one_hot_states[position]
            one_hot_map = self.one_hot(state['map'][0])
            
            action, probs = self.agent.get_action(position, one_hot_map)
            value = self.agent.critic([np.array([position], dtype= np.int32),np.array([one_hot_map])]).numpy()
            print("value")
            print(value)
            episode_values.append(value[0][0])
            next_position, reward, done, _ = env.step(action)
            episode_dones.append(done)
            episode_rewards.append(reward[0])
            self.buffer.storeTransition(position, action, reward[0], value[0][0], probs[0], done, one_hot_map)
            
            position = next_position['position'][0][0]* env.nrow+ next_position['position'][0][1]
            if done:
                state = env.reset_agent(0)
            
                position = state['position'][0][0]* env.nrow+ state['position'][0][1]
                
                #ohs = one_hot_states[position]
                one_hot_map = self.one_hot(state['map'][0])
                value = self.agent.critic([np.array([position], dtype= np.int32),np.array([one_hot_map])]).numpy()
                print("value")
                print(value)
                episode_values.append(value[0][0])
                c +=1 
                d_returns = self.buffer.calculate_disc_returns(episode_rewards, self.gamma)
                adv = self.buffer.calculate_advantage(episode_rewards, episode_values, episode_dones, self.gamma)
                episode_values = []
                episode_dones = []
                episode_rewards = []
            
        
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer)
        
    '''

    def get_performance(self, agent, n_tests):
        n_tiles = (self.env.nrow) * (self.env.ncol)
        total_rewards = []
        buffer = Buffer.Buffer()
        for i in range(n_tests): 
            
            state = self.env.reset_agent(0)
            n_steps = 0
            done = False
            rewards = []
            while not done:
                position = state['position'][0][0]* self.env.nrow+ state['position'][0][1]
                one_hot_map = self.one_hot(state['map'][0])
                action, _ = agent.get_action(position, one_hot_map)
                n_steps += 1
                state, reward, done, _ = self.env.step(action)
                #rewards.append(reward)
            #discounted = buffer.calculate_disc_returns(self, rewards, self.gamma)
            discounted = np.power(self.gamma, n_steps) * reward
            total_rewards.append(discounted)
            buffer.clear()

            # as a reward of 1 for every done would be difficult, 
            # we chose to take the steps relative to the size of the map as reward, if the agent was successful
            #total_rewards.append(done * (n_tiles-n_steps) / n_tiles)
        return total_rewards
if __name__=="__main__":
    env = ReparameterizedAdversarialEnv(n_holes = 1, size = 2, render_mode = "human", agent_view_size = 2)
    env.reset()
    ad = AdversarialDriver(env, 20, 10, 2)
    ad.adversarial_epoch()
  