

from PPO_Agent import PPO_Agent
from Buffer import Buffer
import numpy as np
import gym
import sys
import tensorflow as tf
#import os

#sys.path.insert(0, '..'))
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake')
import multifrozenlake 
import sys
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake\\envs')
#import envs.adversarial
from adversarial import ReparameterizedAdversarialEnv

class PPO(object):
    def __init__(self, action_space, map_size, gamma: float = 0.95,):

        self.agent = PPO_Agent(action_space, map_size, 128)
        self.buffer = Buffer()
        self.episode_reward = []
        self.total_average = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []
        self.gamma = 0.95
        self.map_size = map_size
    '''
    def test_reward(self, env):
        total_reward = 0
        indices = [i for i in range(4)]
        depth = 4
        one_hot_states = tf.one_hot(indices, depth)
        
        state = env.reset_agent(0)
        position = state['position'][0][0]* env.nrow+ state['position'][0][1]
        ohs = one_hot_states[position]
        one_hot_map = self.one_hot(state['map'][0])
        done = False
        while not done:
            ohs = one_hot_states[position]
            action = np.argmax(self.agent.actor([np.array([[ohs]], dtype= np.int32),np.array([one_hot_map])]).numpy())
            next_position, reward, done, _ = env.step([action])
            position = next_position['position'][0][0]* env.nrow+ next_position['position'][0][1]
            one_hot_map = self.one_hot(next_position['map'][0])
            total_reward += reward[0]
        return total_reward
    '''
    @staticmethod
    def one_hot(map):
        ohm = [x[:] for x in map]
        for i in range(len(map)):
            for j in range(len(map)):
                if map[i][j] == 'G':
                    ohm[i][j] = [0, 0, 1]
                if map[i][j] == 'H':
                    ohm[i][j] = [0, 1, 0]
                if map[i][j] == 'F':
                    ohm[i][j] = [1, 0, 0]
                if map[i][j] == None:
                    ohm[i][j] = [0, 0, 0]
        return ohm

    def run(self, env, episode_number: int, episode_length: int):
        
        target = False
        best_reward = 0
        counter = 0
        #indices = [i for i in range(4)]
        depth = 4
        #one_hot_states = tf.one_hot(indices, depth)
        
        while counter <= episode_number:
            if target == True:
                break
            done = False
            self.buffer.clear()
            
            state = env.reset_agent(0)
            
            position = state['position'][0][0]* env.nrow+ state['position'][0][1]
            print("pos")
            print(position)
            c = 0
            episode_values = []
            episode_dones = []
            episode_rewards = []
            while c <= episode_length:
                
                #ohs = one_hot_states[position]
                one_hot_map = PPO.one_hot(state['map'][0])
                
                action, probs = self.agent.get_action(position, one_hot_map)
                value = self.agent.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])]).numpy()
                print("value")
                print(value)
                episode_values.append(value[0][0])
                next_position, reward, done, _ = env.step(action)
                episode_dones.append(done)
                episode_rewards.append(reward[0])
                self.buffer.storeTransition([position], action, reward[0], value[0][0], probs[0], done, one_hot_map)
                
                position = next_position['position'][0][0]* env.nrow+ next_position['position'][0][1]
                if done:
                    state = env.reset_agent(0)
                
                    position = state['position'][0][0]* env.nrow+ state['position'][0][1]
                    
                    #ohs = one_hot_states[position]
                    one_hot_map = PPO.one_hot(state['map'][0])
                    value = self.agent.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])]).numpy()
                    print("value")
                    print(value)
                    episode_values.append(value[0][0])
                    c +=1 
                    d_returns = self.buffer.calculate_disc_returns(episode_rewards, self.gamma)
                    adv = self.buffer.calculate_advantage(episode_rewards, episode_values, episode_dones, self.gamma)
                    episode_values = []
                    episode_dones = []
                    episode_rewards = []    

            print('Training')
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer)
        
            #rewards = [self.test_reward(env) for _ in range(10)]
            #avg_reward = self.test_reward(env)#np.mean(rewards)
            avg_reward = 0
            best_reward = 0
            print('Episode: ' + str(counter))
            print(f"total test reward is {avg_reward}")
            if avg_reward > best_reward:
                print('best reward=' + str(avg_reward))
                algo.agent.actor.save('model_actor_{}_{}'.format(counter, avg_reward), save_format="tf")
                algo.agent.critic.save('model_critic_{}_{}'.format(counter, avg_reward), save_format="tf")
                best_reward = avg_reward
            if best_reward*100 == 75:
                target = True
            env.reset_agent(0)
            state = env.agent_pos[0] #env.reset()
            position = state[0]* env.nrow+ state[1]
            for i in range(4):#env.position_space.n):
                print('probs for state:' + str(i))
                print(self.agent.actor.predict([np.array([[position]], dtype= np.int32),np.array([one_hot_map])]))
            counter+=1
        env.close()

if __name__=="__main__":
    env = ReparameterizedAdversarialEnv(n_holes = 3, size = 2, render_mode = "human", agent_view_size = 2, max_steps = 7)
    env.reset()
  
    map, time,done, inf =env.step_adversary(0)
    map, time,done, inf = env.step_adversary(1)
    map, time,done, inf = env.step_adversary(3)
   
    map, time,done, inf = env.step_adversary(2)
    
  
    #while not done:
    # map, time,done, inf = env.step_adversary(np.random.randint(2,4))
    

    #env = gym.make("FrozenLake-v1", desc=["SFFF", "FHFH", "FFFH", "HFFG"], is_slippery=False)

    algo = PPO(4, 2)#env.position_space.n)
    algo.run(env, 40, 8)



