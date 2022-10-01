

from .PPO_Agent import PPO_Agent
from .Buffer import Buffer
import numpy as np

class PPO(object):
    def __init__(self, action_space, map_size, gamma: float = 0.95,):

        self.agent = PPO_Agent(action_space, map_size, 128)
        self.buffer = Buffer()
        self.episode_reward = []
        self.total_average = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []
        self.gamma =  gamma
        self.map_size = map_size

    @staticmethod
    def one_hot(map):
        '''a method to one hot the three possible states for cells of a map'''
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

    def run(self, env, episode_number: int = 1, episode_steps: int = 64):
        '''runs a given number of epochs and steps over the given environment, collects trajectories and trains the agent with them default set for one epoch'''
        
        target = False
        best_reward = 0
        counter = 0
        impossible = False
        while counter <= episode_number:
            if target == True:
                break
            done = False
            self.buffer.clear()
            
            state = env.reset_agent(0)
            
            position = state['position'][0][0]* env.nrow+ state['position'][0][1]
            c = 0
            '''lists for values of current episode, cleared when done flag is reached'''
            episode_values = []
            episode_dones = []
            episode_rewards = []
            episode_length = env.ncol * env.nrow * 3 # value chosen for max steps before checking if env is valid

            while c <= episode_steps:
                '''if the agent gets stuck, this is meant to check if the environment is even valid. If not, end this epoch'''
                if episode_length == 0:
                    if env.valid is False:
                        impossible = True
                        print("Environment impossible to solve")
                        break

                '''block for stepping and collecting environment values for one step'''
                one_hot_map = PPO.one_hot(state['map'][0])
                action, probs = self.agent.get_action(position, one_hot_map)
                value = self.agent.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])]).numpy()
                episode_values.append(value[0][0])
                next_position, reward, done, _ = env.step(action)
                episode_dones.append(done)
                episode_rewards.append(reward[0])
                self.buffer.storeTransition([position], action, reward[0], value[0][0], probs[0], done, one_hot_map)
                position = next_position['position'][0][0]* env.nrow+ next_position['position'][0][1]
               
                '''block for collecting and processing values if done flag is reached by agent'''
                if done:
                    state = env.reset_agent(0)
                    
                    position = state['position'][0][0]* env.nrow+ state['position'][0][1]
                    one_hot_map = PPO.one_hot(state['map'][0])
                    value = self.agent.critic([np.array([[position]], dtype= np.int32),np.array([one_hot_map])]).numpy()
                    episode_values.append(value[0][0])
                    c +=1 
                    self.buffer.calculate_disc_returns(episode_rewards, self.gamma)
                    self.buffer.calculate_advantage(episode_rewards, episode_values, episode_dones, self.gamma)

                    '''clearing of episode lists '''
                    episode_values = []
                    episode_dones = []
                    episode_rewards = []    
                
                episode_length -= 1 # counts down as agent moves

            print('Training')
            '''ends epoch if environment is impossible'''
            if impossible is True:
                break
            
            
            '''learn from collected values'''
            self.agent.learn(self.buffer)
    
            '''tests and saves model'''
            avg_reward = 0
            print('Episode: ' + str(counter))
            print(f"total test reward is {avg_reward}")
            if avg_reward > best_reward:
                print('best reward=' + str(avg_reward))
                self.agent.actor.save('model_actor_{}_{}'.format(counter, avg_reward), save_format="tf")
                self.agent.critic.save('model_critic_{}_{}'.format(counter, avg_reward), save_format="tf")
                best_reward = avg_reward
            if best_reward*100 == 75:
                target = True
            env.reset_agent(0)
            state = env.agent_pos[0] 
            position = state[0]* env.nrow+ state[1]
            for i in range(env.nrow*env.ncol):
                print('probs for state:' + str(i))
                print(self.agent.actor.predict([np.array([[i]], dtype= np.int32),np.array([one_hot_map])]))
            counter+=1
        env.close()



