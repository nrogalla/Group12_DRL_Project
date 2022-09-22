

from PPO_Agent import PPO_Agent
from Buffer import Buffer
import numpy as np
import gym
import sys
#import os

#sys.path.insert(0, '..'))
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake')
import multifrozenlake 
import sys
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake\\envs')
#import envs.adversarial
from envs.adversarial import ReparameterizedAdversarialEnv

class PPO(object):
    def __init__(self, action_space, observation_space):

        self.agent = PPO_Agent(action_space, observation_space)
        self.buffer = Buffer()
        self.episode_reward = []
        self.total_average = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []

    def test_reward(self, env):
        total_reward = 0
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(self.agent.actor(np.array([state])).numpy())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward

    def run(self, env, episode_length: int, steps: int):
        
        target = False
        best_reward = 0
        
        for s in range(episode_length):
            if target == True:
                break
            done = False
            self.buffer.clear()
            state = env.agent_pos[0]#env.reset_agent(0)['map']
            print(state)
            print("new step")

            for e in range(steps):
                observation = state[0]* env.nrow+ state[1]
                action, probs = self.agent.get_action(observation)
                value = self.agent.critic(np.array([observation])).numpy()

                next_state, reward, done, _ = env.step(action)
                print("next_state")
                print(next_state)
                self.buffer.storeTransition(observation, action, reward, value[0][0], probs[0], 1-done)
                state = next_state[0]

                if done:
                    env.reset_agent(0)

            value = self.agent.critic(np.array([observation])).numpy()
            self.buffer.values.append(value[0][0])
            probs_list = self.buffer.probs
            np.reshape(probs_list, (len(probs_list),4))
            probs = np.stack(probs_list, axis=0)
            
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer.states, self.buffer.actions, self.buffer.rewards,
                                        self.buffer.values, self.buffer.dones, probs)

            rewards = [self.test_reward(env) for _ in range(5)]
            avg_reward = np.mean(rewards)
            best_reward = 0
            print(f"total test reward is {avg_reward}")
            if avg_reward > best_reward:
                print('best reward=' + str(avg_reward))
                algo.agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
                algo.agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
                best_reward = avg_reward
            if best_reward == 200:
                target = True
            env.reset()
        env.close()

if __name__=="__main__":
    env = ReparameterizedAdversarialEnv(n_holes = 3,size = 5, render_mode = "human", agent_view_size = 2, max_steps = 2)#gym.make("FrozenLake-v1")
    env.reset()
  
    map, time,done, inf =env.step_adversary(0)
    map, time,done, inf = env.step_adversary(1)
  
    while not done:
     map, time,done, inf = env.step_adversary(np.random.randint(2,4))
    

  
    algo = PPO(4, 5*5)#env.observation_space.n)
    algo.run(env, 50, 32)



