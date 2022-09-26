

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
from adversarial import ReparameterizedAdversarialEnv

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
            next_state, reward, fail, done, _ = env.step(action)
            if fail:
                next_state = env.reset()[0]
            state = next_state
            total_reward += reward

        return total_reward

    def run(self, env, episode_length: int, steps: int):
        
        target = False
        best_reward = 0
        self.agent.old_probs = [[0.25, 0.25, 0.25, 0.25] for i in range(steps)]
        
        
        for e in range(episode_length):
            print('.')
            if target == True:
                break
            done = False
            self.buffer.clear()
            #state = env.agent_pos[0]#env.reset_agent(0)['map']
            state = env.reset()[0]

            for s in range(steps):
                action, probs = self.agent.get_action(state)
                value = self.agent.critic(np.array([state])).numpy()

                next_state, reward, fail, done, _ = env.step(action)
                self.buffer.storeTransition(state, action, reward, value[0][0], probs[0], done)
                state = next_state
                if fail:
                    state = env.reset()[0]
                if done:
                    state = env.reset()[0]

            value = self.agent.critic(np.array([state])).numpy()
            self.buffer.values.append(value[0][0])
            probs_list = np.array(self.buffer.probs)
            
            if np.max(self.buffer.rewards) > 0:
                print('Training')
                for epochs in range(10):
                    actor_loss, critic_loss = self.agent.learn(self.buffer.states, self.buffer.actions, self.buffer.rewards,
                                            self.buffer.values, self.buffer.dones, probs)
            
                rewards = [self.test_reward(env) for _ in range(100)]
                avg_reward = np.mean(rewards)
                best_reward = 0
                print('Episode: ' + str(e))
                print(f"total test reward is {avg_reward}")
                if avg_reward > best_reward:
                    print('best reward=' + str(avg_reward))
                    algo.agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
                    algo.agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
                    best_reward = avg_reward
                if best_reward*100 == 75:
                    target = True
                env.reset()
                for i in range(env.observation_space.n):
                    print('probs for state:' + str(i))
                    print(self.agent.actor.predict(np.array([i])))
        env.close()

if __name__=="__main__":
    #env = ReparameterizedAdversarialEnv(n_holes = 3, size = 5, render_mode = "human", agent_view_size = 3, max_steps = 128)
    #env.reset()
  
    #map, time,done, inf =env.step_adversary(0)
    #map, time,done, inf = env.step_adversary(1)
  
    #while not done:
    # map, time,done, inf = env.step_adversary(np.random.randint(2,4))
    

    env = gym.make("FrozenLake-v1", desc=["SFFF", "FHFH", "FFFH", "HFFG"], is_slippery=False)

    algo = PPO(env.action_space.n, env.observation_space.n)#env.observation_space.n)
    algo.run(env, 5000, 128)



