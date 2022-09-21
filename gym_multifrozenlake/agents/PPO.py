from PPO_Agent import PPO_Agent
from Buffer import Buffer
import numpy as np
import gym

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
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(self.agent.actor(np.array([state])).numpy())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward

    def run(self, env, episode_length: int, steps: int):

        for s in range(episode_length):
            
            done = False
            self.buffer.clear()
            state = env.reset()[0]

            for e in range(steps):
                observation = state
                action, probs = self.agent.get_action(observation)
                value = self.agent.critic(np.array([observation])).numpy()

                next_state, reward, done, _, p = env.step(action)
                self.buffer.storeTransition(observation, action, reward, value[0][0], probs[0], done)
                state = next_state

                if done == 1:
                    env.reset()

            value = self.agent.critic(np.array([observation])).numpy()
            self.buffer.values.append(value[0][0])
            probs_list = self.buffer.probs
            np.reshape(probs_list, (len(probs_list),4))
            probs = np.stack(probs_list, axis=0)
            
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer.states, self.buffer.actions, self.buffer.rewards,
                                        self.buffer.values, self.buffer.dones, probs)

        rewards = [self.test_reward(env) for _ in range(5)]
        average = np.mean(rewards)
        best = np.max(rewards)

        return average, best

if __name__=="__main__":
    env = gym.make("FrozenLake-v1")
    algo = PPO(env.action_space.n, env.observation_space.n)
    #print(env.actions)
    avg_rewards_list = []
    target = False

    while target == False:
        algo.run(env, 5000, 256)
        avg_reward, _ = np.mean([algo.test_reward(env) for _ in range(5)])
        print(f"total test reward is {avg_reward}")
        avg_rewards_list.append(avg_reward)
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            algo.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            algo.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
        if best_reward == 200:
            target = True
        env.reset()



