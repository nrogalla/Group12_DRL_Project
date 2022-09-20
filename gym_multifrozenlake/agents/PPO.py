import PPO_Agent
import Buffer
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

    def test_reward(env):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agentoo7.actor(np.array([state])).numpy())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward

    def run(self, env, episode_length: int, steps: int):

        for s in range(epsiode_length):
            
            done = False
            state = env.reset()

            for e in steps:
                action, probs = self.agent.get_action(state)
                value = self.agent.critic(np.array([state])).numpy()
                next_state, reward, done, _ = env.step(action)
                self.buffer.storeTransition(state, action, reward, value, probs, done)
                state = next_state

                if done == 1:
                    env.reset()

            value = self.agent.critic(np.array([state])).numpy()
            self.buffer.values.append(value[0][0])
            np.reshape(probs, (len(probs),2))
            probs = np.stack(probs, axis=0)
            
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer.states, self.buffer.actions, self.buffer.rewards,
                                        self.buffer.values, self.buffer.dones, self.buffer.probs)

        rewards = [test_reward(env) for _ in range(5)]
        average = np.mean(rewards)
        best = np.max(rewards)

        return average, best


env = gym.make("FrozenLake-v1")
algo = PPO(env.action_space.n, env.observation_space.n)
while target == False:
    avg, max_r = algo.run(env, 500, 256)
    if avg > 200:
        target = True
    



