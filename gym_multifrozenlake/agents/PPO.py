import PPO_Agent
import Buffer
import numpy as np

class PPO(object):
    def __init__(self, action_space, observation_space):

        self.agent = PPO_Agent(action_space, observation_space)
        self.buffer = Buffer()
        self.ep_reward = []
        self.total_avgr = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []

    # only copied
    def process_buffer(states, actions, rewards, dones, values, gamma):
        g = 0
        lambda_value = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
           delta = rewards[i] + gamma * values[i + 1] * dones[i] - values[i]
           g = delta + gamma * lambda_value * dones[i] * g
           returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv   
    
    def run(self, env, steps: int, episode_length: int):
        for s in range(steps):
            if target == True:
                break 
            
            done = False
            state = env.reset()

            for e in episode_length:
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

            states, actions, returns, advantages = self.buffer.process_buffer(states, actions, rewards, done, values) 

            for epochs in range(10):
                al,cl = self.agent.learn(states, actions, advantages, probs, returns)









