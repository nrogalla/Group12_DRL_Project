import numpy as np
class Buffer(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []        
        self.dones = []
        self.discounted_returns = []
        self.advantage = []
            
    def storeTransition(self, state, action, reward, value, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []        
        self.dones = []
        self.discounted_returns = []
        self.advantage = []
    
    # computes discounted returns for one episode
    def calculate_disc_returns(self, rewards, gamma):
       # print(rewards)
        d_returns = []
        batch_size = len(rewards) 
        for t in reversed(range(batch_size)): 
            ret = 0
            for i in range(0, batch_size-t):
                ret += np.power(gamma,i) * rewards[t+i]
            d_returns.append(ret)
        d_returns.reverse()
        self.discounted_returns += d_returns
        
        return d_returns
    # calculates advantages for one episode
    def calculate_advantage(self, rewards, values, dones, gamma):
        g = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * dones[i] - values[i]
            if dones[i] == 1:
                g = delta + gamma * 0.95 * 0 * g
            else:
                g = delta + gamma * 0.95 * 1 * g
            
        adv = g
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        self.advantage += adv.tolist()
        return adv