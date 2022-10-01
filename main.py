from envs.adversarial import ReparameterizedAdversarialEnv
from agents.driver import AdversarialDriver
import matplotlib.pyplot as plt

def visualize_performance(protagonist_rewards, antagonist_rewards, regret):
    '''Visualize rewards for protagonist and antagonist and adversary regret.'''
    plt.figure()
    line1, = plt.plot(protagonist_rewards)
    line2, = plt.plot(antagonist_rewards)
    line3, = plt.plot(regret)
    plt.xlabel("Epoch")
    plt.ylabel("Reward/Regret")
    plt.legend((line1,line2, line3),("Protagonist","Antagonist", "Regret"))
    plt.show()     
    
if __name__ == "__main__":
    env = ReparameterizedAdversarialEnv(n_holes = 1, size = 3, render_mode = "human", fully_observed = True)
    ad = AdversarialDriver(env, 2, 1, 3)
    p_rewards = []
    a_rewards = []
    regrets = []
    
    for i in range(10):
        p, a, r = ad.adversarial_epoch()
        p_rewards.append(p)
        a_rewards.append(a)
        regrets.append(r)
    visualize_performance(p_rewards, a_rewards, regrets)