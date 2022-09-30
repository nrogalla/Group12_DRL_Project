from envs.adversarial import ReparameterizedAdversarialEnv
from agents.driver import AdversarialDriver

if __name__ == "__main__":
    env = ReparameterizedAdversarialEnv(n_holes = 1, size = 3, render_mode = "human", fully_observed = True)
    ad = AdversarialDriver(env, 2, 1, 2)
    ad.adversarial_epoch()