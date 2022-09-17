class PPO_Agent(object):

    def __init__(number_actions: int, number_observations: int, alpha: float = 0.001, gamma: float = 0.7):
        self.n_actions = number_actions
        self.n_observations = number_observations
        self.learning_rate = alpha
        self.gamma = gamma
        self.actor = Actor(number_actions, number_observations, 256, 128, 64)
        self.critic = Critic(number_observations, 256, 128, 64)

    def get_action(self, observation):
        
        action_probs = self.actor(observation)
        value = self.critic(observation)
        action = action_probs.sample() # get action with according to probabilities

        return action, value, action_probs
