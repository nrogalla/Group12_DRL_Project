import numpy as np
import tensorflow as tf
import PPO_Agent

class AdversarialDriver(object):
    """Runs the environment adversary and agents to collect episodes."""

    def __init__(self,
                env,
                agent,
                action_space,
                observation_space,
                map_action_space,
                adversary_agent,
                adversary_env,
                episode_length,
                steps
                ):
        self.env = env
        self.protagonist = PPO_Agent(action_space, observation_space)
        self.protagonist.old_probs = [[0.25, 0.25, 0.25, 0.25] for i in range(steps)]
        self.antagonist = PPO_Agent(action_space, observation_space)
        self.antagonist.old_probs = [[0.25, 0.25, 0.25, 0.25] for i in range(steps)]
        self.adversary_agent = PPO_Agent(map_action_space, observation_space)
        self.adversary_env = adversary_env
        self.episode_length = episode_length
        self.steps = steps

    def run_adversary_epsiode(self):
        """Episode in which adversary constructs environment and agents play it."""
        # Build environment with adversary.
        adversary_map = self.adversary.run_map

        #run protagonist in adversary map
        self.run_agent(adversary_map, self.protagonist, self.episode_length, self.steps)
        protagonist_reward = np.array(self.test_agent(adversary_map, self.protagonist, 25))
        mean_reward = np.mean(protagonist_reward)

        #run antagonist in adversary map
        self.run_agent(adversary_map, self.antagonist, self.episode_length, self.steps)
        antagonist_reward = np.array(self.test_agent(adversary_map, self.antagonist, 25))
        max_reward = np.max(antagonist_reward)

        # Use agents' reward to compute and set regret-based rewards for PAIRED.
        # By default, regret = max(antagonist) - mean(protagonist).
        # Clipped with 0 as lower bound
        regret = np.max(max_reward - mean_reward,0)
        # Note: as max reward is 1, maybe use loss, or mean(best(antagonist)) - mean(protagonist)
        self.adversary.fit(regret)
        return protagonist_reward, antagonist_reward, regret

    def run_agent(self, env, agent, episode_length: int, steps: int):
        
        target = False
        best_reward = 0
        
        counter = 0
        indices = [i for i in range(env.observation_space.n)]
        depth = env.observation_space.n
        one_hot_states = tf.one_hot(indices, depth)
        
        while counter <= episode_length:
            if target == True:
                break
            done = False
            self.buffer.clear()
            state = env.reset()

            for s in range(steps):
                ohs = one_hot_states[state]
                action, probs = self.agent.get_action(ohs)
                value = self.agent.critic(np.array([ohs])).numpy()

                next_state, reward, done, _ = env.step(action)
                self.buffer.storeTransition(ohs, action, reward, value[0][0], probs[0], done)
                state = next_state

                if done:
                    state = env.reset()

            value = self.agent.critic(np.array([one_hot_states[state]])).numpy()
            self.buffer.values.append(value[0][0])
            probs_list = np.array(self.buffer.probs)
            
            if np.max(self.buffer.rewards) > 0:
                print('Training')
                for epochs in range(10):
                    actor_loss, critic_loss = self.agent.learn(self.buffer.states, self.buffer.actions, self.buffer.rewards,
                                            self.buffer.values, self.buffer.dones, probs)
                        
        env.close()

        def test_agent(self, env, agent, test_runs):
            total_reward = []
            indices = [i for i in range(env.observation_space.n)]
            depth = env.observation_space.n
            one_hot_states = tf.one_hot(indices, depth)
            state = env.reset()
            done = False
            for i in range(test_runs):

                while not done:
                    ohs = one_hot_states[state]
                    action = np.argmax(self.agent.actor(np.array([ohs])).numpy())
                    next_state, reward, done, _ = env.step(action)
                    if done and reward == 0:
                        next_state = env.reset()
                    state = next_state
                    total_reward.append(reward)

            return total_reward