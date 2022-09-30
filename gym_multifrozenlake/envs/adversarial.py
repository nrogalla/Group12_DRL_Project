# coding=utf-8

"""An environment which is built by a learning adversary.
Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import random
from stringprep import map_table_b2
from typing import Optional
import gym
#import networkx as nx
#from networkx import map_graph
import numpy as np
import sys
#import os

#sys.path.insert(0, '..'))
sys.path.insert(0, 'c:\\Users\\Nicole\\Documents\\UNI\\Cognitive_Science\\DRL\\PAIRED-Project\\Group12_DRL_Project\\gym_multifrozenlake')
import multifrozenlake 


class AdversarialEnv(multifrozenlake.MultiFrozenLakeEnv):
  """map world where an adversary build the environment the agent plays.
  The adversary places the goal, agent, and up to n_holes blocks in sequence.
  The action dimension is the number of squares in the map, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, n_holes=50, size=8, agent_view_size=5, max_steps=250,
               goal_noise=0., random_z_dim=50, choose_goal_last=False, render_mode: Optional[str] = None, fully_observed = True):
    """Initializes environment in which adversary places goal, agent, obstacles.
    Args:
      n_holes: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the map; i.e. make a
        size x size map.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the map.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
      goal_noise: The probability with which the goal will move to a different
        location than the one chosen by the adversary.
      random_z_dim: The environment generates a random vector z to condition the
        adversary. This gives the dimension of that vector.
      choose_goal_last: If True, will place the goal and agent as the last
        actions, rather than the first actions.
    """
    self.goal_pos = None
    self.n_holes = n_holes
    self.goal_noise = goal_noise
    self.random_z_dim = random_z_dim
    self.choose_goal_last = choose_goal_last
    self.ncol = self.nrow = size
    

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_holes + 2

    super().__init__(
        n_agents=1,
        map_size=size,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        competitive=True,
        render_mode = render_mode,
        fully_observed = fully_observed
    )
    self.start_agent_pos = [[None,None]] * self.n_agents
    # Metrics
    self.reset_metrics()

    # Create spaces for adversary agent's specs.
    self.adversary_action_dim = (size)**2
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
    self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    self.adversary_map_obs_space = gym.spaces.Box(
        low=0,
        high=4,
        shape=(self.ncol, self.nrow, 3),
        dtype='uint8')

    # Adversary observations are dictionaries containing an encoding of the
    # map, the current time step, and a randomly generated vector used to
    # condition generation (as in a GAN).
    self.adversary_observation_space = gym.spaces.Dict(
        {'map': self.adversary_map_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space})
  
    self.hole_locs = []

  def _gen_map(self, ncol, nrow):
    """map is initially empty, because adversary will create it."""
    # Create an empty map
    self.map = multifrozenlake.generate_empty_map(ncol, nrow)
    
    return self.map

  def get_goal_x(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[0]

  def get_goal_y(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[1]

  def reset_metrics(self):
    self.distance_to_goal = -1
    self.n_holes_placed = 0
    self.deliberate_agent_placement = -1
    self.passable = -1
    self.shortest_path_length = (self.ncol - 2) * (self.nrow - 2) + 1

  def reset(self):
    """Fully resets the environment to an empty map with no agent or goal."""
    self.hole_locs = []
    self.step_count = 0
    self.adversary_step_count = 0
    self.lastaction = [None]*self.n_agents

    # Current position and direction of the agent
    self.reset_agent_status()

    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Generate the map. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_map(self.ncol, self.nrow)# adapt for is slippery

    obs = {
        'map': self.map,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }
    

    return obs

  def reset_agent_status(self):
    """Reset the agents position, direction, done."""
    self.agent_pos = [[None,None]] * self.n_agents
    
    self.done = [False] * self.n_agents

  def reset_agent(self, agent_id):
    """Resets the agents start positions, but leaves goal and holes."""
    
    # Current position of the agent
    self.reset_agent_status()

    if self.start_agent_pos[agent_id][0] is None:
      raise ValueError('Trying to place agent at empty start position.')
    else:
      self.map[self.start_agent_pos[agent_id][0]][self.start_agent_pos[agent_id][1]] = 'F' #str(agent_id)
      self.agent_pos[0] = self.start_agent_pos[0]

    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None
   
    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    if self.render_mode == "human":
      self.render()
    
    return obs

  def generate_random_z(self):
    return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

  def step_adversary(self, loc):
    """The adversary gets n_holes + 2 moves to place the goal, agent, holes.
    The action space is the number of possible squares in the map. The squares
    are numbered from left to right, top to bottom.
    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.
    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    
    if loc >= self.adversary_action_dim:
      raise ValueError('Position passed to step_adversary is outside the map.')

    x = int(loc / (self.ncol))
    y = int(loc % (self.ncol))
    done = False

    if self.choose_goal_last:
      should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
      should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
    else:
      should_choose_goal = self.adversary_step_count == 0
      should_choose_agent = self.adversary_step_count == 1

    # Place goal
    if should_choose_goal:
      # If there is goal noise, sometimes randomly place the goal
      
      if random.random() < self.goal_noise:
        self.goal_pos = self.place_obj("G", max_tries=100) 
      else:
        self.map[x][y] = "G"
    
    # Place the agent
    elif should_choose_agent:
      
      # Goal has already been placed here
      if self.map[x][y] is not None:
        # Place agent randomly
        self.agent_pos[0] = self.start_agent_pos[0] = self.place_one_agent(0) 
        self.deliberate_agent_placement = 0
      else:
        self.agent_pos[0] = self.start_agent_pos[0] = [x, y]
        self.map[self.agent_pos[0][0]][self.agent_pos[0][1]] = 'F' #str(0)
        self.deliberate_agent_placement = 1

    # Place hole
    elif self.adversary_step_count < self.adversary_max_steps:
      # If there is already an object there, action does nothing
      if self.map[x][y] is None:
        self.map[x][y] = "H"
        self.n_holes_placed += 1
        self.hole_locs.append((x-1, y-1))

    self.adversary_step_count += 1
   
    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True
      for i in range(self.ncol):
        for j in range(self.nrow):
          if self.map[i][j] is None:
            self.map[i][j] = "F"
      self.generate_P(self.nrow,self.ncol, self.map, False)
      self. valid = multifrozenlake.is_valid(self.map, self.nrow, self.ncol)
      if self.render_mode == "human":
            self.render()
     
    obs = {
        'map' : self.map,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs, 0, done, {}

  def reset_random(self):
    """Use domain randomization to create the environment."""

    self.step_count = 0
    self.adversary_step_count = 0

    # Current position and direction of the agent
    self.reset_agent_status()

    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Create empty map
    self._gen_map(self.ncol, self.nrow)

    # Randomly place goal
    self.goal_pos = self.place_obj("G", max_tries=100)

    # Randomly place agent
    self.agent_pos[0] = self.start_agent_pos[0]= self.place_one_agent(0)

    # Randomly place holes
    for _ in range(int(self.n_holes / 2)):
      self.place_obj("H", max_tries=100)

    self.n_holes_placed = int(self.n_holes / 2)
    if self.render_mode == "human":
            self.render()
    return self.reset_agent(0)

class ReparameterizedAdversarialEnv(AdversarialEnv):
  """Grid world where an adversary builds the environment the agent plays.
  In this version, the adversary takes an action for each square in the grid.
  There is no limit on the number of blocks it can place. The action space has
  dimension 4; at each step the adversary can place the goal, agent, a wall, or
  nothing. If it chooses to place the goal or agent when they have previously
  been placed at a different location, they will move to the new location.
  """

  def __init__(self, n_holes=50, size=15, agent_view_size=5, max_steps=250, render_mode: Optional[str] = None, fully_observed = True):
    """Initializes environment in which adversary places goal, agent, obstacles.
    Args:
      n_holes: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the map; i.e. make a
        size x size map.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the map.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
    """
    super().__init__(
        size=size,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        render_mode = render_mode,
        n_holes = n_holes,
        fully_observed=fully_observed
    )

    # Adversary has four actions: place agent, goal, wall, or nothing
    self.adversary_action_dim = 4
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)

    # Reparam adversaries have additional inputs for the current x,y coords
    self.adversary_xy_obs_space = gym.spaces.Box(
        low=1, high=size-2, shape=(1,), dtype='uint8')

    # Observations are dictionaries containing an encoding of the grid and the
    # agent's direction
    self.adversary_observation_space = gym.spaces.Dict(
        {'map': self.adversary_map_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space,
         'x': self.adversary_xy_obs_space,
         'y': self.adversary_xy_obs_space})

    self.adversary_max_steps = (size)**2

    self.wall_locs = []

  def reset(self):
    self.hole_locs = []
    obs = super().reset()
    obs['x'] = [1]
    obs['y'] = [1]
    return obs

  def select_random_grid_position(self):
    row =  np.random.random_integers(0,self.nrow-1)
    col = np.random.random_integers(0, self.ncol-1)
    return np.array([
        row,
        col
    ])

  def get_xy_from_step(self, step):
    # Add offset of 1 for outside walls
    x = int(step % (self.ncol)) 
    y = int(step / (self.ncol))
    return x, y

  def step_adversary(self, action):
    """The adversary gets a step for each available square in the grid.
    At each step it chooses whether to place the goal, the agent, a block, or
    nothing. If it chooses agent or goal and they have already been placed, they
    will be moved to the new location.
    Args:
      action: An integer in range 0-3 specifying which object to place:
        0 = goal
        1 = agent
        2 = hole
        3 = frozen
    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    done = False

    if self.adversary_step_count < self.adversary_max_steps:
      x, y = self.get_xy_from_step(self.adversary_step_count)

      # Place goal
      if action == 0:
        if self.goal_pos is not None:
          self.map[self.goal_pos[0]][self.goal_pos[1]] = None
        self.map[x][y] = "G"
        self.goal_pos = (x, y)

      # Place the agent
      elif action == 1:
        
        if self.start_agent_pos[0][0] is not None and self.start_agent_pos[0][1] is not None:
          self.map[self.start_agent_pos[0][0]][self.start_agent_pos[0][1]] = None
        
        self.agent_pos[0]= self.start_agent_pos[0] = np.array([x, y])
        self.map[x][y] = 'F' #str(0)
      
      # Place hole
      elif action == 2:
        self.map[x][y] = "H"
        self.n_holes_placed += 1

        self.hole_locs.append((x, y))
    
    self.adversary_step_count += 1
    
    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True
      for i in range(self.ncol):
        for j in range(self.nrow):
          if self.map[i][j] is None:
            self.map[i][j] = "F"
      self.generate_P(self.nrow,self.ncol, self.map, False)
      self. valid = multifrozenlake.is_valid(self.map, self.nrow, self.ncol)
      print(self.map)
      print("AGENTPOS")
      print(self.agent_pos)
      # If the adversary has not placed the agent or goal, place them randomly
      if self.agent_pos[0][0] is None:
        self.agent_pos[0] = self.start_agent_pos[0] =  self.select_random_grid_position()
        print("random agent pos")
        print(self.agent_pos)
        self.map[self.agent_pos[0][0]][self.agent_pos[0][0]] = 'F'# str(0)
        self.deliberate_agent_placement = 0
      else:
        self.deliberate_agent_placement = 1

      if self.goal_pos is None:
        self.goal_pos = self.select_random_grid_position()
        print("GOAL")
        print(self.goal_pos)
        self.map[self.goal_pos[0]][self.goal_pos[1]] = "G"

      self.valid = multifrozenlake.is_valid(self.map, self.nrow, self.ncol)
      
      if self.render_mode == "human" and done == True:
        print(self.map)
        self.render()
    else:
      x, y = self.get_xy_from_step(self.adversary_step_count)

    obs = {
        'map': self.map,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z(),
    }
   
    return obs, 0, done, {}


if __name__=="__main__":
  env = ReparameterizedAdversarialEnv(n_holes = 3,size = 5, render_mode = "human", fully_observed = True, max_steps = 2)
  env.reset()
  
  map, time,done, inf =env.step_adversary(0)
  map, time,done, inf = env.step_adversary(1)
  
  while not done:
    map, time,done, inf = env.step_adversary(np.random.randint(2,4))
  
 
  env.step([multifrozenlake.RIGHT])
  

