# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import gym
#import networkx as nx
#from networkx import map_graph
import numpy as np
import sys
sys.path.insert(0, '..')

import multifrozenlake 


class AdversarialEnv(multifrozenlake.MultiFrozenLakeEnv):
  """map world where an adversary build the environment the agent plays.
  The adversary places the goal, agent, and up to n_holes blocks in sequence.
  The action dimension is the number of squares in the map, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, n_holes=50, size=15, agent_view_size=5, max_steps=250,
               goal_noise=0., random_z_dim=50, choose_goal_last=False):
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
    self.agent_start_pos = None
    self.goal_pos = None
    self.n_holes = n_holes
    self.goal_noise = goal_noise
    self.random_z_dim = random_z_dim
    self.choose_goal_last = choose_goal_last
    self.width = self.height = size

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_holes + 2

    super().__init__(
        n_agents=1,
        map_size=size,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        competitive=True,
    )

    # Metrics
    self.reset_metrics()

    # Create spaces for adversary agent's specs.
    self.adversary_action_dim = (size - 2)**2
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
    self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    self.adversary_image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.width, self.height, 3),
        dtype='uint8')

    # Adversary observations are dictionaries containing an encoding of the
    # map, the current time step, and a randomly generated vector used to
    # condition generation (as in a GAN).
    self.adversary_observation_space = gym.spaces.Dict(
        {'image': self.adversary_image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space})

    # NetworkX graph used for computing shortest path
    #self.graph = map_graph(dim=[size-2, size-2])
    self.hole_locs = []

  def _gen_map(self, width, height):
    """map is initially empty, because adversary will create it."""
    # Create an empty map
    self.map = multifrozenlake.generate_empty_map(width, height)

    # Generate the surrounding walls
    #self.map.wall_rect(0, 0, width, height)

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
    self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

  def reset(self):
    """Fully resets the environment to an empty map with no agent or goal."""
    #self.graph = map_graph(dim=[self.width-2, self.height-2])
    self.hole_locs = []

    self.step_count = 0
    self.adversary_step_count = 0

    #self.agent_start_dir = self._rand_int(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Generate the map. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_map(self.width, self.height)

    #image = self.map.encode()
    obs = {
        #'image': image,
        'map': self.map,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs

  def reset_agent_status(self):
    """Reset the agent's position, direction, done."""
    self.agent_pos = [[None,None]] * self.n_agents
    #self.agent_dir = [self.agent_start_dir] * self.n_agents
    self.done = [False] * self.n_agents

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and holes."""
    # Remove the previous agents from the world
    for a in range(self.n_agents):
      if self.agent_pos[a] is not None:
        self.map[self.agent_pos[a][0]][self.agent_pos[a][1]] = None

    # Current position and direction of the agent
    self.reset_agent_status()

    if self.agent_start_pos is None:
      raise ValueError('Trying to place agent at empty start position.')
    else:
      self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = str(0)

    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None
      #assert self.agent_dir[a] is not None

      # Check that the agent doesn't overlap with an object
      #start_cell = self.map[self.agent_pos[a]]
      #if not (start_cell.type == 'agent' or
      #        start_cell is None or start_cell.can_overlap()):
      #  raise ValueError('Wrong object in agent start position.')

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    return obs
  '''
  def remove_wall(self, x, y):
    if (x-1, y-1) in self.wall_locs:
      self.wall_locs.remove((x-1, y-1))
    obj = self.map.get(x, y)
    if obj is not None and obj.type == 'wall':
      self.map.set(x, y, None)
  '''
  def compute_shortest_path(self):
    if self.agent_start_pos is None or self.goal_pos is None:
      return

    self.distance_to_goal = abs(
        self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the map, but not in the
    # networkx graph.
    '''
    self.passable = nx.has_path(
        self.graph,
        source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    '''
    self.valid = multifrozenlake.is_valid(self.map, self.nrow, self.ncol)
    if self.valid:
      # Compute shortest path !!!! IMPLEMENT
      '''
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(self.agent_start_pos[0]-1, self.agent_start_pos[1]-1),
          target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
      '''
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

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

    
    x = int(loc % (self.ncol - 2))
    y = int(loc / (self.ncol - 2))
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
        #self.remove_wall(x, y)  # Remove any walls that might be in this loc
        #self.put_obj("G", x, y)
        self.map[x*self.ncol+y] = "G"
        self.goal_pos = (x, y)

    # Place the agent
    elif should_choose_agent:
      #self.remove_wall(x, y)  # Remove any walls that might be in this loc

      # Goal has already been placed here
      if self.map[x*self.ncol+y] is not None:
        # Place agent randomly
        self.agent_start_pos = self.place_one_agent(0) 
        self.deliberate_agent_placement = 0
      else:
        self.agent_start_pos = np.array([x, y]) # or x * self.ncol +y
        self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = str(0)
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
      # Build graph after we are certain agent and goal are placed
      #for w in self.wall_locs:
       # self.graph.remove_node(w)
      self.compute_shortest_path()

    #image = self.map.encode()
    obs = {
        #'image': image,
        'map' : map,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs, 0, done, {}

  def reset_random(self):
    """Use domain randomization to create the environment."""
    #self.graph = map_graph(dim=[self.width-2, self.height-2])

    self.step_count = 0
    self.adversary_step_count = 0

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Create empty map
    self._gen_map(self.width, self.height)

    # Randomly place goal
    self.goal_pos = self.place_obj("G", max_tries=100)

    # Randomly place agent
    self.agent_start_dir = self._rand_int(0, 4)
    self.agent_start_pos = self.place_one_agent(0, rand_dir=False)

    # Randomly place walls
    for _ in range(int(self.n_holes / 2)):
      self.place_obj("H", max_tries=100)

    self.compute_shortest_path()
    self.n_holes_placed = int(self.n_holes / 2)

    return self.reset_agent()

if __name__=="__main__":
  map = AdversarialEnv()
  print(map)
