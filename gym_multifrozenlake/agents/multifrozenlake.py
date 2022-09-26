import gym
import math
from gym.envs.toy_text.utils import categorical_sample
from abc import abstractmethod

from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, utils

from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


AGENT_COLOURS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple"
]

##COPIED FROM FROZENLAKE ENV  WITH REPLACING SIZE BY NROW AND NCOL FOR FINER CONTROL

# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_nrow: int, max_ncol: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_nrow or c_new < 0 or c_new >= max_ncol:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

def generate_empty_map(nrow: int = 8, ncol: int = 8):
    """Generates an empty map
    Args:
        nrow: number of rows in the map
        ncol: number of columns in the map
    Returns:
        An empty map
    """
    assert ncol >= 3
    assert nrow >= 3

    return [[None for _ in range(ncol)] for _ in range(nrow)]


class MultiFrozenLakeEnv(Env):
    """
    Frozen lake environment with multi-agent support.

    ### Action Space
    The agent takes a n_agent-element vector for actions, specifying the action to be taken for each of the agents.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:
    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a n_agents-element vector containing 2-element vectors with both 
    row and column coordinate of the agents positions on the map. 
    
    ### Rewards
    Reward schedule:
    - Reach goal(G): +1
    - Reach hole(H): 0
    - Reach frozen(F): 0
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map=None,
        map_size=8,
        nrow=None,
        ncol=None,
        is_slippery=True,
        n_agents=3,
        competitive=False,
        fixed_environment=False,
        fully_observed=False,
        agent_view_size=2,
        max_steps=200,
        seed=52

    ):
        """Constructor for multi-agent frozenlake environment generator.
        Args:
        map: Used to specify custom map for frozen lake.
        map_size: Number of tiles for the width and height of the square map.
        ncol: Number of tiles across map width.
        nrow: Number of tiles in map of grid.
        is_slippery: True/False. If True will move in intended direction with
            probability of 1/3 else will move in either perpendicular direction with
            equal probability of 1/3 in both directions.
        n_agents: The number of agents playing in the world.
        competitive: If True, as soon as one agent locates the goal, the episode
            ends for all agents. If False, if one agent locates the goal it is
            respawned somewhere else in the grid, and the episode continues until
            max_steps is reached.
        fixed_environment: If True, will use the same random seed each time the
            environment is generated, so it will remain constant / be the same
            environment each time.
        fully_observed: If True, each agent will receive an observation of the
            full environment state, rather than a partially observed, ego-centric
            observation.
        agent_view_size: Number of tiles in the agent's square, partially
            observed view of the world.
        max_steps: Number of environment steps before the episode end (max
            episode length).
        seed: Random seed used in generating environments.
        
        
        """
        self.fully_observed = fully_observed

        # Can't set both map_size and nrow/ncol
        if map_size:
            assert nrow is None and ncol is None
            nrow = map_size
            ncol = map_size
        
        self.map = map

        self.n_agents = n_agents
        self.competitive = competitive
        self.max_steps = max_steps
        
        if self.n_agents == 1:
            self.competitive = True
        
        self.agent_view_size = agent_view_size
        if self.fully_observed:
            self.agent_view_size = max(nrow, ncol)

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Compute observation and action spaces
        if self.fully_observed:
            obs_map_shape = (ncol,nrow, 3)
        else:
            obs_map_shape = (self.agent_view_size, self.agent_view_size, 3)
        
        self.action_space = gym.spaces.Box(low=0, high=3,
                                        shape=(self.n_agents,), dtype='int64')

        map_space = gym.spaces.Box(
          low= 0,
          high= 2,
          shape=(self.n_agents,) + obs_map_shape,
          dtype='uint8')

        # Observations are dictionaries containing an encoding of the grid and the
        # agent's direction
        observation_space = {'map': map_space}
        if self.fully_observed:
            self.position_obs_space = gym.spaces.Box(low=0,
                                                high=max(nrow, ncol),
                                                shape=(self.n_agents, 2),
                                                dtype='uint8')
            observation_space['position'] = self.position_obs_space
        self.observation_space = gym.spaces.Dict(observation_space)

        # Environment configuration
        self.nrow = nrow
        self.ncol = ncol
        self.render_mode = render_mode
        self.is_slippery = is_slippery

        #current position of agents
        self.agent_pos = [[None, None]] * self.n_agents
        for a in range(self.n_agents):
            self.agent_pos[a][0] = 0
            self.agent_pos[a][1] = 0

        # Maintain a done variable for each agent
        self.done = [False] * self.n_agents
        
        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

        self.seed_value = seed
        self.seed = seed
        self.fixed_environment = fixed_environment
        #Initialize the state
        self.reset()

    # initializes probability table for map with probability of transitioning to the next tile 
    # in the intended direction based on the slippery nature of the map
    def generate_P(self, nrow, ncol, map, is_slippery):
        nA = 4
        nS = nrow * ncol

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
       
        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = map[newrow][newcol]
            terminated = newletter in "GH" 
            reward = float(newletter == "G")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = map[row][col]
                    if letter in "GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
        


    def reset(self, options: Optional[dict] = None):
        if self.fixed_environment:
            self.seed(self.seed_value)

        # Current position of the agent
        self.agent_pos = [[None, None]] * self.n_agents

        self.done = [False] * self.n_agents
        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self.map = self._gen_map(self.map,self.ncol, self.nrow)
        self.generate_P(self.nrow, self.ncol, self.map, self.is_slippery)
        print(self.P)
        self.step_count = 0
        # should be defined by _gen_map
        for a in range(self.n_agents):
          assert self.agent_pos[a] is not None
        
        for a in range(self.n_agents):
            pos = categorical_sample(0., self.np_random)
            self.agent_pos[a][0] = pos // self.ncol
            self.agent_pos[a][1] = pos % self.ncol
        self.lastaction = [None]*self.n_agents

        if self.render_mode == "human":
            self.render()
        
        obs = self.gen_obs()

        return obs
        
        #return self.agent_pos

    def gen_obs(self):
        """Generate the stacked observation for all agents."""
        maps = []
        positions = []
        for a in range(self.n_agents):
            
            if self.fully_observed:
                map = self.map
                
            else:
                map = self.gen_obs_map(a)
            maps.append(map)
            
            positions.append(self.agent_pos[a])

        obs = {
            'map': maps,
        }
        if self.fully_observed:
            obs['position'] = positions

        return obs

    
    def get_view_exts(self, agent_id):
        """Get the extents of the square set of tiles visible to the agent.
        
        Args:
        agent_id: Integer ID of the agent.
        Returns:
        Top left and bottom right (x,y) coordinates of set of visible tiles.
        """
        # Facing left
        if self.lastaction[agent_id] == 0:
            top_x = self.agent_pos[agent_id][0] - self.agent_view_size + 1
            top_y = self.agent_pos[agent_id][1] +1 - self.agent_view_size // 2
        # Facing down
        elif self.lastaction[agent_id] == 1 or self.lastaction[agent_id] is None:
            top_x = self.agent_pos[agent_id][0]+1 - self.agent_view_size // 2
            top_y = self.agent_pos[agent_id][1]
        # Facing right
        elif self.lastaction[agent_id] == 2:
            top_x = self.agent_pos[agent_id][0]
            top_y = self.agent_pos[agent_id][1] +1 - self.agent_view_size // 2
        # Facing up
        elif self.lastaction[agent_id] == 3:
            top_x = self.agent_pos[agent_id][0] +1 - self.agent_view_size // 2
            top_y = self.agent_pos[agent_id][1] +1 - self.agent_view_size + 1
        else:
            assert False, 'invalid agent direction'
        if top_x < 0: 
            top_x = 0
        if top_y < 0:
            top_y = 0
        bot_x = top_x + self.agent_view_size
        bot_y = top_y + self.agent_view_size
        if bot_x > self.nrow:
            bot_x = self.nrow -1
        if bot_y > self.ncol:
            bot_y = self.ncol -1

        return (top_x, top_y, bot_x, bot_y)

    def gen_obs_map(self, agent_id):
        """Generate the sub-map observed by the agent.
        Args:
        agent_id: Integer ID of the agent for which to generate the grid.
        Returns:
        Sub-map
        """
        top_x, top_y, bot_x, bot_y = self.get_view_exts(agent_id)
    
        obs_map = [sub[top_x:bot_x] for sub in self.map[top_y:bot_y]]

        return obs_map

    
    @abstractmethod
    def _gen_map(self, map, ncol, nrow):
        pass
    

    def step_one_agent(self, act, agent_id):
        transitions = self.P[self.agent_pos[agent_id][0]*self.ncol + self.agent_pos[agent_id][1]][act]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, pos, r, t = transitions[i]
        print(t)
        posit = [None] *2
        posit[0],posit[1] = pos // self.ncol, pos % self.ncol
        agent_blocking = False
        for a in range(self.n_agents):
            if a != agent_id and np.array_equal(self.agent_pos[a], posit):
                agent_blocking = True
                r = 0
        if not agent_blocking and not self.done[agent_id]:
            self.agent_pos[agent_id] = posit
            self.lastaction[agent_id] = act
            if self.map[posit[0]][posit[1]] == 'G':
                #self.agent_is_done(agent_id)
                self.done[agent_id] = True
            if t is True:
                #self.agent_is_done(agent_id)
                self.done[agent_id] = True
            
                
            

        if self.render_mode == "human":
            self.render()
        return r 

    def step(self, actions):

        rewards = [0] * self.n_agents

        # Randomize order in which agents act for fairness
        agent_ordering = np.arange(self.n_agents)
        np.random.shuffle(agent_ordering)

        # Step each agent
        for a in agent_ordering:
            rewards[a] = self.step_one_agent(actions[a], a)
        self.step_count += 1
        collective_done = False
        # In competitive version, if one agent finishes the episode is over.
        if self.competitive:
            collective_done = np.sum(self.done) >= 1 

        # Running out of time applies to all agents
        if self.step_count >= self.max_steps:
          collective_done = True
        print("agentpos")
        print(self.agent_pos)
        return self.agent_pos, rewards, collective_done, {}
    
    
    def place_obj(self,
                    obj,
                    top=None,
                    size=None,
                    max_tries=math.inf):
        """Place an object at an empty position in the map.
        Args:
        obj: "H", "F", "G" or Agentnumber depending on the object to be placed
        top: (x,y) position of the top-left corner of rectangle where to place.
        size: Size of the rectangle where to place.
        max_tries: Throw an error if a position can't be found after this many
            tries.
        Returns:
        Position where object was placed.
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.ncol, self.nrow)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise gym.error.RetriesExceededError('Rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((np.random.randint(top[0],
                                            min(top[0] + size[0], self.ncol)),
                            np.random.randint(top[1],
                                            min(top[1] + size[1], self.nrow))))
            
            break
        if obj == "H" or obj == "F": 
            self.map[pos[0]][pos[1]] = obj
        else:
            self.map[pos[0]][pos[1]] = 'F'
            

        return pos

    def place_agent(self, top=None, size=None, max_tries=math.inf):
        """Set the starting point of all agents in the world.
        Name chosen for backwards compatibility.
        Args:
          top: (x,y) position of the top-left corner of rectangle where agents can
          be placed.
          size: Size of the rectangle where to place.
          rand_dir: Choose a random direction for agents.
          max_tries: Throw an error if a position can't be found after this many
            tries.
        """
        for a in range(self.n_agents):
            self.place_one_agent(a, top=top, size=size, max_tries=math.inf)

    def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      max_tries=math.inf):
        """Set the agent's starting point at an empty position in the map."""

        self.agent_pos[agent_id] = None
        pos = self.place_obj(str(agent_id), top, size, max_tries=max_tries)
        self.agent_pos[agent_id] = pos
        self.lastaction[agent_id] = None
        return pos

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[toy_text]`")

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
    
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]
        
        map = self.map
        assert isinstance(map, list), f"map should be a list or an array, got {map}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if map[y][x] == "H":
                    self.window_surface.blit(self.hole_img, pos)
                elif map[y][x] == "G":
                    self.window_surface.blit(self.goal_img, pos)
                elif map[y][x] == "S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elfs
        
        for a in range(self.n_agents):
            bot_row, bot_col = self.agent_pos[a][0], self.agent_pos[a][1]
            
            cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
            
            last_action = self.lastaction[a] if self.lastaction[a] is not None else 1
            elf_img = self.elf_images[last_action]

            if map[bot_row][bot_col] == "H":
                self.window_surface.blit(self.cracked_hole_img, cell_rect)
            else:
                self.window_surface.blit(elf_img, cell_rect)
    
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(1)#self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        map = self.map.tolist()
        outfile = StringIO()
        row = [None] * self.n_agents
        col = [None] * self.n_agents
        for a in range(self.n_agents):
          row[a], col[a] = self.agent_pos[a][0], self.agent_pos[a][1]
        map = [[c.decode("utf-8") for c in line] for line in map]
        
        for a in range(self.n_agents):
          map[row[a]][col[a]] = utils.colorize(map[row[a]][col[a]], AGENT_COLOURS[a], highlight=True)
          
          if self.lastaction[a] is not None:
              outfile.write(f"Agent {a}: ({['Left', 'Down', 'Right', 'Up'][self.lastaction[a]]})\n")
          else:
              outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in map) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def agent_is_done(self, agent_id):

        self.done[agent_id] = True

        # Respawn agent in new location
        self.place_one_agent(agent_id)

if __name__=="__main__":
    fl = MultiFrozenLakeEnv(render_mode='human', is_slippery = False)
    fl.reset()
    

 
