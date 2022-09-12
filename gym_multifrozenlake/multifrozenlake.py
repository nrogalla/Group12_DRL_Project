import gym
import math
from gym.envs.toy_text.utils import categorical_sample
from abc import abstractmethod

from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, spaces, utils

from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

AGENT_COLOURS = [
    np.array([60, 182, 234]),  # Blue
    np.array([229, 52, 52]),  # Red
    np.array([144, 32, 249]),  # Purple
    np.array([69, 196, 60]),  # Green
    np.array([252, 227, 35]),  # Yellow
] 

COLOURS = [
    "red",
    "blue",
    "green"
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

##COPIED FROM FROZENLAKE ENV WITH REPLACING SIZE BY NROW AND NCOL FOR FINER CONTROL
'''
def generate_random_map(nrow: int = 8, ncol: int = 8, p: float = 0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (nrow, ncol), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, nrow, ncol)
    print(board)
    print("valid")
    return ["".join(x) for x in board]
'''
def generate_empty_map(nrow: int = 8, ncol: int = 8):
    assert ncol >= 3
    assert nrow >= 3

    return [[None for i in range(ncol)] for j in range(nrow)]

class Agent():
  def __init__(self, agent_id, state):
    super(Agent,self).__init__('agent')
    self.agent_id = agent_id
    self.dir = state
  
  #adapt rendering
  '''
  def render(self, img):
    tri_fn = rendering.point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )

    # Rotate the agent based on its direction
    tri_fn = rendering.rotate_fn(
        tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
    color = AGENT_COLOURS[self.agent_id]
    rendering.fill_coords(img, tri_fn, color)
  '''


class MultiFrozenLakeEnv(Env):
    """Frozen lake environment with multi-agent support."""

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
        #frozenlake_mode=False,
        max_steps=200,
        seed=52

    ):
        self.fully_observed = fully_observed

        # Can't set both map_size and nrow/ncol
        if map_size:
            assert nrow is None and ncol is None
            nrow = map_size
            ncol = map_size
        
        #self._gen_grid(map, nrow, ncol)
        self.map = map

        self.n_agents = n_agents
        self.competitive = competitive
        self.max_steps = max_steps
        
        if self.n_agents == 1:
            self.competitive = True
        
        #self.actions = 

        self.agent_view_size = agent_view_size
        if self.fully_observed:
            self.agent_view_size = max(nrow, ncol)

        # Range of possible rewards
        self.reward_range = (0, 1)

        #self.generate_P(nrow, ncol, self.map, is_slippery)

        

        # Compute observation and action spaces
        # Direction always has an extra dimension for tf-agents compatibility
        self.direction_obs_space = gym.spaces.Box(
            low=0, high=3, shape=(self.n_agents,), dtype='uint8')

        #self.frozenlake_mode = frozenlake_mode
        
        if self.fully_observed:
            obs_image_shape = (ncol,nrow, 3)
        else:
            obs_image_shape = (self.agent_view_size, self.agent_view_size, 3)
        '''
        if self.frozenlake_mode: 
            msg = 'Backwards compatibility with minigrid only possible with 1 agent'
            assert self.n_agents == 1, msg

            # Single agent case
            self.action_space = gym.spaces.Discrete(len(self.actions))
            #self.observation_space = gym.spaces.Discrete(nrow * ncol)

            # Images have three dimensions
            self.image_obs_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_image_shape,
                dtype='uint8')
               
        else:
          '''
        self.action_space = gym.spaces.Box(low=0, high=3,
                                        shape=(self.n_agents,), dtype='int64')

        self.image_obs_space = gym.spaces.Box(
          low=0,
          high=255,
          shape=(self.n_agents,) + obs_image_shape,
          dtype='uint8')

        # Observations are dictionaries containing an encoding of the grid and the
        # agent's direction
        observation_space = {'image': self.image_obs_space,
                            'direction': self.direction_obs_space}
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

        #current position and direction of agents
        self.agent_pos = [[None, None]] * self.n_agents
        for a in range(self.n_agents):
            self.agent_pos[a][0] = 0
            self.agent_pos[a][1] = 0
        #self.agent_dir = [None] * self.n_agents

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
    def generate_P(self, nrow, ncol, map, is_slippery):
        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(map == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

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
            newletter = map[newrow][newcol]#map[newrow, newcol]
            terminated = newletter in "GH" #bytes(newletter) in b"GH"
            reward = float(newletter == "G")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = map[row][col]#map[row, col]
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
        


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.fixed_environment:
            self.seed(self.seed_value)

        # Current position and direction of the agent
        self.agent_pos = [[None, None]] * self.n_agents
        #self.agent_dir = [None] * self.n_agents
        self.done = [False] * self.n_agents

        self.map = self._gen_map(self.map,self.ncol, self.nrow)
        self.generate_P(self.nrow, self.ncol, self.map, self.is_slippery)

        
        for a in range(self.n_agents):
          assert self.agent_pos[a] is not None
        
        for a in range(self.n_agents):
            pos = categorical_sample(self.initial_state_distrib, self.np_random)
            self.agent_pos[a][0] = pos // self.ncol
            self.agent_pos[a][1] = pos % self.ncol
        self.lastaction = [None]*self.n_agents

        if self.render_mode == "human":
            self.render()
        return self.agent_pos#, {"prob": 1}
        #obse = self.gen_obs()

    '''
    def _gen_map(self, map, nrow, ncol):
        if map is None: 
              map = generate_random_map(nrow, ncol)
        print("Generated")
        print(map)
        self.map = map = np.asarray(map, dtype = "c").decode("UTF-8")
        print(map)
        self.nrow, self.ncol = nrow, ncol = map.shape
        return self.map
    '''
    @abstractmethod
    def _gen_map(self, map, ncol, nrow):
        pass
    

    def step_one_agent(self, act, agent_id):
        transitions = self.P[self.agent_pos[agent_id][0]*self.ncol + self.agent_pos[agent_id][1]][act]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, pos, r, t = transitions[i]
        posit = [None] *2
        posit[0],posit[1] = pos // self.ncol, pos % self.ncol
        agent_blocking = False
        for a in range(self.n_agents):
            if a != agent_id and np.array_equal(self.agent_pos[a], posit):
                agent_blocking = True
                r = 0
        if not agent_blocking:
            self.agent_pos[agent_id] = posit
            self.lastaction[agent_id] = act

        if self.render_mode == "human":
            self.render()
        return r #(int(s), r, t, False, {"prob": p})

    def step(self, actions):

        rewards = [0] * self.n_agents

        # Randomize order in which agents act for fairness
        agent_ordering = np.arange(self.n_agents)
        np.random.shuffle(agent_ordering)

        # Step each agent
        for a in agent_ordering:
            rewards[a] = self.step_one_agent(actions[a], a)

        #obs = self.gen_obs()

        collective_done = False
        # In competitive version, if one agent finishes the episode is over.
        if self.competitive:
            collective_done = np.sum(self.done) >= 1 ## WO WIRD SELF.DONE GESETZT?

        # Running out of time applies to all agents
        # if self.step_count >= self.max_steps:
        #  collective_done = True
        
        return self.agent_pos, rewards, collective_done, {}
    
    
    """
    @abstractmethod
    def _gen_map(self, nrow, ncol): # needs to be implemented in adversary
        pass
    """
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

            pos = np.array((self._rand_int(top[0],
                                            min(top[0] + size[0], self.ncol)),
                            self._rand_int(top[1],
                                            min(top[1] + size[1], self.nrow))))

            # Don't place the object on top of another object
            if self.map[pos[0]][pos[1]] is not None:
                continue
            # ALREADY CHECKED THROUGH NONE_CHECK
            # Don't place the object where the agent is
            #pos_no_good = False
            #for a in range(self.n_agents):
            #    if np.array_equal(pos, self.agent_pos[a]):
            #    pos_no_good = True
            #if pos_no_good:
            #    continue

            # Check if there is a filtering criterion
            #if reject_fn and reject_fn(self, pos):
             #   continue

            break

        self.map[pos[0]][pos[1]] = obj

        #if obj is not None:
        #obj.init_pos = pos
        #obj.cur_pos = pos

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
        
       
       # print(self.map.tolist())
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
          map[row[a]][col[a]] = utils.colorize(map[row[a]][col[a]], COLOURS[a], highlight=True)
          
          if self.lastaction[a] is not None:
              outfile.write(f"Agent {a}: ({['Left', 'Down', 'Right', 'Up'][self.lastaction[a]]})\n")
          else:
              outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in map) + "\n")

        with closing(outfile):
            return outfile.getvalue()

if __name__=="__main__":
    fl = MultiFrozenLakeEnv(render_mode='human', is_slippery = False)
    #print(fl.map)
    
    #print("map")
    #map[0][0] = "h"
    #print(map)
    fl.reset()
    #fl.step([LEFT, DOWN, LEFT])
    #fl.step([RIGHT, DOWN, LEFT])
    #fl.step([RIGHT, DOWN, LEFT])
    #fl.step([RIGHT, DOWN, LEFT])
    #print(fl.action_space)
    #print(fl.map)
    #fl.render()
