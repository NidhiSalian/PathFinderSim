import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from create_scene import Scene

class FooEnv(gym.Env):

    metadata = {'render.modes': ['human', 'console']}
    ACTIONS = {
        0: 'IDLE',
        1: 'UP',
        2: 'DOWN',
        3: 'LEFT',
        4: 'RIGHT'}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, grid_size=20, agent_start_pos=None, agent_start_dir=None, target = None):
        super(FooEnv, self).__init__()
        # Size of the 1D-grid
        self.grid_size = grid_size
        #Starting position
        self.agent_start_pos = agent_start_pos
        self.agent_dir = agent_start_dir
        self.start_target = target
        
        self.n_obstacles = random.randint(self.grid_size, self.grid_size * 5)
        self.obstacles = np.random.randint(low = 1, high = self.grid_size, size=(self.n_obstacles, 2))
        if self.start_target is None:
            self.target = np.random.randint(low = 1, high = self.grid_size, size=(2))
        else:
            self.target = self.start_target
        

        # Initialize the agent at the starting position
        if self.agent_start_pos is None:
            self.agent_pos = np.random.randint(low = 1, high = self.grid_size, size=(2))
        else:
            self.agent_pos = agent_start_pos
        self.scene = Scene(self)
        if self.agent_dir is None:
            self.reset_direction()
        self.action = 'IDLE'
        self.timesteps = 0
        self.observation_interval = self.grid_size
        
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # The observation will be the coordinate of the agent
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                        shape=(2,), dtype=np.float32)
    
    def step(self, action):
        if action == 'IDLE':
            self.agent_pos= self.agent_pos
        elif action == 'UP':
            self.agent_pos[0] -= 1
        elif action == 'DOWN':
            self.agent_pos[0] += 1
        elif action == 'LEFT':
            self.agent_pos[1] -= 1
        elif action == 'RIGHT':
            self.agent_pos[1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size -1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size -1)

        # Are we at the left of the grid?
        done = bool(self.agent_pos[0] == self.target[0] and self.agent_pos[1] == self.target[1])

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if done else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        self.timesteps += 1

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def reset_direction(self, target_direction = None): #TODO
        if self.agent_dir is None:
            self.agent_dir = self.ACTIONS[random.randint(1, len(self.ACTIONS) -1)]
        elif target_direction is None:
            while self.obstacle_ahead():
                self.agent_dir = self.ACTIONS[random.randint(1, len(self.ACTIONS) -1)]
        else:
            self.agent_dir = target_direction
    
    def goal_reached(self): #TODO
        ...
    
    def obstacle_ahead(self): 
        return self.scene.obstacle_ahead(self.agent_dir)
    
    def optimal_action(self):
        
        # heuristic based goal seeking behavior
        if(self.timesteps % (self.observation_interval+1) == 0):
            agent_trapped, goal_direction = self.scene.heuristic_based_action(self.target)
            self.reset_direction(goal_direction)
            if agent_trapped:
                print("The agent is trapped!")
                self.observation_interval = self.observation_interval * random.randint(5, 15)
                self.reset_direction()
            else:
                self.observation_interval = self.observation_interval//3

        # avoid obstacles
        if(self.obstacle_ahead()):
            self.action = 'IDLE' 
            self.reset_direction()
            self.action = self.agent_dir
        else:
            # continue to wander
            self.action = self.agent_dir
        return self.action
        

    def reset(self):
        #Reset obstacles and target
        self.n_obstacles = random.randint(self.grid_size, self.grid_size * 3)
        self.obstacles = np.random.randint(low = 1, high = self.grid_size, size=(self.n_obstacles, 2))
        if self.start_target is None:
            self.target = np.random.randint(low = 1, high = self.grid_size, size=(2))
        else:
            self.target = self.start_target

        #Reset the agent 
        
        if self.agent_start_pos is None:
            self.agent_pos = np.random.randint(low = 1, high = self.grid_size, size=(2))
        else:
            self.agent_pos = self.agent_start_pos
        self.scene = Scene(self)
        if self.agent_dir is None:
            self.reset_direction()
        self.action = 'IDLE'
        self.timesteps = 0
        self.observation_interval = self.grid_size
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)


    def render(self, mode='console'):
        if mode != 'console':
            self.scene.display(self.timesteps, self.agent_pos, self.action)

        else:
            print("\n")

            # agent is represented as a cross, rest as a dot
            for i in range(self.grid_size):
                if i != (self.agent_pos[0] + 1):
                    print("o  " * self.grid_size, end="")
                else:
                    print("o  " * self.agent_pos[1], end="")
                    print("x  ", end="")
                    print("o  " * (self.grid_size - self.agent_pos[1] - 1) , end="")
                print("\n")

    def close(self):
        self.scene.clear()
        







