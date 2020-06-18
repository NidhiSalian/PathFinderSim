import numpy as np
import copy 
from matplotlib import pyplot

class Scene(object):

    def __init__(self, env):
        self.clear()
        self.grid_size = env.grid_size
        self.grid = self.get_matrix(env.grid_size, env.obstacles, env.agent_pos, env.target)
        self.directional_update = {'UP': (-1, 0), 'DOWN' : (1,0), 'LEFT' : (0,-1), 'RIGHT': (0,1) }
        self.current_agent_pos = copy.deepcopy(env.agent_pos)
        self.last_heuristic_update = float('inf') 
        self.trap_trigger_max = 4
        self.trap_trigger_count = 0

        self.ax = pyplot.figure().add_subplot()

    
    def get_matrix(self, grid_size, obstacles, agent_pos, goal):
        grid = np.zeros([grid_size, grid_size], dtype = int)
        for obstacle in obstacles:
            grid[tuple(obstacle)] = -100
        grid[tuple(agent_pos)] = 50
        grid[tuple(goal)] = 100
        return grid

    def update(self, agent_pos):
        self.grid[tuple(self.current_agent_pos)] = 0
        self.current_agent_pos = copy.deepcopy(agent_pos)
        self.grid[tuple(agent_pos)] = 50
    
    def clear(self):
        pyplot.close()

    def obstacle_ahead(self, agent_dir):
               
        new_pos = tuple(map(lambda i, j: i + j, \
            tuple(self.current_agent_pos), self.directional_update[agent_dir]))

        if(new_pos[0] < 0 or new_pos[0] >= self.grid_size):
            return True
        elif(new_pos[1] >= self.grid_size or new_pos[1] < 0):
            return True
        return bool(self.grid[new_pos] == -100)
    
    def trapped(self, heuristic):
        if heuristic >= self.last_heuristic_update:
            self.trap_trigger_count += 1
            if self.trap_trigger_count >= self.trap_trigger_max :
                return True
        return False
    
    def heuristic_based_action(self, target):
        min_distance_dir = None
        min_distance = float('inf') 
        for direction in self.directional_update:
            new_pos = tuple(map(lambda i, j: i + j, \
                        tuple(self.current_agent_pos), self.directional_update[direction]))
            euclid_distance = np.linalg.norm(np.array(new_pos) - np.array(target))
            
            if euclid_distance < min_distance:
                min_distance_dir = direction
                min_distance = euclid_distance
        current_heuristic_observation = round(min_distance, 2)
        agent_trapped = self.trapped(current_heuristic_observation)
        self.last_heuristic_update = current_heuristic_observation
        return agent_trapped, min_distance_dir



    def display(self, timestep, agent_pos, action):
        
        self.update(agent_pos)
        pyplot.cla()
        self.ax.title.set_text("Step : " + str(timestep) +" Action : " + action + " Target Distance : " + str(self.last_heuristic_update))
        self.ax.imshow(self.grid)
        pyplot.draw()
        pyplot.pause(0.02)