from pathlib import Path
from typing import Callable
import numpy as np
from QAgent_Enums import Direction, AUV_ACTIONS
from Q_environment import Q_Environment
from reward_funcs import reward_gas_level
from utils import lawnmower_path as lp
import pickle # Too store the q-table

class QAgent:
    def __init__(self, 
                 env:Q_Environment, 
                 q_table_shape:tuple[int,int,int,int] = (3, 3, 3, 3), 
                 alpha:float   = 0.1, 
                 gamma:float   = 0.9, 
                 epsilon:float = 0.1,
                 reward_func = reward_gas_level) -> None:
        
        self._q_table = np.zeros_like(q_table_shape, dtype=np.int32)
        self._alpha: float   = alpha
        self._gamma: float   = gamma
        self._epsilon: float = epsilon

        
        # I prefere this setup with the callable as an argument
        self._reward_function: Callable[..., int] = reward_func
        # I dont like these.
        self._time_steps_in_high: int   = 0
        self._time_steps_in_medium: int = 0

        # metadata
        self._env = env
        self._position: tuple[int, int, int] = (0, 0, 0)
        self._heading = Direction.North

        self._actions_performed: list = []

    def choose_action(self, state) -> int:
        if np.random.rand() < self._epsilon: # soft_max?
            return np.random.choice(self.possible_actions)
        return np.argmax(self._q_table[state]) # Picks the best action
    
    def update_q_table(self, state:int, action:int, reward:int, next_state:int) -> None:
        current_q = self._q_table[state][action]
        maximum_future_reward = np.max(self._q_table[next_state])
        new_q = (1 - self._alpha) * current_q + self._alpha * (reward + self._gamma * maximum_future_reward)
        self._q_table[state][action] = new_q
    
    def execute_action(self, action) -> tuple:
        next_state: tuple = self.perform_action(action)
        self._actions_performed.append(action)
        return next_state
    
    def perform_action(self, action) -> tuple:
        if action == AUV_ACTIONS.RIGHT:
            self._heading = Direction(Direction.value((self._heading+1)%4))
        
        if action == AUV_ACTIONS.LEFT:
            self._heading = Direction(Direction.value((self._heading-1)%4))
        
        return self._move_forward()

    def _move_forward(self) -> tuple:
        x, y, z = self._position
        match self._heading:
            case Direction.North:
                new_pos = (x, y +1, z)
            case Direction.South:
                new_pos = (x, y - 1, z)
            case Direction.East:
                new_pos = (x + 1, y, z)
            case Direction.West:
                new_pos = (x-1, y, z)
        
        if self._env.inbounds(new_pos):
           self._position = new_pos # We can throw a ValueError here to catch during training.
        return self._env.get_current_pH_values(self._position, self._heading)


    def _move_to_max_gas_value(self) -> None:
        """
        Moves the AUV to a target location. OBS this allows 180 degree turn
        input:
          target : tuple[int, int, int] - x, y, z coords
        output: None
        """
        x_target, y_target , _= self._env.maximum_gas_position
        x, y, _ = self._position
        x_dir = Direction.East if x_target - x < 0 else Direction.West
        y_dir = Direction.North if x_target - x < 0 else Direction.South
        self._heading = x_dir
        while x_target - self._position[0] != 0:
            self._move_forward()
        
        self._heading = y_dir
        while y_target - self._position[1] != 0:
            self._move_forward()

        
    def generate_lawnmower_path(self, width, min_turn_radius, direction='y'):
        """
        Generates a lawnmower path and collects initial data.
        """
        # Generate waypoints for the lawnmower pattern
        x_data = np.linspace(self.x_min, self.x_max, 100)
        y_data = np.linspace(self.y_min, self.y_max, 100)
        waypoints, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
            x_data, y_data, width, min_turn_radius, self.z, direction
        )
        print(waypoints)
        # Simulate moving and collecting data along the waypoints
        for x, y, z in waypoints:
            self.x, self.y, self.z = x, y, z
            pH_readings = self._env.get_current_pH_values(self._position, self._heading)
            self.collected_data.append((x, y, z, pH_readings))

if __name__ == "__main__":
    env= Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc")) 
    agent = QAgent(env)
    agent.generate_lawnmower_path()

        



        







