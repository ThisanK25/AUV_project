from pathlib import Path
from typing import Callable
import numpy as np
from QAgent_Enums import Direction, AUV_ACTIONS
from Q_environment import Q_Environment
from reward_funcs import reward_gas_level
from utils import lawnmower_path as lp

class Q_Agent:
    def __init__(self, 
                 env:Q_Environment, 
                 alpha:float   = 0.1, 
                 gamma:float   = 0.9, 
                 epsilon:float = 0.1,
                 reward_func = reward_gas_level) -> None:
        
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

    def run(self, max_steps= 100):
        # ! Lawnmower goes here
        # self._move_to_max_gas_value()
        # min_ph = self._env.min_pH_position
        reward = 0
        for step in range(max_steps):
            current_state = self._env.get_state_from_position(self._position, self._heading)
            # Convert state to tuple of integers

            current_state = tuple(map(lambda x: x.value, current_state))
            action = self.choose_action(current_state)
            next_state = self.execute_action(action)
            reward += self._reward_function(self, next_state)
            self.update_q_table(current_state, action, reward, next_state)
            current_state = next_state

    def choose_action(self, state:tuple[int, int, int]) -> int:
        if np.random.rand() < self._epsilon:
            return np.random.choice(3)
        return np.argmax(self.q_table[state])
    
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
        return tuple(map(lambda x: x.value, self._env.get_state_from_position(self._position, self._heading)))


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
        
    def generate_lawnmower_path(self, width=10, min_turn_radius=5, direction='y'):
        """
        Generates a lawnmower path and collects initial data.
        """

        # Generate waypoints for the lawnmower pattern
        x_data = np.linspace(0, 250, 100)
        y_data = np.linspace(0, 250, 100)
        waypoints, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
            x_data, y_data, width, min_turn_radius, 68, direction
        )
        print(x_coords)
        x_coords = zip(map(int, x_coords), map(int, y_coords))
        print([x for x in x_coords])
        # Simulate moving and collecting data along the waypoints
        #for x, y, z in waypoints:
        #    self.x, self.y, self.z = x, y, z
        #    pH_readings = self._env.get_current_pH_values(self._position, self._heading)
        #    self.collected_data.append((x, y, z, pH_readings))

    @property
    def q_table(self):
        return self._q_table
    
    @q_table.setter
    def q_table(self, table):
        self._q_table = table


if __name__ == "__main__":
    env= Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc")) 

        



        







