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
                 reward_func = reward_gas_level,
                 start_position = None) -> None:
        
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
        self._position: tuple[int, int, int] = start_position if start_position else env.upper_left_corner
        self._heading = Direction.North

        self._actions_performed: list = []

    def run(self, lawnmower_size, max_steps = 100) -> None:
        self.perform_cartesian_lawnmower(lawnmower_size)

        self._move_to_max_gas_value()
        print(self._position)
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
        return next_state
    
    def perform_action(self, action) -> tuple:
        action = AUV_ACTIONS(action)
        heading = self._heading.value
        if action == AUV_ACTIONS.RIGHT:
            self._heading = Direction((heading+1)%4)
        
        if action == AUV_ACTIONS.LEFT:
            self._heading = Direction((heading - 1) % 4) 
        print(self._heading, action)
        return self._move_forward()


    def _next_position(self) -> tuple[int, int, int]:
        x, y, z = self._position
        match self._heading:
            case Direction.North:
                new_pos = (x, y - 1, z)
            case Direction.South:
                new_pos = (x, y + 1, z)
            case Direction.East:
                new_pos = (x + 1, y, z)
            case Direction.West:
                new_pos = (x-1, y, z)
        return new_pos
    
    def _move_forward(self) -> tuple:
        new_pos = self._next_position()
        if self._env.inbounds(new_pos):
            self._position = new_pos # We can throw a ValueError here to catch during training.
            self._actions_performed.append(new_pos[:2])
        return tuple(map(lambda x: x.value, self._env.get_state_from_position(self._position, self._heading)))


    def _move_to_max_gas_value(self) -> None:
        """
        Moves the AUV to a target location. OBS this allows 180 degree turn
        input:
          target : tuple[int, int, int] - x, y, z coords
        output: None
        """
        x_target, y_target , _= self._env.min_pH_position
        x, y, _ = self._position
        x_dir = Direction.East  if x_target - x  > 0 else Direction.West
        y_dir = Direction.North if y_target - y < 0 else Direction.South
        self._heading = x_dir
        while x_target - self._position[0] != 0:
            self._move_forward()
        

        self._heading = y_dir
        while y_target - self._position[1] != 0:
            self._move_forward()
        
    def perform_cartesian_lawnmower(self, turn_length:int = 70, start_direction: Direction = Direction.East) -> None:
        def move_east():
            while self._env.inbounds(self._next_position()):
                self._move_forward()
            self._heading = Direction.South
        
        def move_west():
            while self._env.inbounds(self._next_position()):
                self._move_forward()

            self._heading = Direction.South
        
        def move_south(turn_dir:Direction):
            count = 0
            while self._env.inbounds(self._next_position()) and count < turn_length:
                self._move_forward()
                count += 1
            self._heading = Direction.East if turn_dir == Direction.West else Direction.West
        
        self._heading = start_direction
        previous_heading = self._heading
        while self._position != self._env.lower_right_corner:
            match self._heading:
                case Direction.East:
                    move_east()
                    previous_heading = Direction.East
                    
                case Direction.South:
                    move_south(previous_heading)
                    
                case Direction.West:
                    move_west()
                    previous_heading = Direction.West

    @property
    def q_table(self):
        return self._q_table
    
    @q_table.setter
    def q_table(self, table):
        self._q_table = table


if __name__ == "__main__":
    env= Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc")) 

        



        







