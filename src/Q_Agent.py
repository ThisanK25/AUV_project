from pathlib import Path
from typing import Callable
import numpy as np
from QAgent_Enums import Direction, AUV_ACTIONS, PH_Reading
from Q_environment import Q_Environment
from reward_funcs import reward_gas_level
from policy_funcs import episilon_greedy
from utils import lawnmower_path as lp

class Q_Agent:
    def __init__(self, 
                 env:Q_Environment, 
                 alpha:float   = 0.1, 
                 gamma:float   = 0.9, 
                 epsilon:float = 0.1,
                 temperature:float = 1.0,
                 reward_func = reward_gas_level,
                 policy = episilon_greedy,
                 start_position: tuple[int, int, int] | None = None) -> None:
        
        self._alpha: float       = alpha
        self._gamma: float       = gamma
        self._epsilon: float     = epsilon
        self._temperature: float = temperature
        
        # I prefere this setup with the callables as arguments
        self._reward_function: Callable[..., int] = reward_func
        self._policy = policy
       
        # I dont like these.
        self._time_steps_in_high: int   = 0
        self._time_steps_in_medium: int = 0

        # metadata
        self._env: Q_Environment = env
        self._position: tuple[int, int, int] = start_position if start_position else env.lower_left_corner
        self._heading = Direction.North
        
        self._visited = set()
        self._actions_performed: list = [self._position]
        self._lawnmover_actions: int  = 0
        
    def run(self, lawnmower_size:int=70, max_steps:int = 100) -> None:
        """
        Runs the agent for a given number of steps, starting with a lawnmover path with a given line distance.
        """
        self._env.register_new_agent()
        self.perform_cartesian_lawnmower(lawnmower_size)
        self._move_to(self._env.min_unvisited_position(self._visited))
        # Store the number of positions visited before choosing actions.
        self._lawnmover_actions = len(self._actions_performed)
        reward = 0
        num_bad_steps = 0
        for step in range(max_steps):
            current_state: tuple[PH_Reading, ...] = self._env.get_state_from_position(self._position, self._heading)
            bad_state = all((x == PH_Reading.HIGH for x in current_state))
            # TODO This is a bit ugly, but the following functions expect state as a tuple of integers. If I had time I would refactor this.
            current_state = tuple(map(lambda x: x.value, current_state))
            
            # ! If the agent havent found a good state for 15 steps it will find the best position it has seen, and move there. 
            # It still will only choose actions from its immediate neighbourhood, so I dont consider this breaching the first person architecture
            if bad_state:
                num_bad_steps += 1
                # ?  should these count as steps? This will pollute the action_performed list.
                if num_bad_steps == 15:
                    self._move_to(self._env.min_unvisited_position(self._visited))
            else:
                num_bad_steps = 0

            action = self._choose_action(current_state)
            next_state = self._execute_action(action)
            reward += self._reward_function(self, next_state)
            self._update_q_table(current_state, action, reward, next_state)
            current_state = next_state


    def _choose_action(self, state:tuple[int, int, int]) -> int:
        # Calls the policy function.
        return self._policy(self, state)
    
    def _update_q_table(self, state:int, action:int, reward:int, next_state:int) -> None:
        """
        Updates the q-table.
        """
        current_q = self._q_table[state][action]
        maximum_future_reward = np.max(self._q_table[next_state])
        new_q = (1 - self._alpha) * current_q + self._alpha * (reward + self._gamma * maximum_future_reward)
        self._q_table[state][action] = new_q
    
    def _execute_action(self, action) -> tuple:
        next_state: tuple = self._perform_action(action)
        return next_state
    
    def _perform_action(self, action) -> tuple:
        """
        Performes the action, and moves the agent.
        """
        action = AUV_ACTIONS(action)
        heading = self._heading.value
        if action == AUV_ACTIONS.RIGHT:
            self._heading = Direction((heading+1)%4)
        
        if action == AUV_ACTIONS.LEFT:
            self._heading = Direction((heading - 1) % 4) 
        return self._move_forward()


    def _next_position(self) -> tuple[int, int, int]:
        """
        Calculates the next potential position without bounds checking
        """
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
        """
        Adds the current position to the visited set, then moves the agent if it can. Else stay stationary. 
        """
        self._visited.add(self._position)
        new_pos: tuple[int, int, int] = self._next_position()
        if self._env.inbounds(new_pos):
            self._position = new_pos # We can throw a ValueError here to catch during training.
            self._actions_performed.append(new_pos[:2])
        return tuple(map(lambda x: x.value, self._env.get_state_from_position(self._position, self._heading)))


    def _move_to(self, gas_pos: tuple[int, int, int]) -> None:
        """
        Moves the AUV to a target location. OBS this allows 180 degree turn
        input:
          target : tuple[int, int, int] - x, y, z coords
        output: None
        """
        if not len(gas_pos) == 3:
            x_target, y_target = gas_pos
        else:
            x_target, y_target , _= gas_pos
        x, y, _ = self._position
        x_dir = Direction.East  if x_target - x  > 0 else Direction.West
        y_dir = Direction.North if y_target - y  < 0 else Direction.South
        self._heading = x_dir
        while x_target - self._position[0] != 0:
            self._move_forward()
        
        self._heading = y_dir
        while y_target - self._position[1] != 0:
            self._move_forward()
        
    def perform_cartesian_lawnmower(self, turn_length:int = 70, start_direction: Direction = Direction.East) -> None:
        """
        Performes lawnmover path with the supplide turn lenght from the current location of the AUV to the upper right corner of the environmnent.
        """
        def move_east() -> None:
            while self._env.inbounds(self._next_position()):
                self._move_forward()
            self._heading = Direction.South
        
        def move_west() -> None:
            while self._env.inbounds(self._next_position()):
                self._move_forward()

            self._heading = Direction.South
        
        def move_south(turn_dir:Direction) -> None:
            count = 0
            while self._env.inbounds(self._next_position()) and count < turn_length:
                self._move_forward()
                count += 1
            self._heading = Direction.East if turn_dir == Direction.West else Direction.West
        
        self._heading = start_direction
        previous_heading = self._heading
        while self._position != self._env.upper_right_corner:
            match self._heading:
                case Direction.East:
                    move_east()
                    previous_heading = Direction.East

                case Direction.South:
                    move_south(previous_heading)

                case Direction.West:
                    move_west()
                    previous_heading = Direction.West

    def visited(self) -> bool:
        """
        Checks if the AUV has been on its current position before.
        """
        return self._position in self._visited

    @property
    def q_table(self) -> np.ndarray:
        """
        Returns the q_table stored. If a Q_table is not set it creates a new table.
        """
        if not hasattr(self, "_q_table"):
            # Sets an empty q_table if non is set.
            self._q_table = np.zeros((3, 3, 3, 3), dtype=np.int32)

        return self._q_table
    
    @property
    def epsilon(self) -> float:
        """
        Returns the current value of epsilon.
        """
        return self._epsilon
    
    @property
    def lawnmover_actions(self) -> list:
        """
        List of the positions that was part of the lawnmover pattern
        """
        return self._actions_performed[:self._lawnmover_actions]

    @property
    def position_history(self) -> list:
        """
        complete list of positions
        """
        return self._actions_performed
    
    @property
    def actions_performed(self) -> list:
        """
        List of the actions taken after lawnmower
        """
        return self._actions_performed[self._lawnmover_actions:]
    
    
    @q_table.setter
    def q_table(self, table:np.ndarray) -> None:
        self._q_table: np.ndarray = table


    def gas_coords_visited(self, gas_coords: set) -> float:
        """
        Returns the ratio of gas coordinates visited.
        input: gas_coords : A set of (x, y, z) coordinates as tuples
        """
        if len(gas_coords) < 1:
            return 0 
        # The current location of the AUV is never in the set, so we add it here.
        self._visited.add(self._position)
        visited_after_lawn = self._visited - set(self.lawnmover_actions)

        num_visited_gas_coords = len(visited_after_lawn & gas_coords) 

        return num_visited_gas_coords / len(gas_coords)

if __name__ == "__main__":
    env= Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"))
