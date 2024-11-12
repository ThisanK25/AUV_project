from typing import Any, Generator
from QAgent_Enums import Direction, PH_Reading
from xarray import Dataset
from utils import chem_utils
import numpy as np
from pathlib import Path

class Q_Environment:
    def __init__(self, chem_data_path:str | Path, 
                 reading_radius:int = 5,
                 confined:bool=False, 
                 x_bounds=(0, 250), y_bounds=(0, 250),
                 depth=69) -> None:
        self._datapath: Path = Path(chem_data_path)
        self._chemical_dataset: Dataset = chem_utils.load_chemical_dataset(chem_data_path)
        self._confined: bool = confined
        self._x_size: tuple[int, int] = x_bounds
        self._y_size: tuple[int, int] = y_bounds
        self._reading_radius: int = reading_radius
        self._collected_data = np.full((x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]), np.inf, dtype=np.float64)
        self._depth: int = depth
        self._current_classification_limit: list[float] = self._set_classification_limit()
        self._seen_by_agent = set()

    def register_new_agent(self) -> None:
        """
        Reset the seen_by_agent set.
        """
        # TODO this should be moved to the agent-class, but to facilitate cashing of gas-values they where placed here
        # TODO Changig this should not be a to large task, but for now just rememeber to call this function if using multiple agents in the same environment
        self._seen_by_agent = set()

    def _set_classification_limit(self) -> list[float]:
        """
        Sets the classification limits for PH-values in accordance with the depth. 
        """
        ph_69 = [7.7, 7.5] 
        ph_68 = [7.74, 7.62] 
        ph_67 = [7.76, 7.69] 
        ph_66 = [7.77, 7.745] 
        ph_65 = [7.772, 7.755] 
        ph_64 = [7.773, 7.766]
        
        current_pH_classification_limits = []
        
        match self._depth:
            case 69:
                current_pH_classification_limits = ph_69
            case 68:
                current_pH_classification_limits = ph_68
            case 67:
                current_pH_classification_limits = ph_67
            case 66:
                current_pH_classification_limits = ph_66
            case 65:
                current_pH_classification_limits = ph_65
            case 64:
                current_pH_classification_limits = ph_64
        return current_pH_classification_limits

    def get_state_from_position(self, pos, heading) -> tuple[PH_Reading, ...]:
        """
        Classifies the PH-values that corresponds with the position and heading of the AUV.
        """
        return tuple(map(self.classify_pH, self.get_current_pH_values(pos, heading)))

    def get_current_pH_values(self, position, heading)-> tuple:
        """
        Return the actual pH values for the agent's front, left, and right positions (not the classified states).
        """
        time_target = 0  # Example: Use the first time step
        steps = 1  # Coordinate steps

        x, y, z = position
        # Define positions relative to the agent's current heading
        if heading == Direction.North:
            front_pos = (x, y + steps, z)
            left_pos = (x - steps, y, z)
            right_pos = (x + steps, y, z)
        elif heading == Direction.East:
            front_pos = (x + steps, y, z)
            left_pos = (x, y + steps, z)
            right_pos = (x, y - steps, z)
        elif heading == Direction.South:
            front_pos = (x, y - steps, z)
            left_pos = (x + steps, y, z)
            right_pos = (x - steps, y, z)
        elif heading == Direction.West:
            front_pos = (x - steps, y, z)
            left_pos =  (x, y - steps, z)
            right_pos = (x, y + steps, z)
        
        # Extract actual pH values (not classified), this stores the values in the grid
        right_pH = self.get_pH_at_position(right_pos, time_target)
        left_pH = self.get_pH_at_position(left_pos, time_target)
        front_pH = self.get_pH_at_position(front_pos, time_target)

        return right_pH, left_pH, front_pH

    def get_pH_at_position(self, pos:tuple[int, int, int], time_target:int) -> float:
        """
        Extracts the average ph_values for the three possible actions.
        The values are then cashed for future use befor returnd as a tuple.
        """
        if not self.inbounds(pos):
            return float('inf')
        x, y, z = pos
        metadata = (x, y, z, time_target, self._reading_radius)

        avg_value = self._get_table_data_at_position(pos)
        if avg_value == float('inf'):
            avg_value, _ = chem_utils.extract_chemical_data_for_volume(self._chemical_dataset, metadata, data_variable="pH")
        
        self._seen_by_agent.add(pos)
        # store the data in the table
        self._insert_data_value(avg_value, pos)
        return avg_value
    

    def classify_pH(self, pH_value) -> PH_Reading:
        """
        Classify pH value into discrete states as defined by the classification limit:
        LOW
        MEDIUM
        HIGH
        """
        high, low = self._current_classification_limit
        if pH_value > high:
            return PH_Reading.HIGH
        elif pH_value < low:
            return PH_Reading.LOW
        else:
            return PH_Reading.MEDIUM


    def inbounds(self, pos:tuple[int, int, int]) -> bool:
        """
        Checks if position is inbouds of the environment.
        """
        x, y, _ = pos
        return self._x_size[0] <= x < self._x_size[1] and \
               self._y_size[0] <= y < self._y_size[1]

    def _insert_data_value(self, avg_value:float, pos:tuple[int, int, int]) -> None:
        """
        Stores the datavalues in the _collected data table.
        """
        x, y , _= pos
        self._collected_data[x - self._x_size[0]][y - self._y_size[0]] = avg_value

    def _get_table_data_at_position(self, pos:tuple[int, int, int]) -> float:
        """
        Extracts the table data from a position. Infinity returned if out of bounds.
        """
        if not self.inbounds(pos):
            return float('inf')
        x, y, _ = pos
        return self._collected_data[x-self._x_size[0]][y-self._y_size[0]]
    
    def min_unvisited_position(self, visited: set) -> tuple[int, int, int]:
        """
        Finds the minimum ph-value that is not visited, but known by an agent.
        !!! If the encvironment is used for several agent, the agent must first call register_new_agent
        """
        min_value: float = np.inf
        min_position = None
        # The agent has not seen all the indexes in collected data, so we need to check if the current agent knows about a location.
        unexplored_coords:set = self._seen_by_agent - visited
        for index, value in np.ndenumerate(self._collected_data):
            if value < min_value and index + (self._depth,) in unexplored_coords:
                min_value = value
                min_position = index

        return min_position


    @property
    def min_pH_position(self) -> tuple[int, int, int]:
        """
        Returns the minimum explored value. This can only be used by single agent environments. 
        """
        min_pH_position = np.unravel_index(np.argmin(self._collected_data), self._collected_data.shape)
        
        return (min_pH_position[0] + self._x_size[0], min_pH_position[1] + self._y_size[0], self._depth)  
   
    @property
    def lower_left_corner(self) -> tuple[int, int, int]:
        return self._x_size[0], self._y_size[0], self._depth
    
    @property
    def upper_right_corner(self) -> tuple[int, int, int]:
        return self._x_size[1]-1, self._y_size[1]-1, self._depth
    
    @property
    def datapath(self) -> Path:
        return self._datapath
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @property
    def traverse_environment(self) -> Generator[tuple[int, int, int], Any, None]:
        """
        Returns a generator that moves across the entire environment. Can be used to map out all data values.
        """
        x_min, x_max = self._x_size
        y_min, y_max = self._y_size
        z = self._depth
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                yield x, y, z

    
        