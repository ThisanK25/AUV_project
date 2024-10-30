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

        self._chemical_dataset: Dataset = chem_utils.load_chemical_dataset(chem_data_path)
        self._confined: bool = confined
        self._x_size: tuple[int, int] = x_bounds
        self._y_size: tuple[int, int] = y_bounds
        self._reading_radius: int = reading_radius
        self._collected_data = np.full((x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]), np.inf, dtype=np.float64)
        self._current_classification_limit: list[float] = self._set_classification_limit(depth)
        self._depth: int = depth

    def _set_classification_limit(self, depth) -> list[float]:
        ph_69 = [7.7, 7.5] 
        ph_68 = [7.74, 7.62] 
        ph_67 = [7.76, 7.69] 
        ph_66 = [7.77, 7.745] 
        ph_65 = [7.772, 7.755] 
        ph_64 = [7.773, 7.766]
        
        current_pH_classification_limits = []
        
        match depth:
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

    def get_state_from_position(self, pos, heading):
        return tuple(map(self._classify_pH, self.get_current_pH_values(pos, heading)))

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
        right_pH = self._get_pH_at_position(right_pos, time_target)
        left_pH = self._get_pH_at_position(left_pos, time_target)
        front_pH = self._get_pH_at_position(front_pos, time_target)

        return right_pH, left_pH, front_pH

    def _get_pH_at_position(self, pos:tuple[int, int, int], time_target:int) -> float:
        if not self.inbounds(pos):
            return float('inf')
        x, y, z = pos
        metadata = (x, y, z, time_target, self._reading_radius)

        avg_value = self._get_table_data_at_position(pos)
        if avg_value == float('inf'):
            avg_value, _ = chem_utils.extract_chemical_data_for_volume(self._chemical_dataset, metadata, data_variable="pH")
        
        # store the data in the table
        self._insert_data_value(avg_value, pos)
        return avg_value
    

    def _classify_pH(self, pH_value):
        """
        Classify pH value into discrete states:
        - Return 0 if pH > 7.77 (basic)
        - Return 2 if pH < 7.5 (acidic)
        - Return 1 if 7.5 <= pH <= 7.77 (neutral)
        - This is basic settings for depth: 69 meter
        """
        high, low = self._current_classification_limit
        if pH_value > high:
            return PH_Reading.HIGH
        elif pH_value < low:
            return PH_Reading.LOW
        else:
            return PH_Reading.MEDIUM


    def inbounds(self, pos:tuple[int, int, int]) -> bool:
        x, y, _ = pos
        return self._x_size[0] <= x < self._x_size[1] and \
               self._y_size[0] <= y < self._y_size[1]

    def _insert_data_value(self, avg_value:float, pos:tuple[int, int, int]) -> None:
        x, y , _= pos
        self._collected_data[x - self._x_size[0]][y - self._y_size[0]] = avg_value

    def _get_table_data_at_position(self, pos:tuple[int, int, int]) -> float:
        if not self.inbounds(pos):
            return float('inf')
        x, y, _ = pos
        return self._collected_data[x-self._x_size[0]][y-self._y_size[0]]
    
    @property
    def min_pH_position(self) -> tuple[int, int, int]:
        min_pH_position = np.unravel_index(np.argmin(self._collected_data), self._collected_data.shape)
        
        return (min_pH_position[0] + self._x_size[0], min_pH_position[1] + self._y_size[0], self._depth)  
   
    @property
    def upper_left_corner(self) -> tuple[int, int, int]:
        return self._x_size[0], self._y_size[0], self._depth
    
    @property
    def lower_right_corner(self) -> tuple[int, int, int]:
        return self._x_size[1]-1, self._y_size[1]-1, self._depth
    
        