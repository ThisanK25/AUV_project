from pathlib import Path
import pickle

import numpy as np
import reward_funcs
import policy_funcs
from tqdm import tqdm
from QAgent_Enums import PH_Reading
from QAgent_new import Q_Agent
from Q_environment import Q_Environment



class Q_Simulator:
    def __init__(self, env: Q_Environment, agent: Q_Agent | None = None) -> None:
        self._agent: Q_Agent     = agent
        self._env: Q_Environment = env
        self._gas_coords: set = self._find_gas_coords(env.datapath)

    
    def _find_gas_coords(self, datapaht: Path) -> set:
        """
        Reads every coordinate of the gas plume and returns the plume locations as a set.
        This is a slow operation so the set is written to a pickle file located at ./sim/plume_map/<file_name>_depth_<depth>.pkl
        """
        # ? Skulle sikkert ha lagra hele tabeller, og ikke bare gas-koordinatene. Settet kunne jeg ha lagd senere.
        if datapaht.parent != Path("sim"):
            raise ValueError(r"datapaths has to be on the form ./sim/<SMART_AUVs_*>")
        
        # Check if the plume data exists
        plume_map_path = datapaht.parent / "plume_map" / datapaht.stem
        plume_map_path = Path(f'{plume_map_path}_depth_{self._env.depth}.pkl')
        
        if plume_map_path.exists():
            with open(plume_map_path, "rb") as plume_map:
                return pickle.load(plume_map)
            
        gas_coords = set()        
        # Setting up a progress bar
        coords_list = list(self._env.traverse_environment)
        total_coords = len(coords_list)
        
        red_format = '{l_bar}{bar} \033[91m [elapsed: {elapsed} remaining: {remaining}]'

        with tqdm(total=total_coords, ncols=100, desc='Processing Coordinates', bar_format=red_format, colour='green', position=1) as pbar:
            for idx, coords in enumerate(coords_list):
                gas_val = self._env.get_pH_at_position(coords, 0)
                if self._env.classify_pH(gas_val) != PH_Reading.HIGH:
                    gas_coords.add(coords)
                pbar.update(1)

        # Create the path to the plume maps if they dont exists.
        plume_map_path.parent.mkdir(exist_ok=True, parents=True)
        with open(plume_map_path, "wb") as plume_map:
            pickle.dump(gas_coords, plume_map)
        return gas_coords
    



    def test_agent(self, max_steps=2000, reward_func = None, policy = None, q_table = None, start_position = None) -> float:
        """
        Test the frequency of witch the agent visits gas nodes
        """
        # We could create the agent here as well I suppose

        if reward_func is None:
            reward_func = reward_funcs.reward_trace_area
        if policy is None:
            policy = policy_funcs.episilon_greedy
        if start_position is None:
            start_position = (0, 0, self._env.depth)
        if q_table is None:
            q_table = np.zeros((3, 3, 3, 3))

        self._agent = Q_Agent(
            self._env,  
            reward_func=reward_func, 
            policy=policy, 
            start_position=start_position
        )
        
        # Set the trained Q-table if provided
        self._agent.q_table = q_table
        self._agent.run(max_steps = max_steps)
        return self._agent.gas_coords_visited(self._gas_coords)

    @property
    def agent(self) -> Q_Agent:
        return self._agent


def load_q_table(q_table_pkl_file:Path) -> np.ndarray:
    """
    Loads a q_table from a .pkl file 
    """
    with open(q_table_pkl_file, "rb") as q_paht:
        return pickle.load(q_paht)

def load_q_tables_sorted_by_episode(reward_func, policy_func, lawn_size) -> map:
    """
    Fetches the stored_q_tables
    """
    def extract_episode_number(path:Path) -> int:
        return int(path.stem.split("_")[1])

    reward_func_name: str = reward_func.__name__
    policy_func_name: str = policy_func.__name__
    reward_func_name_length: int = len(reward_func_name.split("_"))
    directory = Path(r"./results/q_tables/q_tables_by_episodes") / policy_func_name
    if not directory.exists():
        raise FileExistsError("q_tables are not generated")
    
    q_files = []
    # Find all correct q_tables
    for file in filter(Path.is_file, directory.iterdir()):
        split_file: list[str] = file.stem.split("_")
        # Extract the reward function name to get the correct q-tables
        reward_name = "_".join(split_file[2:2+reward_func_name_length])
        if int(split_file[-1]) == lawn_size and reward_name == reward_func_name:
            q_files.append(file)

    q_files.sort(key=extract_episode_number)
    return map(load_q_table, q_files)

def fetch_sim_files(directory = Path(r"./sim")) -> filter:
    return filter(Path.is_file, directory.iterdir())

def read_and_store_sim_files() -> None:
    sim_files = list(fetch_sim_files())
    with tqdm(total=len(sim_files) * 6, ncols=100, desc="Reading files", bar_format='\033[0m{l_bar}{bar} \033[91m [elapsed: {elapsed} remaining: {remaining}]', colour='red', position=0) as pbar:
        for file_path in sim_files:
            for depth in range(64, 70):
                env = Q_Environment(file_path, depth=depth)
                sim = Q_Simulator(env, Q_Agent(env))
                pbar.update(1)



if __name__ == "__main__":
    q_tables_by_episde: map = load_q_tables_sorted_by_episode(policy_func=policy_funcs.soft_max, reward_func=reward_funcs.reward_trace_area, lawn_size=50)

    env = Q_Environment(list(fetch_sim_files())[0], depth = 65)
    sim = Q_Simulator(env)
    for q_table in q_tables_by_episde:
        print(sim.test_agent(reward_func=reward_funcs.reward_trace_area, policy=policy_funcs.soft_max, q_table=q_table))

