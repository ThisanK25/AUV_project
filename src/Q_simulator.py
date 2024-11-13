from pathlib import Path
import pickle
import random

import numpy as np
import reward_funcs
import policy_funcs
from tqdm import tqdm
from QAgent_Enums import PH_Reading
from Q_Agent import Q_Agent
from Q_environment import Q_Environment
from AUV_plot_utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
        
        # ? We should have generated complete numpy grids instead and used them through out, but too late now.
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
    

    def test_agent(self, max_steps=5000, reward_func = None, policy = None, q_table = None, start_position = None) -> float:
        """
        Test the frequency of witch the agent visits gas nodes
        """

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
    

def load_plume_map(plume_map_path:Path) -> set:
    """
    Loads the gas_node set from a pkl file. 
    """
    if not plume_map_path.exists():
        raise ValueError(f"invalid path: {plume_map_path}")
    with open(plume_map_path, "rb") as plume_map:
        return pickle.load(plume_map)

def load_all_plume_maps():
    """
    A generator of all plume maps in the ./sim/plume_map directory.
    This is populated as simumator environments are used.
    """
    for file in Path(r"./sim/plume_map").iterdir():
        if file.is_file():
            yield load_plume_map(file)

def load_q_table(q_table_pkl_file:Path) -> np.ndarray:
    """
    Loads a q_table from a .pkl file 
    """
    with open(q_table_pkl_file, "rb") as q_paht:
        return pickle.load(q_paht)
    
def extract_q_table_files(reward_func, policy_func, lawn_size) -> list:
    """
    Locates all q_tables that are matching the parameters.
    """
    reward_func_name: str = reward_func.__name__
    policy_func_name: str = policy_func.__name__
    reward_func_name_length: int = len(reward_func_name.split("_"))
    
    directory = Path(r"./results/q_tables/q_tables_by_episodes") / policy_func_name
    if not directory.exists():
    # I guess we could generate them here if we want to, but seems like a lot of work.
        raise FileExistsError("q_tables are not generated")

    q_files = []
    # Find all correct q_tables
    for file in filter(Path.is_file, directory.iterdir()):
        split_file: list[str] = file.stem.split("_")
        # Extract the reward function name to get the correct q-tables
        reward_name = "_".join(split_file[2:2+reward_func_name_length])
        if int(split_file[-1]) == lawn_size and reward_name == reward_func_name:
            q_files.append(file) 
    return q_files

def extract_episode_number(path:Path) -> int:
        """
        Returns the training episode number from a path, this can be used as a key for sorting files.
        """
        # Key for sorting the files
        return int(path.stem.split("_")[1])

def load_q_tables_sorted_by_episode(reward_func, policy_func, lawn_size) -> map:
    """
    Fetches the stored_q_tables sorted by episode number
    """
    q_files= extract_q_table_files(reward_func, policy_func, lawn_size)
    q_files.sort(key=extract_episode_number)
    return map(load_q_table, q_files)

def fetch_sim_files(directory = Path(r"./sim")) -> filter:
    """
    Returns a filter of files in the ./sim-directory. This should only be the raw environment simulation data.
    """
    return filter(Path.is_file, directory.iterdir())

def read_and_store_sim_files() -> None:
    """
    Reads a sim_file, and creates a simulation environment. This will generate a plume map matching the environment
    !!! This is slow
    """
    sim_files = list(fetch_sim_files())
    with tqdm(total=len(sim_files) * 6, ncols=100, desc="Reading files", bar_format='\033[0m{l_bar}{bar} \033[91m [elapsed: {elapsed} remaining: {remaining}]', colour='red', position=0) as pbar:
        for file_path in sim_files:
            for depth in range(64, 70):
                env = Q_Environment(file_path, depth=depth)
                sim = Q_Simulator(env, Q_Agent(env))
                pbar.update(1)

def plot_results() -> None:
    # TODO
    q_tables_dir = Path('results/q_tables')
    q_table_files: list[Path] = [f for f in q_tables_dir.iterdir() if f.is_file() and f.name.startswith('episode')]
    
    # Initialize lists to store episodes and gas accuracies
    episodes = []
    gas_accuracies = []
    episodes_numbers_to_plot: list[int] = [0, random.randint(1, len(q_table_files)-1, len(q_table_files))] 
    agents_behaviours_to_plot: list[list[tuple[int, int, int]]] = []

    for idx, q_table_file in enumerate(q_table_files):
        episode_number = int(q_table_file.stem.split('_')[1])
        q_table = load_q_table(q_table_file)
        env = Q_Environment(list(fetch_sim_files())[0], depth=65)
        sim = Q_Simulator(env)
        gas_accuracy: float = sim.test_agent(reward_func=reward_funcs.reward_trace_area, policy=policy_funcs.episilon_greedy, q_table=q_table)
        episodes.append(episode_number)
        gas_accuracies.append(gas_accuracy)
        if idx in episodes_numbers_to_plot:
            agents_behaviours_to_plot.append(sim.agent.position_history)

    # Sort episodes and gas accuracies
    sorted_indices = sorted(range(len(episodes)), key=lambda k: episodes[k])
    episodes_sorted = [episodes[i] for i in sorted_indices]
    gas_accuracies_sorted = [gas_accuracies[i] for i in sorted_indices]

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Top Plot: Gas Accuracy vs Episodes
    axs[0].plot(episodes_sorted, gas_accuracies_sorted, marker='o', linestyle='-', color='b')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Gas Accuracy')
    axs[0].set_title('Gas Accuracy vs. Episodes')

    # Bottom Plot: Agent Behavior for Specific Episodes
    for idx, specific_episode in enumerate(specific_episodes):
        q_table_filename = f'episode_{specific_episode}_reward_trace_area_episilon_greedy_lawn_size_50'
        q_table_path = q_tables_dir / q_table_filename
        if not q_table_path.exists():
            continue
        q_table = load_q_table(q_table_path)
        env = Q_Environment(list(fetch_sim_files())[0], depth=65)
        sim = Q_Simulator(env)
        sim.test_agent(reward_func=reward_funcs.reward_trace_area, policy=policy_funcs.episilon_greedy, q_table=q_table)
        position_history = sim.agent.position_history
        axs[1].subplot(3, 1, idx + 1)
        plot_agent_behavior(position_history, 'path_to_chemical_dataset_file', time_target=0, z_target=0, zoom=True)

    plt.tight_layout()
    plt.show()

def run_tests() -> None:
    """
    Loads q-tables by episode, and plots the frequency of gas-nodes visited.
    """
    q_tables_by_episode: map = load_q_tables_sorted_by_episode(policy_func=policy_funcs.soft_max, reward_func=reward_funcs.reward_trace_area, lawn_size=50)
    q_table_names = extract_q_table_files(policy_func=policy_funcs.soft_max, reward_func=reward_funcs.reward_trace_area, lawn_size=50)
    q_table_names.sort(key=extract_episode_number)
    # Here we want to test on the other file (ot both?), but I only have the one.
    depth = 65
    env = Q_Environment(list(fetch_sim_files())[0], depth)
    sim = Q_Simulator(env)
    gas_accuracy = []
    agent_behavior: list[list] = []
    for q_table in q_tables_by_episode:
        gas_accuracy.append(sim.test_agent(reward_func=reward_funcs.reward_trace_area, max_steps=2, policy=policy_funcs.episilon_greedy, q_table=q_table))
        agent_behavior.append(sim.agent.position_history)
        #run_tests_and_plot_specific_episodes_combined(gas_accuracy=gas_accuracy, agent_behavior=agent_behavior, z_target=depth, episodes_to_plot=[1, 25, 50], q_table_names = q_table_names)

if __name__ == "__main__":
    run_tests()
