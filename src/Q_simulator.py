from pathlib import Path
import pickle

import numpy as np
import reward_funcs
import policy_funcs
from tqdm import tqdm
from QAgent_Enums import PH_Reading
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
from AUV_plot_utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


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


def load_q_table(q_table_pkl_file:Path) -> np.ndarray:
    """
    Loads a q_table from a .pkl file 
    """
    with open(q_table_pkl_file, "rb") as q_paht:
        return pickle.load(q_paht)
    
def extract_q_table_files(reward_func, policy_func, lawn_size):
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
        # Key for sorting the files
        return int(path.stem.split("_")[1])

def load_q_tables_sorted_by_episode(reward_func, policy_func, lawn_size) -> map:
    """
    Fetches the stored_q_tables
    """
    q_files= extract_q_table_files(reward_func, policy_func, lawn_size)
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

def plot_results():
    q_tables_dir = Path('results/q_tables')
    q_table_files = [f for f in q_tables_dir.iterdir() if f.is_file() and f.name.startswith('episode')]
    
    # Initialize lists to store episodes and gas accuracies
    episodes = []
    gas_accuracies = []

    for q_table_file in q_table_files:
        episode_number = int(q_table_file.stem.split('_')[1])
        q_table = load_q_table(q_table_file)
        env = Q_Environment(list(fetch_sim_files())[0], depth=65)
        sim = Q_Simulator(env)
        gas_accuracy = sim.test_agent(reward_func=reward_funcs.reward_trace_area, policy=policy_funcs.episilon_greedy, q_table=q_table)

        episodes.append(episode_number)
        gas_accuracies.append(gas_accuracy)

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
    specific_episodes = [1, 25, 50]
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

def animate_lawnmower_and_actions(sim, save_path="./results/agent_animation.gif"):
    """
    Animates the agent's lawnmower path and subsequent actions on the environment in a pointwise manner,
    completing in 10 seconds and accounting for environment boundaries.
    Saves the animation as a GIF.
    """
    fig, ax = plt.subplots()
    ax.set_title("Lawnmower Path and Actions Performed in Environment (Pointwise)")
    
    # Environment bounds from Q_Environment
    x_min, x_max = sim._env._x_size
    y_min, y_max = sim._env._y_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Paths
    lawnmower_path = sim._agent.lawnmover_actions
    actions_path = sim._agent.actions_performed

    # Initialize scatter plots for each path
    lawnmower_scatter = ax.scatter([], [], color='blue', label="Lawnmower Path")
    actions_scatter = ax.scatter([], [], color='red', label="Actions Performed After")
    ax.legend()

    # Lists to store the points for each scatter plot
    lawnmower_x_data, lawnmower_y_data = [], []
    actions_x_data, actions_y_data = [], []

    # Update function for animation
    def update(frame):
        # Plot lawnmower path points up to the current frame
        if frame < len(lawnmower_path):
            x, y, *_ = lawnmower_path[frame]  # Extract x, y
            lawnmower_x_data.append(x)
            lawnmower_y_data.append(y)
            lawnmower_scatter.set_offsets(list(zip(lawnmower_x_data, lawnmower_y_data)))
        else:
            # Start plotting action points after lawnmower path is complete
            adjusted_frame = frame - len(lawnmower_path)
            if adjusted_frame < len(actions_path):  # Avoid exceeding length
                x, y, *_ = actions_path[adjusted_frame]
                actions_x_data.append(x)
                actions_y_data.append(y)
                actions_scatter.set_offsets(list(zip(actions_x_data, actions_y_data)))

        return lawnmower_scatter, actions_scatter

    # Calculate total frames and set the interval for 10 seconds duration
    total_frames = len(lawnmower_path) + len(actions_path)
    interval = 10000 / total_frames  # Duration in milliseconds divided by total frames

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Save the animation as a GIF using PillowWriter
    ani.save(save_path, writer=PillowWriter(fps=1000 / interval))
    plt.show()

def run_tests():
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
    # def plot_line_pointwise(x, y):
    #     # Create a figure and axis object
    #     plt.ion()  # Turn on interactive mode
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(min(x), max(x))  # Set x-axis limits
    #     ax.set_ylim(min(y), max(y))  # Set y-axis limits
    #     line, = ax.plot([], [], 'b-')  # Initialize an empty line (blue solid line)

    #     # Plot each point one by one to form the line
    #     for i in range(len(x)):
    #         line.set_data(x[:i+1], y[:i+1])  # Update the data for the line
    #         plt.draw()  # Redraw the plot
    #         plt.pause(0.05)  # Pause to update the plot, adjust timing as needed
        
    #     # Turn off interactive mode and show final plot
    #     plt.ioff()
    #     plt.show()

    # # Example usage
    # x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
    # y = np.sin(x)  # Sine wave values for y

    # plot_line_pointwise(x, y)

    env = Q_Environment(f"./sim/SMART-AUVs_OF-June-1c-0002.nc")
    agent = Q_Agent(env)
    agent.q_table = load_q_table("./results/q_tables/q_tables_by_episodes/episilon_greedy/episode_50_reward_trace_area_episilon_greedy_lawn_size_50.pkl")
    sim = Q_Simulator(env, agent)
    sim.test_agent(q_table=agent.q_table)
    animate_lawnmower_and_actions(sim)
