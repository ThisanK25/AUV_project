from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from xarray import DataArray
from utils import chem_utils
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from Q_environment import Q_Environment
from QAgent_new import Q_Agent
import reward_funcs
import policy_funcs
import Q_trainer

def plot_agent_behavior(position_history, chemical_file_path, time_target, z_target, data_parameter='pH', zoom=False, figure_name=None) -> None:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "Dejavu Serif",
        "mathtext.fontset": "dejavuserif"
    })
    # Load chemical dataset
    chemical_dataset = chem_utils.load_chemical_dataset(chemical_file_path)
    # Extract data to plot the environment
    val_dataset: DataArray = chemical_dataset[data_parameter].isel(time=time_target, siglay=z_target)
    val = val_dataset.values[:72710]
    x = val_dataset['x'].values[:72710]
    y = val_dataset['y'].values[:72710]
    x = x - x.min()
    y = y - y.min()
    # Plot environment and agent's path on the same axes
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2, alpha=0.6, label='Chemical Environment')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{data_parameter} Value')
        
    x_coords, y_coords = zip(*position_history)
    ax.plot(x_coords, y_coords, marker='o', color='black', label='Agent Path')
    #for i, (x_pos, y_pos) in enumerate(self.position_history):
    #    ax.annotate(f'{i}', (x_pos, y_pos))
    if zoom:
        # Calculate bounds
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        # Set plot limits
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    
    # Add labels and title
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    plt.title('Agent Path with Chemical Environment')
    plt.grid(True)
    plt.legend()
    if figure_name:
        figure_name = Path(figure_name)
        figure_name.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_name)
        plt.close()
    else:
        plt.show()  


def plot_gas_accuracy_vs_episodes(ax, episodes_trained, gas_accuracy):
    if len(episodes_trained) != len(gas_accuracy):
        raise ValueError("The length of episodes_trained should match the length of gas_accuracy")
    
    ax.plot(episodes_trained, gas_accuracy, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Episodes Trained')
    ax.set_ylabel('Gas Accuracy')
    ax.set_title('Gas Accuracy vs Episodes Trained')
    ax.grid(True)

def plot_agent_behavior_specific_episode(ax, position_history, chemical_file_path, time_target, z_target, data_parameter='pH', zoom=False):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "Dejavu Serif",
        "mathtext.fontset": "dejavuserif"
    })
    # Load chemical dataset
    chemical_dataset = chem_utils.load_chemical_dataset(chemical_file_path)
    # Extract data to plot the environment
    val_dataset = chemical_dataset[data_parameter].isel(time=time_target, siglay=z_target)
    val = val_dataset.values[:72710]
    x = val_dataset['x'].values[:72710]
    y = val_dataset['y'].values[:72710]
    x = x - x.min()
    y = y - y.min()
    # Plot environment and agent's path on the same axes
    scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2, alpha=0.6, label='Chemical Environment')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{data_parameter} Value')
    
    x_coords, y_coords = zip(*position_history)
    ax.plot(x_coords, y_coords, marker='o', color='black', label='Agent Path')

    if zoom:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title('Agent Path with Chemical Environment')
    ax.grid(True)
    ax.legend()

def run_tests_and_plot_specific_episodes_combined(gas_accuracy:list[float], agent_behavior:list[int], z_target:int, q_table_names, time_target:int=0, episodes_to_plot=[1, 25, 50]):
    
    episodes = [i for i in range(51)]
    # Fetching figurename 
    figure_names = [q_table_names[i] for i in [1, 25, 50]]
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1.5])

    # Plot gas accuracy on the top
    ax_gas_accuracy = fig.add_subplot(gs[0, 0])
    plot_gas_accuracy_vs_episodes(ax_gas_accuracy, episodes, gas_accuracy)

    for i, episode in enumerate(episodes_to_plot):
        if episode < len(agent_behavior):
            chemical_file_path = f"episode_{episode}_reward_trace_area_episilon_greedy_lawn_size_50"
            ax_agent_behavior = fig.add_subplot(gs[i + 1, 0])
            figure_name = figure_names[i]
            plot_agent_behavior_specific_episode(ax_agent_behavior, agent_behavior[episode], chemical_file_path, time_target, z_target, figure_name=figure_name)
        else:
            print(f"Episode {episode} not available. Maximum available episode is {len(agent_behavior) - 1}.")

    plt.tight_layout()

    plt.savefig(figure_name)
    plt.close()
    print(f"Saved figure: {figure_name}")
    
    plt.show()

def animate_lawnmower_and_actions(env, agent, save_path="./results/agent_animation.gif"):
    """
    Animates the agent's lawnmower path and subsequent actions on the environment in a pointwise manner,
    completing in 10 seconds and accounting for environment boundaries.
    Saves the animation as a GIF.
    """
    fig, ax = plt.subplots()
    ax.set_title(f"Lawnmower Path and Further Exploration (Turn length=50; Depth={env.depth})")
    
    # Environment bounds from Q_Environment
    x_min, x_max = env._x_size
    y_min, y_max = env._y_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Plot gas field
    chemical_dataset = env._chemical_dataset
    val_dataset = chemical_dataset['pH'].isel(time=0, siglay=env.depth)
    val = val_dataset.values[:72710]
    x = val_dataset['x'].values[:72710]
    y = val_dataset['y'].values[:72710]
    x = x - x.min()
    y = y - y.min()
    
    # Plot gas levels
    scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Value')
    
    # Paths
    lawnmower_path = agent.lawnmover_actions
    actions_path = agent.actions_performed

    # Initialize scatter plots for each path
    lawnmower_scatter = ax.scatter([], [], color='black', label="Lawnmower Path")
    actions_scatter = ax.scatter([], [], color='white', label="Actions Performed After")
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
    ani.save(save_path, writer=PillowWriter(fps=5000 / interval))
    plt.show()


# Call the combined plot function to visualize the results
if __name__ == "__main__":
    env = Q_Environment("./sim/SMART-AUVs_OF-June-1c-0002.nc")
    agent = Q_Agent(env)
    trainer = Q_trainer.Q_trainer(env)
    trainer.train(lawnmover_size=50)
    agent.run(lawnmower_size=50, max_steps=2000)
    animate_lawnmower_and_actions(env, agent)