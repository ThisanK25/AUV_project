from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from xarray import DataArray
from utils import chem_utils


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
        raise ValueError(f"Episodes trained does not match gas accuracy {len(episodes_trained)=} {len(gas_accuracy)=}")
    
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
    # ? Hvorfor 51? Dette bør egentlig sttes dynamisk.
    episodes = [len(gas_accuracy)]

    figure_names = [q_table_names[i-1] for i in episodes_to_plot]
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1.5])

    # Plot gas accuracy on the top
    ax_gas_accuracy = fig.add_subplot(gs[0, 0])
    plot_gas_accuracy_vs_episodes(ax_gas_accuracy, episodes, gas_accuracy)

    for i, episode in enumerate(episodes_to_plot):
        if episode < len(agent_behavior):
            # TODO dette kan ikke være hardkoda på denne måten
            chemical_file_path = f"episode_{episode}_reward_trace_area_episilon_greedy_lawn_size_50"
            ax_agent_behavior = fig.add_subplot(gs[i + 1, 0])
            figure_name = figure_names[i]
            # skal dette skje hver episode?
            plot_agent_behavior_specific_episode(ax_agent_behavior, agent_behavior[episode], chemical_file_path, time_target, z_target, figure_name=figure_name)
        else:
            print(f"Episode {episode} not available. Maximum available episode is {len(agent_behavior) - 1}.")

    plt.tight_layout()

    plt.savefig(figure_name)
    plt.close()
    print(f"Saved figure: {figure_name}")
    
    plt.show()

def draw_environment(chemical_file_path, time_target, z_target, data_parameter='pH') -> tuple[Figure, Axes]:
    """
    Creates a base environment figure and returns the figure and axes.
    This can be used as a base for drawing.
    """
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
    
    # Plot environment
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2, alpha=0.6, label='Chemical Environment')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{data_parameter} Value')
    
    return fig, ax


def animate_agent_behavior(position_history, chemical_file_path, time_target, z_target, data_parameter='pH', zoom=False, gif_name=None, interval=10, head_color='red'):
    fig, ax = draw_environment(chemical_file_path, time_target, z_target, data_parameter)

    x_coords, y_coords = zip(*[(pos[0], pos[1]) for pos in position_history if pos[2] == z_target])

    # Initial state of the agent's path
    agent_path, = ax.plot([], [], 'k-', markersize=5, label='Agent Path')
    agent_head, = ax.plot([], [], 'o', markersize=8, color=head_color, label='Current Position')

    # TODO We can find some AUV-sprite to put here.
    #sprite_image = plt.imread(sprite_path)
    #sprite_artist = None
#
    #if sprite_image is not None:
    #    sprite_offset_image = OffsetImage(sprite_image, zoom=0.2)
    #    sprite_artist = AnnotationBbox(sprite_offset_image, (x_coords[0], y_coords[0]), frameon=False)
    #    ax.add_artist(sprite_artist)

    if zoom:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    def update(frame):
        agent_path.set_data(x_coords[:frame+1], y_coords[:frame+1])
        agent_head.set_data(x_coords[frame], y_coords[frame])
        return agent_path, agent_head,

    ani = FuncAnimation(fig, update, frames=range(len(x_coords)), interval=interval, blit=True)

    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    plt.title('Agent Path with Chemical Environment')
    plt.grid(True)
    plt.legend()

    if gif_name:
        gif_name = Path(gif_name)
        gif_name.parent.mkdir(exist_ok=True, parents=True)
        ani.save(gif_name, writer='imagemagick')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from QAgent_new import Q_Agent
    from Q_environment import Q_Environment
    from policy_funcs import episilon_greedy
    from Q_simulator import load_q_table

    env = Q_Environment(r"sim\SMART-AUVs_OF-June-1c-0002.nc", depth=66)
    agent = Q_Agent(env, policy=episilon_greedy)
    agent.q_table = load_q_table(r"results\q_tables\q_tables_by_episodes\episilon_greedy\episode_49_reward_trace_area_depth_67_lawn_size_50.pkl")
    agent.run(max_steps=500)
    animate_agent_behavior(agent.position_history, r"sim\SMART-AUVs_OF-June-1c-0002.nc", 0, 66)