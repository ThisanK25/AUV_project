from pathlib import Path
from matplotlib import pyplot as plt
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
        print(f"Saved figure: {figure_name}")
    else:
        plt.show()

    import os


def combined_plot():
    q_tables_dir = Path('results/q_tables')
    q_table_files = [f for f in q_tables_dir.iterdir() if f.is_file() and f.name.startswith('episode')]
    
    # Initialize lists to store episodes and gas accuracies
    episodes = []
    gas_accuracies = []

    for q_table_file in q_table_files:
        episode_number = int(q_table_file.stem.split('_')[1])
        q_table = load_q_table(q_table_file)
        env = Q_environment(list(fetch_sim_files())[0], depth=65)
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
        env = Q_environment(list(fetch_sim_files())[0], depth=65)
        sim = Q_Simulator(env)
        sim.test_agent(reward_func=reward_funcs.reward_trace_area, policy=policy_funcs.episilon_greedy, q_table=q_table)
        position_history = sim.agent.position_history
        axs[1].subplot(3, 1, idx + 1)
        plot_agent_behavior(position_history, 'path_to_chemical_dataset_file', time_target=0, z_target=0, zoom=True)

    plt.tight_layout()
    plt.show()

# Call the combined plot function to visualize the results
if __name__ == "__main__":
    combined_plot()