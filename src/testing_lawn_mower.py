# %%
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable
import numpy as np
from QAgent_Enums import Direction, AUV_ACTIONS
from Q_environment import Q_Environment
from reward_funcs import reward_gas_level
from utils import lawnmower_path as lp
from QAgent_new import Q_Agent
from utils import chem_utils
from Q_trainer import Q_trainer


def plot_lawnmower_pattern(coordinates):
    """
    Plots the lawnmower pattern based on provided coordinates.
    
    Args:
    coordinates (list of tuple): List of tuples containing x and y coordinates.
    """
    
    # Unzip the list of coordinates into two lists: x_values and y_values
    x_values, y_values = zip(*coordinates)
    
    # Plot the lawnmower pattern
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    
    # Adding labels and title to the plot
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Lawnmower Pattern')
    
    # Optionally add a grid
    plt.grid(True)
    
    # Show the plot
    plt.show()


def plot_lawnmower_with_gas_levels(chemical_dataset, data_parameter: str, depth: int, time_target: int, coordinates: list[tuple[int, int]]) -> None:
    """
    Plots the gas levels at a given depth and overlays the lawnmower pattern.
    
    Args:
    chemical_dataset: The chemical dataset containing the gas data.
    data_parameter (str): The parameter to plot (e.g., 'pH', 'pCO2').
    depth (int): The depth at which to plot the gas levels.
    time_target (int): The time slice to use for the plot.
    coordinates (list of tuple): List of tuples containing x and y coordinates for the lawnmower pattern.
    """
    val_dataset = chemical_dataset[data_parameter].isel(time=time_target, siglay=depth)
    val = val_dataset.values[:72710]
    x = val_dataset['x'].values[:72710]
    y = val_dataset['y'].values[:72710]
    x = x - x.min()
    y = y - y.min()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot gas levels
    scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Value')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title(f'{data_parameter} at {depth}m depth')

    # Plot lawnmower pattern
    x_values, y_values = zip(*coordinates)
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Lawnmower Path')
    
    plt.legend()
    plt.show()



# %%

# Example main section where you create the environment and agent, and run the agent
if __name__ == "__main__":
    path = r"./sim/SMART-AUVs_OF-June-1c-0002.nc"
    env = Q_Environment(Path(path), depth=68) 
    trainer = Q_trainer(env)
    trainer.train(episodes=1, max_steps_per_episode=100)
    agent = Q_Agent(env)
    chemical_dataset = chem_utils.load_chemical_dataset(path)

    data_parameter = 'pH'  # Adjust parameter as needed
    time_target = 7  # Adjust time slice as needed
    depth = 68  # Adjust depth as neede
    
    # Run the perform_cartesian_lawnmower method to generate the path
    agent.perform_cartesian_lawnmower(turn_length=10)
    plot_lawnmower_with_gas_levels(chemical_dataset, data_parameter, depth, time_target, agent._actions_performed)

    
    # Plot the lawnmower pattern
    plot_lawnmower_pattern(agent._actions_performed)

# %%

