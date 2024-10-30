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

# Example main section where you create the environment and agent, and run the agent
if __name__ == "__main__":
    env = Q_Environment(Path(r"../sim/SMART-AUVs_OF-June-1c-0002.nc")) 
    agent = Q_Agent(env)
    
    # Run the perform_cartesian_lawnmower method to generate the path
    agent.perform_cartesian_lawnmower()
    
    # Plot the lawnmower pattern
    plot_lawnmower_pattern(agent._actions_performed)

# %%
