from pathlib import Path
from pprint import pprint
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
from policy_funcs import soft_max, episilon_greedy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import chem_utils


class Q_trainer:
    def __init__(self, env: Q_Environment, q_table_shape=(3, 3, 3, 3), policy = episilon_greedy) -> None:
        self._env = env
        self._q_table = np.zeros(q_table_shape, dtype=np.int32)
        self._policy = policy

    def train(self, episodes=10, max_steps_per_episode=2000, lawnmower_size=70):
        for episode in range(episodes):
            agent = Q_Agent(self._env, policy=self._policy)
            agent.q_table = self._q_table
            agent.run(lawnmower_size=lawnmower_size, max_steps=max_steps_per_episode)
            self._q_table = agent.q_table
            print(f"Episode {episode + 1}/{episodes} completed.")
        pprint(self._q_table)
        
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self._q_table, f)
        self._save_position_history(agent)

    def _save_position_history(self, agent: Q_Agent):
        self.position_history = agent._actions_performed

    def plot_behavior(self, chemical_file_path, time_target, z_target, data_parameter='pH', zoom=False) -> None:
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
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2, alpha=0.6, label='Chemical Environment')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f'{data_parameter} Value')

        if hasattr(self, 'position_history'):
            x_coords, y_coords = zip(*self.position_history)
            ax.plot(x_coords, y_coords, marker='o', color='black', label='Agent Path')

            for i, (x_pos, y_pos) in enumerate(self.position_history):
                ax.annotate(f'{i}', (x_pos, y_pos))

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
        plt.show()


if __name__ == "__main__":
    for depth in range(64, 70):
        for lawn_size in range(20, 101, 20):
            env = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=depth,)
            trainer = Q_trainer(env, policy = soft_max)
            trainer.train(episodes=3, max_steps_per_episode=1000, lawnmower_size=lawn_size)

            # Example of plotting behavior (adjust the chemical_file_path, time_target, etc. as necessary)
            trainer.plot_behavior(
                chemical_file_path=r"./sim/SMART-AUVs_OF-June-1c-0002.nc",  
                time_target=0,
                z_target=depth,
                data_parameter='pH',
                zoom=True
            )

