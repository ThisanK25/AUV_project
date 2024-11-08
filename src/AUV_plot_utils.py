from matplotlib import pyplot as plt
from xarray import DataArray
import chem_utils


def plot_agent_behavior(q_trainer, chemical_file_path, time_target, z_target, data_parameter='pH', zoom=False, figure_name=None):
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
    if hasattr(q_trainer, 'position_history'):
        x_coords, y_coords = zip(*q_trainer.position_history)
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
        plt.savefig(figure_name)
        plt.close()
        print(f"Saved figure: {figure_name}")
    else:
        plt.show()