from pathlib import Path
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from xarray import DataArray, Dataset
from utils import chem_utils
from PIL import Image, ImageDraw, ImageSequence

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
    chemical_dataset: Dataset = chem_utils.load_chemical_dataset(chemical_file_path)
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


def plot_agent_behavior(position_history, chemical_file_path, time_target, z_target, data_parameter='pH', figure_name=None, ax=None) -> None:
    if ax is None:
        fig, ax = draw_environment(chemical_file_path, time_target, z_target, data_parameter) 
    else:
        fig, _ = draw_environment(chemical_file_path, time_target, z_target, data_parameter)
        
    x_coords, y_coords = zip(*position_history)
    ax.plot(x_coords, y_coords, marker=',', color='black', label='Agent Path')
    
    # Add labels and title
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title('Agent Path with Chemical Environment')
    ax.grid(True)
    ax.legend()
    
    if figure_name:
        figure_name = Path(figure_name)
        figure_name.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_name)
        plt.close()
    else:
        if ax is None:
            plt.show()  

def plot_by_episodes(gas_accuracies, agents_behaviours_to_plot, depth, episodes, chemical_file_path = r"sim\SMART-AUVs_OF-June-1c-0002.nc", figure_name = None) -> None:
    plt.plot(episodes, gas_accuracies, label='Gas Accuracy')
    plt.set_xlabel('Episode')
    plt.set_ylabel('Gas Accuracy')
    plt.set_title('Gas Accuracy by Episode')
    plt.legend()
    if figure_name:
        figure_name = Path(figure_name + "_gas_plot.png")
        figure_name.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_name + "_gas_plot.png")
        plt.close()
    else:
        plt.show()
        plt.close()
    for idx, behaviour in enumerate(agents_behaviours_to_plot, start=1):
        plot_agent_behavior(behaviour, chemical_file_path=chemical_file_path, time_target=0, z_target=depth, data_parameter='pH', figure_name=f'{figure_name}_{idx}.png')

def animate_agent_behavior(position_history, chemical_file_path, time_target, z_target, data_parameter='pH', gif_name=None, interval=100, sprite_path=None) -> None:
    """
    Animates the agens behaviour. 
    !!! Saving the animations is a very intensive task. Running it without storing is fairly quick. !!!
    The sprite was supposed to move along the path, but it is currently not working as expected.
    """
    fig, ax = draw_environment(chemical_file_path, time_target, z_target, data_parameter)
    x_coords, y_coords = zip(*[(pos[0], pos[1]) for pos in position_history])
    agent_path, = ax.plot([], [], 'k-', markersize=5, label='Agent Path')
    
    if sprite_path is None:
        sprite_image = create_auv_sprite()
    else:
        if not Path(sprite_path).exists():
            sprite_image = create_auv_sprite(sprite_path)
        sprite_image = plt.imread(sprite_path)
    
    sprite_offset_image = OffsetImage(sprite_image, zoom=0.2)
    sprite_artist = AnnotationBbox(sprite_offset_image, (x_coords[0], y_coords[0]), frameon=False)
    ax.add_artist(sprite_artist)    

    def update(frame)->list:
        agent_path.set_data(x_coords[:frame+1], y_coords[:frame+1])
        sprite_artist.xy = (x_coords[frame], y_coords[frame])
        return [agent_path, sprite_artist]

    ani = FuncAnimation(fig, update, frames=range(len(x_coords)), interval=interval, blit=True)

    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    plt.title('Agent Path with Chemical Environment')
    plt.grid(True)
    plt.legend()

    if gif_name:
        gif_name = Path(gif_name)
        gif_name.parent.mkdir(exist_ok=True, parents=True)
        try:
            writer = PillowWriter(fps = 30000 // interval)
            ani.save(gif_name, writer=writer)
        except ValueError as e:
            print(f"Error in saving animation: {e}")
            plt.close()
    else:
        plt.show()


def create_auv_sprite(filename=None) -> Image.Image:
    """
    Draws a simple sprite of a yellow submarine. Don't sue us.
    """
    image = Image.new("RGBA", (200, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    uboat_color = "yellow"

    draw.rectangle([60, 50, 140, 80], fill=uboat_color)
    draw.ellipse([40, 50, 80, 80], fill="red")
    draw.ellipse([120, 50, 160,80], fill="blue") 

    draw.rectangle([90, 40, 110, 50], fill=uboat_color)

    draw.line([100, 30, 100, 40], fill="black", width=2)

    if filename:
        if not Path(filename).exists():
            image.save(filename)
    
    return image

def remove_frames_from_gif(input_path:Path, output_path:Path, frames_to_remove:list[int], num_frames_to_skip = 0) -> None:
    """
    Removes frames from a gif.
    input:
        input_path : The gif to edit
        output_path : The path to the new gif
        frames_to_remove : Number of frames to skip in the new gif, effectivly speeding it up
    """

    gif = Image.open(input_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    frames_to_keep = [frame for i, frame in enumerate(frames) if i not in frames_to_remove and i % (num_frames_to_skip+1) == 0]
    frames_to_keep[0].save(output_path, save_all=True, append_images=frames_to_keep[1:], loop=0)


if __name__ == "__main__":
    from Q_Agent import Q_Agent
    from Q_environment import Q_Environment
    from policy_funcs import episilon_greedy
    from Q_simulator import load_q_table

    remove_frames_from_gif(Path(r".\results\gifs\actions.gif"), r".\results\gifs\actions3.gif", list(range(600)) + list(range(1750, 4000)))
    #env = Q_Environment(r"sim\SMART-AUVs_OF-June-1c-0002.nc", depth=66)
    #agent = Q_Agent(env, policy=episilon_greedy)
    #agent.q_table = load_q_table(r"results\q_tables\q_tables_by_episodes\episilon_greedy\episode_49_reward_trace_area_depth_67_lawn_size_50.pkl")
    #agent.run(max_steps=5000)
    #lawnmover_name = None # Path(r'.\results\gifs\lawnmover.gif')
    #action_name = None # Path(r'.\results\gifs\actions.gif')
    #sprite_path = None
    #print("starting animating")
    ## animate_agent_behavior(agent.lawnmover_actions, r"sim\SMART-AUVs_OF-June-1c-0002.nc", 0, 66, gif_name=lawnmover_name, sprite_path=sprite_path)
    #animate_agent_behavior(agent.actions_performed, r"sim\SMART-AUVs_OF-June-1c-0002.nc", 0, 66, gif_name=action_name, sprite_path=sprite_path)
