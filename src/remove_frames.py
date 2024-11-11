from pathlib import Path
from PIL import Image, ImageSequence

def remove_frames_from_gif(input_path, output_path, frames_to_remove):
    # Open the original GIF
    gif = Image.open(input_path)

    # Get all frames from the original GIF
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    # Remove specified frames
    frames_to_keep = [frame for i, frame in enumerate(frames) if i not in frames_to_remove and i % 2 == 0]

    # Save the result as a new GIF
    frames_to_keep[0].save(output_path, save_all=True, append_images=frames_to_keep[1:], loop=0)

remove_frames_from_gif(Path(r'results\gifs\actions.gif'), Path(r'results\gifs\actions2.gif'), list(range(750)) + list(range(1999, 3481)))