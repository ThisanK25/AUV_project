import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None, reverse=False):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
    
    if reverse:
        rgb_list = rgb_list[::-1]

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def plot_colortable(hex_list):
    '''
    Given a list of HEX strings, display the color as a cell, alongside the HEX and RGB values of the color.
    '''
    cell_width = 150
    cell_height = 50
    swatch_width = 48
    margin = 0
    topmargin = 40

    rgb_list = [hex_to_rgb(value) for value in hex_list]
    dec_list = [rgb_to_dec(value) for value in rgb_list]
    names = [f'HEX: {col[0]}\nRGB: {col[1]}' for col in zip(hex_list, rgb_list, dec_list)]
    n = len(names)
    ncols = 4
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 8 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=dec_list[i], linewidth=18)
    return fig


def example_colormap():
    x, y = np.mgrid[-5:5:0.05, -5:5:0.05]
    z = (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))

    fig, axes = plt.subplots(1,2, figsize=(10,10))

    # Plot using default colormap
    ax = axes.flatten()[0]
    im = ax.imshow(z, aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())   # remove y axis ticks
    ax.xaxis.set_major_locator(plt.NullLocator())   # remove x axis ticks
    ax.set_aspect('equal', adjustable='box')        # make subplots square
    ax.set_title(f'Cmap: Default', fontsize=18)      # add a title to each
    divider = make_axes_locatable(ax)               # make colorbar same size as each subplot
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # Plot using custom colormap
    ax = axes.flatten()[1]
    # hex_list = ['#0091ad', '#3fcdda', '#83f9f8', '#d6f6eb', '#fdf1d2', '#f8eaad', '#faaaae', '#ff57bb']
    hex_list = ['#0A2F51', '#0E4D64', '#137177', '#188977', '#1D9A6C', '#39A96B', '#56B870', '#74C67A', '#99D492', '#BFE1B0', '#DEEDCF']
    float_list=[0, 0.05, 0.1, 0.15, 0.2, 0.5, 0.6, 0.85, 0.9, 0.92, 1]
    im = ax.imshow(z, cmap=get_continuous_cmap(hex_list, float_list=float_list), aspect='auto')
    ax.yaxis.set_major_locator(plt.NullLocator())   # remove y axis ticks
    ax.xaxis.set_major_locator(plt.NullLocator())   # remove x axis ticks
    ax.set_aspect('equal', adjustable='box')        # make subplots square
    ax.set_title(f'Cmap: Custom', fontsize=18)      # add a title to each
    divider = make_axes_locatable(ax)               # make colorbar same size as each subplot
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # save out example
    # plt.savefig("./colorbar_example.png", dpi=150)
    plt.show()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    function taken from
    https://stackoverflow.com/questions/7404116/...
        ...defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {  'red': [],  'green': [], 'blue': [],  'alpha': []  }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(newcmap)  # Updated method to register colormap

    return newcmap


def main():
    example_colormap()

if __name__ == "__main__":
    main()