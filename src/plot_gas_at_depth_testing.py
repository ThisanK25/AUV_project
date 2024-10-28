# %%
import matplotlib.pyplot as plt
# import path_utils ## lage selv for å rotere punkter
from utils import chem_utils as cu

plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# File path to the chemical data NetCDF file
chemical_file_path = "../SMART-AUVs_OF-June-1c-0002.nc"
chemical_dataset = cu.load_chemical_dataset(chemical_file_path)

# Define target coordinates and parameters for volume extraction
x_target = 100
y_target = 100
z_target = 69
time_target = 7
radius = 4
metadata = x_target, y_target, z_target, time_target, radius
# data_parameter = 'pCO2'
data_parameter = 'pH'

# Extract chemical data and compute average value within the specified volume
chemical_volume_data_mean, data_within_radius = cu.extract_chemical_data_for_volume(
    chemical_dataset, metadata, data_parameter
)

# %%
# Read and plot depths from dataset


for depth in range(55, 70):
   val_dataset = chemical_dataset[data_parameter].isel(time=time_target, siglay=depth)
   val = val_dataset.values[:72710]
   x = val_dataset['x'].values[:72710]
   y = val_dataset['y'].values[:72710]
   x = x - x.min()
   y = y - y.min()
   fig, ax = plt.subplots(figsize=(8, 6))
   scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2)
   cbar = fig.colorbar(scatter, ax=ax)
   cbar.set_label('Value')

   # Add labels and title
   ax.set_xlabel('Easting [m]')
   ax.set_ylabel('Northing [m]')
   ax.set_title(f'{data_parameter} at {depth}m depth')

   plt.show()

# %%

## Testing: Using lawnmower pattern in dataset

depth = 66



# Print the results
print(
    f"Average {data_parameter} value within spherical radius of {radius} "
    f"around the point ({x_target}, {y_target}, {z_target}) is: {chemical_volume_data_mean:.4f}."
)
print(f"Number of points used in the average: {len(data_within_radius)}.")



# %%
