# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.stats import mode
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, square

# Enable interactive mode
plt.ion()

# Load your data
file_path = r'C:\Users\nanzh\OneDrive - Swinburne University\Desktop\Project with Saulius\Sapphire Si\Experiments\2023.11.1 sapphire ablation - 3D profiler\20.asc'
data = np.loadtxt(file_path, dtype=np.float64)

# Smooth the data
smoothed_data = gaussian_filter(data, sigma=0)  # No smoothing applied, as sigma=0

# Identify the reference surface
mode_data, _ = mode(smoothed_data, axis=0, keepdims=True)
mode_surface = np.tile(mode_data, (data.shape[0], 1))

# Adjust the threshold to include smaller standard deviation values for grooves
smaller_std_dev_threshold = smoothed_data.mean()  # No deviation considered, as the multiplier is 0
grooves_mask_smaller_std_dev = smoothed_data < smaller_std_dev_threshold

# Prepare mesh grid for 3D plotting
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# Create a figure for plotting
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with data, using the groove mask to apply different colors
ax.plot_surface(X, Y, data, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.5)

# Highlight groove areas in a different color
groove_points = np.argwhere(grooves_mask_smaller_std_dev)
for point in groove_points:
    i, j = point
    ax.scatter(X[i, j], Y[i, j], data[i, j], color='red', s=10)  # s is the size of points

ax.set_title('3D Surface Plot with Highlighted Groove Areas')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis (Height)')

# Display the plot interactively
plt.show()


# Set the threshold to the mean of the data to identify grooves
threshold = data.mean()
grooves_mask = data < threshold

# Prepare meshgrid for 3D plotting
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# Create a figure for plotting
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with data
surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none', alpha=0.7)

# Highlight groove areas
groove_points = np.argwhere(grooves_mask)
for point in groove_points:
    i, j = point
    ax.scatter(X[i, j], Y[i, j], data[i, j], color='red', s=10)  # s is the size of points

ax.set_title('3D Surface Plot with Highlighted Groove Areas')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis (Height)')
plt.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to indicate height values

plt.show()

from skimage.measure import label, regionprops

# Labeling connected components in the groove mask
labeled_grooves, num_grooves = label(grooves_mask, return_num=True, connectivity=2)
props = regionprops(labeled_grooves)

# Initialize lists to store measurements
groove_widths = []
groove_depths = []
groove_volumes = []

# Iterate over each labeled groove to calculate measurements
for prop in props:
    # Width can be approximated by the number of columns spanned by the bounding box of the groove
    min_row, min_col, max_row, max_col = prop.bbox
    width = max_col - min_col
    groove_widths.append(width)
    
    # Depth is the average value of the data points within the groove region, subtracted from the threshold
    groove_mask = labeled_grooves == prop.label
    depth = threshold - data[groove_mask].mean()
    groove_depths.append(depth)
    
    # Volume is approximated by summing the depth of all points within the groove
    volume = depth * prop.area  # prop.area gives the number of pixels in the groove
    groove_volumes.append(volume)

# Presenting the results
for i, (width, depth, volume) in enumerate(zip(groove_widths, groove_depths, groove_volumes), start=1):
    print(f"Groove {i}: Width = {width} pixels, Average Depth = {depth:.2f} units, Volume = {volume:.2f} units^3")
