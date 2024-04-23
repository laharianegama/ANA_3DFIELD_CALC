import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# Load the data
df = pd.read_csv('magnetic_field_result.csv')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Increase the arrow size by adjusting the length parameter
length = 0.3

# Add color mapping based on the magnitude of the magnetic field
colors = plt.cm.viridis(np.linalg.norm(df[['Bxi', 'Bxj', 'Bxk']].values, axis=1))

# Plot the magnetic field vectors with color mapping
ax.quiver(df['Xi'], df['Yj'], df['Zk'], df['Bxi'], df['Bxj'], df['Bxk'], length=length, normalize=True, color=colors)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a color bar to provide a reference for the colors
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
cbar.set_label('Magnetic Field Magnitude')

# Show the plot
plt.show()






