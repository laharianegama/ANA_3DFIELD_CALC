import matplotlib.pyplot as plt
import numpy as np

# # Extract data from the CSV file
# data = np.genfromtxt('magnetic_field_result.csv', delimiter=',', skip_header=1)

# # Extract coordinates and magnetic field components
# X = data[:, 3]
# Y = data[:, 4]
# Z = data[:, 5]
# Bx = data[:, 6]
# By = data[:, 7]
# Bz = data[:, 8]

# # Plot a 2D slice (e.g., XY plane) of the magnetic field
# plt.quiver(X, Y, Bx, By)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Magnetic Field Distribution (XY Plane)')
# plt.show()


from mayavi import mlab

# # Extract data from the CSV file
# data = np.genfromtxt('magnetic_field_result.csv', delimiter=',', skip_header=1)

# Nx,Ny,Nz=10,10,10

# # Extract coordinates and magnetic field components
# X = data[:, 3].reshape((Nx, Ny, Nz))
# Y = data[:, 4].reshape((Nx, Ny, Nz))
# Z = data[:, 5].reshape((Nx, Ny, Nz))
# Bx = data[:, 6].reshape((Nx, Ny, Nz))
# By = data[:, 7].reshape((Nx, Ny, Nz))
# Bz = data[:, 8].reshape((Nx, Ny, Nz))

# # Create 3D quiver plot of the magnetic field
# fig = mlab.figure()
# mlab.quiver3d(X, Y, Z, Bx, By, Bz, line_width=3, scale_factor=10)
# mlab.xlabel('X')
# mlab.ylabel('Y')
# mlab.zlabel('Z')
# mlab.title('Magnetic Field Distribution')
# mlab.show()




import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Load the data
# df = pd.read_csv('magnetic_field_result.csv')

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the magnetic field vectors
# ax.quiver(df['Xi'], df['Yj'], df['Zk'], df['Bxi'], df['Bxj'], df['Bxk'], length=0.1, normalize=True)

# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()



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






