import numpy as np

def calculate_magnetic_field(Xd, Yd, Zd, phi, theta, strength, Xv, Yv, Zv, Xc, Yc, Zc, Nx, Ny, Nz):
    # Constants
    mu_0 = 4 * np.pi * 1e-7
    epsilon = 1e-10 
    
    # Define dipole moment vector
    #m = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    
    # Normalize d to ensure unit length
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    d_normalized = np.array([dx, dy, dz]) / magnitude
    
    # Calculate magnetic moment m
    m = strength * d_normalized
    
    # Initialize magnetic field tensor
    B = np.zeros((Nx, Ny, Nz, 3))
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Xi = Xv - Xc/2 + i * Xc/Nx
                Yj = Yv - Yc/2 + j * Yc/Ny
                Zk = Zv - Zc/2 + k * Zc/Nz
                
                R = np.array([Xi - Xd, Yj - Yd, Zk - Zd])
                R_mag = np.linalg.norm(R)
                
                A = (mu_0 / (4*np.pi)) * (np.cross(m, R) / (R_mag**3 + epsilon) )
                #A = (9.8696) * (np.cross(m, R) / R_mag**3)
                
                B[i,j,k] = np.gradient(A)
    
    # Calculate magnitude of B and its gradient
    B_magnitude = np.linalg.norm(B, axis=-1)
    gradient_B_magnitude = np.gradient(B_magnitude)
    
    # Prepare output dataset
    output = np.column_stack((
        np.repeat(np.arange(Nx), Ny*Nz),
        np.tile(np.repeat(np.arange(Ny), Nz), Nx),
        np.tile(np.arange(Nz), Nx*Ny),
        np.repeat(np.linspace(Xv-Xc/2, Xv+Xc/2, Nx), Ny*Nz),
        np.tile(np.repeat(np.linspace(Yv-Yc/2, Yv+Yc/2, Ny), Nz), Nx),
        np.tile(np.linspace(Zv-Zc/2, Zv+Zc/2, Nz), Nx*Ny),
        B[...,0].flatten(),
        B[...,1].flatten(),
        B[...,2].flatten(),
        B_magnitude.flatten(),
        gradient_B_magnitude[0].flatten(),
        gradient_B_magnitude[1].flatten(),
        gradient_B_magnitude[2].flatten()
    ))
    
    return output

# Example usage
Xd, Yd, Zd = 0, 0, 0
phi = np.pi/4
theta = np.pi/6
Xv, Yv, Zv = 0, 0, 0
Xc, Yc, Zc = 10.0, 10.0, 10.0
Nx, Ny, Nz = 10, 10 ,10

#Xd, Yd, Zd = [float(x) for x in input("Enter dipole location (Xd, Yd, Zd): ").split()]
#phi = float(input("Enter dipole orientation angle phi (relative to X-axis in XY-plane): "))
#theta = float(input("Enter dipole orientation angle theta (relative to Z-axis): "))
strength = float(input("Enter dipole strength (default 1.0): ") or 1.0)
#Nx, Ny, Nz = [int(x) for x in input("Enter (Nx,Ny,Nz): ").split()]


result = calculate_magnetic_field(Xd,Yd,Zd ,phi ,theta,strength, Xv ,Yv ,Zv ,Xc ,Yc ,Zc ,Nx ,Ny ,Nz)

# Save result to a .csv file
np.savetxt('magnetic_field_result.csv', result,
           delimiter=',',
           header='i,j,k,Xi,Yj,Zk,Bx,i,Bx,j,Bx,k,Bmagnitude,' +
                  'Gradientx,Bgradienty,Bgradientz')