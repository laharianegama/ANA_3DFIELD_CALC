
import numpy as np

def calculate_magnetic_field(Xd, Yd, Zd, phi_deg, theta_deg, strength, Xv, Yv, Zv, Xc, Yc, Zc, Nx, Ny, Nz):
    """
    Calculate the magnetic field at each point in a specified volume due to a point dipole.

    Args:
        Xd (float): X-coordinate of the dipole location.
        Yd (float): Y-coordinate of the dipole location.
        Zd (float): Z-coordinate of the dipole location.
        phi_deg (float): Orientation angle phi in degrees (0-360).
        theta_deg (float): Orientation angle theta in degrees (0-180).
        strength (float): Strength of the dipole.
        Xv (float): X-coordinate of the volume center.
        Yv (float): Y-coordinate of the volume center.
        Zv (float): Z-coordinate of the volume center.
        Xc (float): Size of the volume along the X-axis.
        Yc (float): Size of the volume along the Y-axis.
        Zc (float): Size of the volume along the Z-axis.
        Nx (int): Number of steps along the X-axis.
        Ny (int): Number of steps along the Y-axis.
        Nz (int): Number of steps along the Z-axis.

    Returns:
        np.ndarray: Array containing magnetic field data at each point in the volume.
    """
    
    # Convert angles from degrees to radians
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    
    # Constants
    mu_0 = 4 * np.pi * 1e-7
    epsilon = 1e-10 
    
    # Define dipole moment vector
    dx = np.sin(theta_rad) * np.cos(phi_rad)
    dy = np.sin(theta_rad) * np.sin(phi_rad)
    dz = np.cos(theta_rad)
    
    # Normalize d to ensure unit length
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    d_normalized = np.array([dx, dy, dz]) / magnitude
    
    # Calculate magnetic moment m
    m = strength * d_normalized
    
    # Initialize magnetic field tensor
    B = np.zeros((Nx, Ny, Nz, 3))
    
    # Calculate magnetic field at each point in the volume
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Xi = Xv - Xc/2 + i * Xc/Nx
                Yj = Yv - Yc/2 + j * Yc/Ny
                Zk = Zv - Zc/2 + k * Zc/Nz
                
                R = np.array([Xi - Xd, Yj - Yd, Zk - Zd])
                R_mag = np.linalg.norm(R)
                
                A = (mu_0 / (4*np.pi)) * (np.cross(m, R) / (R_mag**3 + epsilon) )
                B[i,j,k] = np.gradient(A)
    
    # Calculate magnitude of B and its gradient
    B_magnitude = np.linalg.norm(B, axis=-1)
    gradient_B_magnitude = np.gradient(B_magnitude)
    
    return B, B_magnitude, gradient_B_magnitude

def main():
    # Example usage
    Xd, Yd, Zd = 0, 0, 0
    Xv, Yv, Zv = 0, 0, 0
    Xc, Yc, Zc = 10.0, 10.0, 10.0
    Nx, Ny, Nz = 10, 10 ,10

    strength = float(input("Enter dipole strength (default 1.0): ") or 1.0)
    phi = float(input("Enter dipole orientation angle phi in degrees (0-360): "))
    theta = float(input("Enter dipole orientation angle theta in degrees (0-180): "))

    # Calculate magnetic field
    B, B_magnitude, gradient_B_magnitude = calculate_magnetic_field(Xd, Yd, Zd, phi, theta, strength, Xv, Yv, Zv, Xc, Yc, Zc, Nx, Ny, Nz)

    # Flatten the results for saving to CSV
    result = np.column_stack((
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
    
    # Save result to a .csv file
    result_flat = result.reshape(-1, result.shape[-1])
    np.savetxt('magnetic_field_result_modular.csv', result_flat,
               delimiter=',',
               header='i,j,k,Xi,Yj,Zk,Bx,i,Bx,j,Bx,k,Bmagnitude,' +
                      'Gradientx,Bgradienty,Bgradientz')

if __name__ == "__main__":
    main()