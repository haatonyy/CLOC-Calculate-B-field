import numpy as np
import scipy as sp

mu0_mod = 1e-7 #This is mu0/(4*pi) in SI units
pi = np.pi

#to-do: implement coil orientation

class Coil:
    def __init__(self, I, R, N, position, orient=np.array((0,0,1))):
        self.I = I          # Current in Amperes
        self.R = R          # Radius in meters
        self.N = N          # Number of turns
        self.pos = position  # Center position of the coil (x, y, z)
        self.orient = orient/np.linalg.norm(orient)  # Orientation of the coil (should be a unit vector)
        #self.eff_z = np.dot(self.orient, np.array((0,0,1)))  # Effective z-component of the orientation


    def dl_by_dphi(self, phi):
        """
        Differential length element of the coil divided by dphi in cylindrical coordinates
        
        dl/dphi = (-R*sin(phi), R*cos(phi), 0)
        """
        R = self.R
        return np.array((R*-np.sin(phi), R*np.cos(phi), 0)) 

    def r_prime(self, phi, position):
        """
        Position vector from coil element (whose position is characterized by phi) to observation point:

        r' = r - r_coil (in vector form)
        """

        R = self.R
        z_coil = self.pos[2]
        return position-np.array((R*np.cos(phi), R*np.sin(phi), z_coil))

    def BS_integrand(self, phi, position):
        """
        Integrand for the Biot-Savart law:
        dB = [(mu0*I) * (dl/dphi x r') / |r'|^3
        
        """
        #get r_prime
        rp = self.r_prime(phi, position)
        rp_mag = np.linalg.norm(rp)
        #get dl/dphi
        dl_over_dphi = self.dl_by_dphi(phi)
        return np.cross(dl_over_dphi, rp) / (rp_mag**3)


    def Bx(self, positions):
        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        Bx_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos)[0], 0, 2*pi)
            Bx = (mu0_mod*I*N) * integral
            Bx_arrays = np.append(Bx_arrays, Bx)
        return Bx_arrays

    def By(self, positions):
        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        By_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos)[1], 0, 2*pi)
            By = (mu0_mod*I*N) * integral
            By_arrays = np.append(By_arrays, By)
        return By_arrays

    def Bz(self, positions):
        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        Bz_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos)[2], 0, 2*pi)
            Bz = (mu0_mod*I*N) * integral
            Bz_arrays = np.append(Bz_arrays, Bz)
        return Bz_arrays