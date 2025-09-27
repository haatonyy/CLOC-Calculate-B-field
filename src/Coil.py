import numpy as np
import scipy as sp
import src.utils as utils

mu0_mod = 1e-7 #This is mu0/(4*pi) in SI units
pi = np.pi

class Coil:
    def __init__(self, I, R, N, position, orient=np.array([0,0,1])):
        self.I = I          # Current in Amperes
        self.R = R          # Radius in meters
        self.N = N          # Number of turns
        self.pos = position  # Center position of the coil (x, y, z)
        if isinstance(orient, str) and orient == "z":
            orient = np.array([0,0,1])
        elif isinstance(orient, str) and orient == "y":
            orient = np.array([0,1,0])
        elif isinstance(orient, str) and orient == "x":
            orient = np.array([1,0,0])
        self.orient = utils.normalize(orient)  # Orientation of the coil (should be a unit vector)
        self.ex, self.ey, self.ez = utils.coil_basis(self.orient)    #Basis of the coil

    @classmethod
    def from_dict(cls, params):
        return cls(params['I'], params['R'], params['N'], params['pos'], params.get('orient', np.array([0,0,1])))


    def dl_by_dphi(self, phi):
        """
        Differential length element of the coil divided by dphi in cylindrical coordinates
        
        Note that dl/dphi = R*(-sin(phi), cos(phi), 0) in the coil's local basis (ex, ey, ez)
        dl/dphi = (-R*sin(phi), R*cos(phi), 0)
        """

        return -self.R*np.sin(phi)*self.ex + self.R*np.cos(phi)*self.ey
    
    def r_prime(self, phi, position):
        """
        Position vector from coil element (whose position is characterized by phi) to observation point:

        r_coil = rO + R*cos(phi)*ex + R*sin(phi)*ey
        r' = r - r_coil (in vector form)
        """

        R = self.R
        r_C = self.pos
        r_C_to_phi = R*np.cos(phi)* self.ex + R*np.sin(phi)*self.ey
        r_coil = r_C + r_C_to_phi
        return position-r_coil

    def BS_integrand(self, phi, position, component: str):
        """
        Component of Integrand for the Biot-Savart law:
        dB = [(mu0*I) * (dl/dphi x r') / |r'|^3
        """

        #get r_prime
        rp = self.r_prime(phi, position)
        rp_mag = np.linalg.norm(rp)
        #get dl/dphi
        dl_over_dphi = self.dl_by_dphi(phi)

        if component == 'x':
            return utils.cross_prod_x(dl_over_dphi, rp) / (rp_mag**3)
        elif component == 'y':
            return utils.cross_prod_y(dl_over_dphi, rp) / (rp_mag**3)
        return utils.cross_prod_z(dl_over_dphi, rp) / (rp_mag**3)


    def get_Bx(self, positions):
        """
        Calculate the x-component of the magnetic field at given positions due to this coil.
        """
        
        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        Bx_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos), 0, 2*pi, "x")
            Bx = (mu0_mod*I*N) * integral
            Bx_arrays = np.append(Bx_arrays, Bx)
        return Bx_arrays

    def get_By(self, positions):
        """
        Calculate the y-component of the magnetic field at given positions due to this coil.
        """

        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        By_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos), 0, 2*pi, "y")
            By = (mu0_mod*I*N) * integral
            By_arrays = np.append(By_arrays, By)
        return By_arrays

    def get_Bz(self, positions):
        """
        Calculate the z-component of the magnetic field at given positions due to this coil.
        """

        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        Bz_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand(phi, pos), 0, 2*pi, "z")
            Bz = (mu0_mod*I*N) * integral
            Bz_arrays = np.append(Bz_arrays, Bz)
        return Bz_arrays
    
    def get_B(self, positions):
        """
        Calculate the magnetic field vector at given positions due to this coil.
        """

        Bx = self.get_Bx(positions)
        By = self.get_By(positions)
        Bz = self.get_Bz(positions)
        return_array = np.column_stack((Bx, By, Bz))
        return return_array