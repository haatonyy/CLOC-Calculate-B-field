import numpy as np
import scipy as sp
import src.utils as utils

mu0_mod = 1e-7 #This is mu0/(4*pi) in SI units
pi = np.pi

#to-do: implement coil orientation

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

        #self.eff_z = np.dot(self.orient, np.array((0,0,1)))  # Effective z-component of the orientation

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

    def BS_integrand_x(self, phi, position):
        """
        Integrand for the Biot-Savart law:
        dB = [(mu0*I) * (dl/dphi x r') / |r'|^3
        
        """
        #get r_prime
        rp = self.r_prime(phi, position)
        rp_mag = np.linalg.norm(rp)
        #get dl/dphi
        dl_over_dphi = self.dl_by_dphi(phi)
        return utils.cross_prod_x(dl_over_dphi, rp) / (rp_mag**3)
    
    def BS_integrand_y(self, phi, position):
        """
        Integrand for the Biot-Savart law:
        dB = [(mu0*I) * (dl/dphi x r') / |r'|^3
        
        """
        #get r_prime
        rp = self.r_prime(phi, position)
        rp_mag = np.linalg.norm(rp)
        #get dl/dphi
        dl_over_dphi = self.dl_by_dphi(phi)
        return utils.cross_prod_y(dl_over_dphi, rp) / (rp_mag**3)
    
    def BS_integrand_z(self, phi, position):
        """
        Integrand for the Biot-Savart law:
        dB = [(mu0*I) * (dl/dphi x r') / |r'|^3
        
        """
        #get r_prime
        rp = self.r_prime(phi, position)
        rp_mag = np.linalg.norm(rp)
        #get dl/dphi
        dl_over_dphi = self.dl_by_dphi(phi)
        return utils.cross_prod_z(dl_over_dphi, rp) / (rp_mag**3)


    def Bx(self, positions):
        I = self.I
        N = self.N
        if np.isscalar(positions[0]):
            positions = np.array([positions])

        Bx_arrays = np.array([])
        
        for pos in positions:
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand_x(phi, pos), 0, 2*pi)
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
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand_y(phi, pos), 0, 2*pi)
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
            integral, _ = sp.integrate.quad(lambda phi: self.BS_integrand_z(phi, pos), 0, 2*pi)
            Bz = (mu0_mod*I*N) * integral
            Bz_arrays = np.append(Bz_arrays, Bz)
        return Bz_arrays