import numpy as np
import scipy as sp

mu0_mod = 1e-7
pi = np.pi

coil_params = {
    'I': 1,        # Current in Amperes
    'R': 1,        # Radius in meters
    'N': 100,      # Number of turns
    'pos': (0, 0, 0) # Center position of the coil
}

def dl_by_dphi(phi, coil_params):
    R = coil_params['R']
    return np.array((R*-np.sin(phi), R*np.cos(phi), 0)) 

def r_prime(phi, position, coil_params):
    R = coil_params['R']
    z_coil = coil_params['pos'][2]
    return position-np.array((R*np.cos(phi), R*np.sin(phi), z_coil))

def integrand(phi, position, coil_params):
    #get r_prime and dl/dphi
    rp = r_prime(phi, position, coil_params)
    rp_mag = np.linalg.norm(rp)
    dl_by_dphi = dl_by_dphi(phi, coil_params)
    return np.cross(dl_by_dphi, rp) / (rp_mag**3)


def Bx(position, coil_params):
    I = coil_params['I']
    N = coil_params['N']
    integral, error = sp.integrate.quad(lambda phi: integrand(phi, position, coil_params)[0], 0, 2*pi)
    return (mu0_mod*I*N) * integral

def By(position, coil_params):
    I = coil_params['I']
    N = coil_params['N']
    integral, error = sp.integrate.quad(lambda phi: integrand(phi, position, coil_params)[1], 0, 2*pi)
    return (mu0_mod*I*N) * integral

def Bz(position, coil_params):
    I = coil_params['I']
    N = coil_params['N']
    integral, error = sp.integrate.quad(lambda phi: integrand(phi, position, coil_params)[2], 0, 2*pi)
    return (mu0_mod*I*N) * integral