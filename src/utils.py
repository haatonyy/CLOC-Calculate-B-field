import numpy as np
import scipy as sp

def normalize(v):
    return v/np.linalg.norm(v)

def cross_prod_x(a, b):
    return a[1]*b[2] - a[2]*b[1]

def cross_prod_y(a, b):
    return a[2]*b[0] - a[0]*b[2]

def cross_prod_z(a, b):
    return a[0]*b[1] - a[1]*b[0]

def coil_basis(n):
    ez = normalize(n)
    if ez[1] == 0 and ez[2] == 0:
        ey = np.array([0, 0, 1])
    elif ez[0] == 0 and ez[2] == 0:
        ey = np.array([1,0,0])
    else:    
        ey = np.array([0, n[2], -n[1]])
    ex = np.cross(ey, ez)
    return ex, ey, ez

#to-do: test