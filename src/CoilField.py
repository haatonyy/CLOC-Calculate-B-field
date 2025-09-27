import numpy as np

class CoilField():
    def __init__(self):
        self.coils = []

    def add_coil(self, coil):
        self.coils.append(coil)
    
    def get_Bx(self, positions):
        Bx_total = 0
        for coil in self.coils:
            Bx_total += coil.get_Bx(positions)
        return Bx_total
    
    def get_By(self, positions):
        By_total = 0
        for coil in self.coils:
            By_total += coil.get_By(positions)
        return By_total
    
    def get_Bz(self, positions):
        Bz_total = 0
        for coil in self.coils:
            Bz_total += coil.get_Bz(positions)
        return Bz_total
    
    def get_B(self, positions):
        Bx_total = self.get_Bx(positions)
        By_total = self.get_By(positions)
        Bz_total = self.get_Bz(positions)
        return np.column_stack((Bx_total, By_total, Bz_total))
