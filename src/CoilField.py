class CoilField():
    def __init__(self):
        self.coils = []


    def add_coil(self, coil):
        self.coils.append(coil)
    
    def Bx(self, positions):
        Bx_total = 0
        for coil in self.coils:
            Bx_total += coil.Bx(positions)
        return Bx_total
    
    def By(self, positions):
        By_total = 0
        for coil in self.coils:
            By_total += coil.By(positions)
        return By_total
    
    def Bz(self, positions):
        Bz_total = 0
        for coil in self.coils:
            Bz_total += coil.Bz(positions)
        return Bz_total
    
