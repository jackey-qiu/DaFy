# Define lattice constants that are often used
import numpy as np 

class lattice_parameter:
    def __init__(self, a, c, a_star, c_star):
        self.a = a
        self.c = c
        self.a_star = a_star
        self.c_star = c_star

# Materials with fcc in hexagonal lattice
#####################################################
# Au
a_Au = 2.884
c_Au = 7.065
a_star_Au = 4.*np.pi/a_Au/np.sqrt(3.)
c_star_Au = 2.*np.pi/c_Au
Au = lattice_parameter(a_Au, c_Au, a_star_Au, c_star_Au)

# Pd
a_Pd = 2.751
c_Pd = 6.739
a_star_Pd = 4.*np.pi/a_Pd/np.sqrt(3)
c_star_Pd = 2.*np.pi/c_Pd
Pd = lattice_parameter(a_Pd, c_Pd, a_star_Pd, c_star_Pd)

# PdH06
a_PdH06 = 2.850
c_PdH06 = 6.980
a_star_PdH06 = 4.*np.pi/a_PdH06/np.sqrt(3)
c_star_PdH06 = 2.*np.pi/c_PdH06
PdH06 = lattice_parameter(a_PdH06, c_PdH06, a_star_PdH06, c_star_PdH06)

# Pt
a_Pt = 2.775
c_Pt = 6.797
a_star_Pt = 4.*np.pi/a_Pt/np.sqrt(3)
c_star_Pt = 2.*np.pi/c_Pt
Pt = lattice_parameter(a_Pt, c_Pt, a_star_Pt, c_star_Pt)

# Materials with hcp
#####################################################
# Co
a_Co = 2.507
c_Co = 4.069
a_star_Co = 4.*np.pi/a_Co/np.sqrt(3)
c_star_Co = 2.*np.pi/c_Co
Co = lattice_parameter(a_Co, c_Co, a_star_Co, c_star_Co)

# Materials with diamond structure
#####################################################
# Si
a_Si = 3.840
c_Si = 9.406
a_star_Si = 4.*np.pi/a_Si/np.sqrt(3)
c_star_Si = 2.*np.pi/c_Si
Si = lattice_parameter(a_Si, c_Si, a_star_Si, c_star_Si)

print c_star_Au