"""
Some common fcns
"""
#######################################################################
import numpy as num

#######################################################################

atomic_symbols = [None,
   'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne', 'Na', 'Mg', 'Al',
   'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe',
   'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
   'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
   'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
   'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt',
   'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
   'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']

def cosd(x):
    """
    num.cos(x), x in degrees
    """
    return num.cos(num.radians(x))

def sind(x):
    """
    num.sin(x), x in degrees
    """
    return num.sin(num.radians(x))

def tand(x):
    """
    num.tan(x), x in degrees
    """
    return num.tan(num.radians(x))

def arccosd(x):
    """
    num.arccos(x), result returned in degrees
    """
    return num.degrees(num.arccos(x))

def arcsind(x):
    """
    num.arcsin(x), result returned in degrees
    """
    return num.degrees(num.arcsin(x))

def arctand(x):
    """
    num.arctan(x), result returned in degrees
    """
    return num.degrees(num.arctan(x))

def cartesian_mag(v):
    """
    Calculate the norm of a vector defined in
    a cartesian basis.
    
    This should give same as num.linalg.norm
    """
    m = num.sqrt(num.dot(v,v))
    return m

def cartesian_angle(u,v):
    """
    Calculate angle between two vectors defined in
    a cartesian basis.

    Result is always between 0 and 180 degrees
    """
    uv = num.dot(u,v)
    um = cartesian_mag(u)
    vm = cartesian_mag(v)
    denom = (um*vm)
    if denom == 0: return 0.
    arg = uv/denom
    if num.fabs(arg) > 1.0:
        arg = arg / num.fabs(arg)
    alpha = arccosd(arg)
    return alpha

