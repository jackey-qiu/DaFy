from mayavi import mlab
from tvtk.tools import visual
import numpy as np

def Arrow(x1, y1, z1, x2, y2, z2, color=(1,1,1)):
    ar1=visual.arrow(x=x1, y=y1, z=z1)
    arrow_length=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    ar1.length_cone=0.4/arrow_length
    ar1.radius_shaft = 0.03/arrow_length
    ar1.radius_cone = 0.1/arrow_length
    
    ar1.actor.scale=[arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos/arrow_length
    ar1.axis = [x2-x1, y2-y1, z2-z1]
    ar1.color = color
    return ar1