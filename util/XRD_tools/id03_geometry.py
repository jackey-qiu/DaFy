import id03_tools_old as id03
from mayavi import mlab
from tvtk.tools import visual
import constants as const
import mayavi_functions as mfunc
import numpy as np

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene(size=(600, 800))
scene = engine.scenes[0]
fig = mlab.gcf(engine)
mlab.figure(figure=fig, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), engine=engine)
visual.set_viewer(fig) 


spec_filename = '/home/finn/data/2016_06_MA2858/ma2858_sixcvertical.spec'
edf_path =  '/home/finn/data/2016_06_MA2858/ma2858_img/'

DI = id03.DetectorImg(22.5, (128, 385), (0.055, 0.055), 914, spec_filename, edf_path)

scan_no = 1055
frame_no = 20



intensity, HKL, q = DI.get_HKL_q(scan_no, frame_no, norm_mon=True, norm_transm=True)
qx, qy, qz = q
H, K, L = HKL

theta = -69.5

from PyMca5.PyMcaPhysics import SixCircle 
d = SixCircle.SixCircle()
d.setEnergy(DI.E_keV)
d.setUB(DI.UB)
print d.getQLab(mu=0.34, delta=10, gamma=0.0)


QX, QY, QZ = id03.get_q_from_HKL(H, K, L, theta, const.a_star_Au, const.c_star_Au)



# Plot axes
qxH, qyH, qzH = id03.get_q_from_HKL(1, 0, 0, theta, const.a_star_Au, const.c_star_Au)
qxK, qyK, qzK = id03.get_q_from_HKL(0, 1, 0, theta, const.a_star_Au, const.c_star_Au)
qxL, qyL, qzL = id03.get_q_from_HKL(0, 0, 3, theta, const.a_star_Au, const.c_star_Au)
mfunc.Arrow(0, 0, 0, qxL, qyL, qzL, (0,0,0))
mfunc.Arrow(0, 0, 0, qxH, qyH, qzH, (0,0,0))
mfunc.Arrow(0, 0, 0, qxK, qyK, qzK, (0,0,0))

mlab.mesh(qx, qy, qz, scalars=intensity, vmax=0.05)
mlab.mesh(QX, QY, QZ, scalars=intensity, vmax=0.05)



mlab.show()





