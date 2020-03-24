import id31_tools as id31
import id31_detector_image_simulation_backend as dis
import constants as const

#####################################################################################
# Define parameters for simulation
#####################################################################################

E_keV = 40
theta = 0
alpha = 0
d = 745

#####################################################################################

k_0 = id31.get_K_0(E_keV)

engine = dis.mayavi_init()
qx_range, qy_range = dis.plot_detector(engine, k_0, theta, d)

# Plot Co rods
dis.plot_rods(qx_range, qy_range, const.a_star_Co, const.c_star_Co, k_0, dis.selection_rules_hcp, alpha=0, rod_length=2, color=(0,0.5,0))

# Plot Au rods
dis.plot_rods(qx_range, qy_range, const.a_star_Au, const.c_star_Au, k_0, dis.selection_rules_fcc, alpha=0, rod_length=3, color=(1,1,0))


dis.plot_origin()
dis.show()

