import numpy as np

def make_cif_file_from_batch_surface_file(surface_file="/Users/cqiu/app/SuPerRod/batchfile/full_layer2.str",a=5.038,b=5.434,c=7.3707,alpha=90,beta=90,gamma=90,z_max=2.3):
    with open(surface_file,"r") as f1:
        with open(surface_file.replace(".str",".cif"),"w") as f2:
            f2.write('data_global\n')
            f2.write("_chemical_name_mineral 'Hematite'\n")
            f2.write("_chemical_formula_sum 'Fe2 O3'\n")
            f2.write("_cell_length_a {0}\n".format(a))
            f2.write("_cell_length_b {0}\n".format(b))
            f2.write("_cell_length_c {0}\n".format(z_max*c))
            f2.write("_cell_angle_alpha {0}\n".format(alpha))
            f2.write("_cell_angle_beta {0}\n".format(beta))
            f2.write("_cell_angle_gamma {0}\n".format(gamma))
            f2.write("_cell_volume 201.784\n")
            f2.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n  'x,y,z'\nloop_\n")
            f2.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
            for line in f1.readlines():
                if line[0]!="#":
                    items=line.rsplit(",")
                    s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (items[1],float(items[2]),float(items[3]),float(items[4])/z_max)
                    f2.write(s)
