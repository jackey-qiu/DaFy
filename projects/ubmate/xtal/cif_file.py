"""
Simple cif file module

Authors/Modifications:
----------------------
* Tom Trainor (tptrainor@alaska.edu) 

Todo:
-----
* Cif files may have multiple 'data' blocks.  Currently the CifFile class
  assumes only one data block and parses all data into one dictionary
  - we should allow for multiple data blocks...

* Add some errors/exceptions if cant find the file or unable to read it...
  (ie should fail niceley)

"""
##########################################################################
import os, copy, re
import numpy as num
from xtal import space_grps

atsym_parse = re.compile(r'[A-Z][a-z]?')
##########################################################################
def read(fname):
    """
    Read xtal data from a cif file

    Parameters:
    ----------
    * fname : name of the cif file (including path)

    Returns:
    --------
    * cell:   [a,b,c,alpha,beta,gamma]
    * labels: [site_lbl_0, site_lbl_1 ...]    
    * atsym:  [atmic_symbol_0, atomic_symbol_1 ...]  or  None
    * coords: num.array([[x0,y0,z0],[x1,y1,z1]...])    (fractional coords)
    * occ:    num.array([occ0,occ1...])  (default to ones) 
    * ox:     num.array([ox0,ox1...])   (default to all 99)
    * Uiso:   num.array([U0, U1...])    (default to zeros)
    * Uaniso: num.array([[U0_11, U0_12, U0_13, U0_22, U0_23, U0_33], ...])  (default to zeros)
    * symops: ['sym_str1','sym_str2',...]  or  None
    
    Notes:
    -----
    * If anisotropic displacement parameters are included, they must 
      have the same labels (in the same order) as the atom site labels
    """
    cf = CifFile(fname)
    # unit cell parameters
    a = _get_float(cf['_cell_length_a'])
    b = _get_float(cf['_cell_length_b'])
    c = _get_float(cf['_cell_length_c'])
    alpha = _get_float(cf['_cell_angle_alpha'])
    beta  = _get_float(cf['_cell_angle_beta'])
    gamma = _get_float(cf['_cell_angle_gamma'])
    cell  = [a,b,c,alpha,beta,gamma]
    
    # site labels and fractional coords
    labels = copy.copy(cf['_atom_site_label'])
    x = cf['_atom_site_fract_x']
    y = cf['_atom_site_fract_y']
    z = cf['_atom_site_fract_z']
    nsite = len(labels)
    coords = num.zeros((nsite,3))
    for j in range(nsite):
        coords[j,0] = _get_float(x[j])
        coords[j,1] = _get_float(y[j])
        coords[j,2] = _get_float(z[j])
    
    # atomic symbols
    atsym = cf['_atom_site_type_symbol']
    if atsym is None: atsym = copy.copy(cf['_atom_site_label'])
    for j in range(nsite):
        tmp = atsym_parse.findall(atsym[j])
        if len(tmp) == 0: raise ValueError
        atsym[j] = tmp[0]
    
    # site occupancy 
    occ = cf['_atom_site_occupancy']
    if occ is None:
        occ = num.ones(nsite)
    else:
        for j in range(nsite):
            occ[j] = _get_float(occ[j])
        occ = num.array(occ)
        
    # oxidation states
    ox = cf['_atom_type_oxidation_number']
    if ox is None:
        ox = num.ones(nsite)
        ox = ox*99
    else:
        for j in range(nsite):
            ox[j] = _get_float(ox[j])
    
    # Uiso (Biso)
    Uiso = cf['_atom_site_U_iso_or_equiv']
    if Uiso is None:
        Uiso = cf['_atom_site_B_iso_or_equiv']
        if Uiso is not None:
            for j in range(nsite):
                Uiso[j] = _get_float(Uiso[j]) / (8*num.pi**2)
    else:
        for j in range(nsite):
            Uiso[j] = _get_float(Uiso[j]) 
    if Uiso is None:
        Uiso = num.zeros(nsite)
    else:
        Uiso = num.array(Uiso)

    # Uaniso
    #Uaniso = None
    Uaniso = num.zeros((nsite,6))
    aniso_label = cf['_atom_site_aniso_label']
    if aniso_label is not None:
        # cross check and make sure same order as labels
        if len(aniso_label) != nsite:
            raise ValueError("Error number of anisotropic labels doesnt match atom site labels")
        for j in range(nsite):
            if aniso_label[j] != labels[j]:
                raise ValueError("Error: anisotropic labels dont match atom site labels")
        convert = False
        U_11 = cf['_atom_site_aniso_U_11']
        U_12 = cf['_atom_site_aniso_U_12']
        U_13 = cf['_atom_site_aniso_U_13']
        U_22 = cf['_atom_site_aniso_U_22']
        U_23 = cf['_atom_site_aniso_U_23']
        U_33 = cf['_atom_site_aniso_U_33']
        if U_11 is None:
            convert = True
            U_11 = cf['_atom_site_aniso_B_11']
            U_12 = cf['_atom_site_aniso_B_12']
            U_13 = cf['_atom_site_aniso_B_13']
            U_22 = cf['_atom_site_aniso_B_22']
            U_23 = cf['_atom_site_aniso_B_23']
            U_33 = cf['_atom_site_aniso_B_33']
        for j in range(nsite):
            Uaniso[j,0] = _get_float(U_11[j])
            Uaniso[j,1] = _get_float(U_12[j])
            Uaniso[j,2] = _get_float(U_13[j])
            Uaniso[j,3] = _get_float(U_22[j])
            Uaniso[j,4] = _get_float(U_23[j])
            Uaniso[j,5] = _get_float(U_33[j])
        if convert: Uaniso = Uaniso / (8*num.pi**2)
        
    # symmetry operations
    symops = cf['_space_group_symop_operation_xyz']
    if symops is None:
        symops = cf['_symmetry_equiv_pos_as_xyz']
    # if no symops try to get from sg
    if symops is None:
        # space group (HM)
        sg = cf['_symmetry_space_group_name_H-M']
        if sg is not None:
            symops = space_grps.get_symops(sg)

    return cell,labels,atsym,coords,symops,occ,ox,Uiso,Uaniso

def _get_float(val):
    """
    convert a string to a float
    deal with errors -> 5.235(4)
    """
    try: val = float(val)
    except: val, err = _get_float_err(val)
    return val

def _get_float_err(val):
    """
    convert number 5.23(4) to 5.23, 0.04
    """
    i = val.find('(')
    x = val[:i]
    e = val[i+1]
    ee = "".join(['0' if z is not '.' else '.' for z in x[:-1]])
    ee = ee + e
    #print(val, x, ee)
    return float(x), float(ee)

def write(fname,labels,lattice,coords,atsym=None,symop=None,occ=None,ox=None,Uiso=None,Uaniso=None):
    """
    Write xtal data to a cif file

    Parameters:
    --------
    * fname: file name
    * labels: [site_lbl0, site_lbl1 ...]    
    * lattice: lattice instance with a,b,c,alpha,beta,gamma
    * coords: [[x0,y0,z0],[x1,y1,z1]...]    
    * atsym: [atsym0, atsym1 ...]  or None
    * symop: ['sym_str1','sym_str2',...]  or None
    * occ: [occ0, occ1, ...]  or None
    * ox: [ox0, ox1, ....]  or None
    * Uiso: [Uiso0, Uiso1, ...] or None
    * Uaniso: [[U0_11, U0_12, U0_12, U0_13, U0_22, U0_23, U0_33], ...])  or None
    """
    f = open(fname,'w')
    f.write("data_global\n")
    f.write("_cell_length_a  %4.5f\n" % lattice.a)
    f.write("_cell_length_b  %4.5f\n" % lattice.b)
    f.write("_cell_length_c  %4.5f\n" % lattice.c)
    f.write("_cell_angle_alpha %4.5f\n" % lattice.alpha)
    f.write("_cell_angle_beta  %4.5f\n" % lattice.beta)
    f.write("_cell_angle_gamma %4.5f\n" % lattice.gamma)
    if symop is not None:
        f.write("loop_\n_space_group_symop_operation_xyz\n")
        for sym in symop:
            f.write("  '%s'\n" % sym)
    f.write("loop_\n_atom_site_label\n")
    if atsym is not None: f.write("_atom_site_type_symbol\n")
    f.write("_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
    if occ is not None: f.write("_atom_site_occupancy\n")
    if ox is not None: f.write("_atom_type_oxidation_number\n")
    if Uiso is not None: f.write("_atom_site_U_iso_or_equiv\n")
    for j in range(len(labels)):
        if atsym is not None: s = "%s  %s"  %  (labels[j], atsym[j])
        else: s = "%s"  %  (labels[j])
        f.write("%s  %8.5f  %8.5f  %8.5f"  %  (s, coords[j][0], coords[j][1], coords[j][2])) 
        if occ is not None: f.write("  %3.3f" % occ[j])
        if ox  is not None: f.write("  %3.1f" % ox[j])
        if Uiso is not None: f.write("  %8.5f" % Uiso[j])
        f.write("\n")
    if Uaniso is not None:
        f.write("loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_12")
        f.write("\n_atom_site_aniso_U_13\n_atom_site_aniso_U_22\n_atom_site_aniso_U_23\n_atom_site_aniso_U_33\n")
        for j in range(len(labels)):
            f.write("%s  %8.5f  %8.5f  %8.5f"  %  (labels[j], Uaniso[j][0], Uaniso[j][1], Uaniso[j][2])) 
            f.write("  %8.5f  %8.5f  %8.5f\n"  %  (Uaniso[j][3], Uaniso[j][4], Uaniso[j][5])) 
    f.write("\n")

##########################################################################
class CifFile:
    """
    Parse crystal structure information from a cif file

    Notes:
    ------
    All the data is held in the dictionary self.data
    The dictionary keywords are the cif labels i.e. '_cell_length_a' etc
    For _loop data the resulting value is a list.  e.g. '_atom_site_fract_x'
    Data can be accessed by direcly indexing the object.  

    Example:
    --------
    >>cf = CifFile('quarttz.cif')
    >>a = cf['_cell_length_a']      # return (the string) value of a
    >>x = cf['_atom_site_fract_x']  # returns a list of (the string) x-values of sites
    """
    def __init__(self,fname=None):
        """
        Initialize
        """
        self.lines = []   # all lines in the cif file
        self.data  = {}   # dictionary of data
        if fname is not None:
            self.read(fname)

    def __repr__(self,):
        lout = ""
        for j in range(len(self.lines)): 
            lout = lout + "%i : %s\n" % (j,self.lines[j])
        return lout

    def __getitem__(self,index):
        return self.data.get(index)

    def read(self, fname):
        """
        Read a cif file and extract structural data
        """
        self.lines = []
        # read file
        if os.path.exists(fname) and os.path.isfile(fname):
            f = open(fname)
            tmp = f.readlines()
            f.close()
        else:
            print('File error: cannot find file name %s ' % fname)
            return

        # strip lines, remove quotes and blanks etc.
        lines = []
        for ll in tmp:
            ll = ll.replace("'","")
            ll = ll.replace('"',"")
            ll = ll.strip()
            if ll == "": 
                pass
            elif ll[0] not in ("#",";"):
                lines.append(ll)
        if len(lines) == 0: return
        else: self.lines = lines

        # parse lines into a dictionary
        self._parse_data()

    def _parse_data(self,):
        """
        parse all the cif fields into a dictionary
        """
        lines = self.lines
        nline = len(lines)
        data  = {}
        j = 0
        while j < nline:
            if lines[j] == "loop_":
                j += 1
                st = j
                rdlbls = True
                while j < nline:
                    if (lines[j][0] == "_") and (rdlbls == True):
                        j += 1
                    elif (lines[j][0] == "_") and (rdlbls == False):
                        en = j
                        break
                    elif lines[j] == "loop_":
                        en = j
                        break
                    else: 
                        j += 1
                        rdlbls = False
                if j == nline: en = j
                data.update(self._parse_loop(lines[st:en]))
            elif lines[j][0] == "_":
                tmp1 = lines[j].split()
                if len(tmp1) > 1:
                    nn = len(tmp1[0])
                    dat = lines[j][nn:].strip()
                    #data.update({tmp1[0]:dat})
                    j += 1
                else:
                    dat = ""
                    j += 1
                    while j < nline:
                        if (lines[j][0] == "_") or (lines[j][0] == "loop_"): 
                            break
                        else:
                            dat = dat + " " + lines[j].strip()
                        j += 1
                data.update({tmp1[0]:dat})
            else:
                j += 1
        self.data = data

    def _parse_loop(self, lines):
        """
        make a dictionary from the loop data
        """
        lbls = []
        data = []
        for ll in lines:
            if ll[0] == "_":
                lbls.append(ll)
            else:
                data.append(ll.split())
        loop_dict = {}
        for lbl in lbls:
            loop_dict.update({lbl:[]})
        for j in range(len(data)):
            for k in range(len(lbls)):
                loop_dict[lbls[k]].append(data[j][k])
        return loop_dict

###############################################################################
###############################################################################
if __name__ == '__main__':
    #print(_get_float('5.234(5)'))
    #print(_get_float_err('5.234(5)'))
    #print(_get_float_err('5(1)'))
    #print(_get_float_err('1.230(9)'))
    fname = 'COD_Fe2O3.cif'
    #fname = 'AMS_Fe2O3.cif'
    cell,labels,atsym,coords,symops,occ,ox,Uiso,Uaniso = read(fname)
