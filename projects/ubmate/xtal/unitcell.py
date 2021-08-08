"""
Crystal structure generator

Authors/Modifications:
----------------------
* Tom Trainor (tptrainor@alaska.edu) 

Todo:
-----
* UnitCell
  - method to get all sym_ops from space groupd name...
    (see space_grps.get_symops)

* PositionGenerator
  - Test that the add_sym_op parses all possible operations...


Notes on symmetry operations:
----------------------------
* The UnitCell class uses the PositionGenerator class to make symmetry
  copies of the assymetric unit for generation of space filling list of 
  atomic coordinates (ie P1 cell).

* The PositionGenerator makes symmetry copies according to:

    |x_cp|       |x|
    |y_cp| = R * |y|  +  t     - or -      v_cp  =  R*v + t
    |z_cp|       |z|

  where R is the rotation part and t is the translation part of the operation.

* This can be put in compact notation using Augmented or Sietz matricies:
    
    v_cp  = W * v

  where W is the Seitz matrix:

       |r11, r12, r13,  tx|
  W =  |r21, r22, r23,  ty|
       |r31, r32, r33,  tz|
       |  0,   0,   0,   1|

  and we add a fourth (dummy) index to v and v_cp to make the dimensions work.
  Note that the elements of W must be rational since applying a symmetry operation 
  to a lattice vector must generate another lattice vector.
  
* If we transform the unit cell lattice (ie define a new set of basis vectors)
  we can also transform the symmetry operators according to the following.  
  In the lattice module, a basis transform is defined that results in a 
  a matrix M that is used to compute the indicies of a (stationary) vector in the
  primed basis in terms of the original basis as:

    |x'|      |x|
    |y'| = M* |y|        - or -    v'  = M*v
    |z'|      |z|

  Therefore, for the rotational part of the symmetry operator:
    
    v'_cp  = M*v_cp
           = M*R*v
           = M*R*(M_inv*M)*v
           = M*R*M_inv(M*v)
           = M*R*M_inv * v'
           = R'*v'
    R'  = M*R*M_inv

  where M_inv is the N matrix in the lattice module.

* We can also directly apply the augmented transform matricies (P and Q) 
  to the Seitz matricies, therefore including the transformation of the 
  shift vectors (see the lattice module):

    W'  = Q*W*P

References:
-----------
* International Tables for Crystallography, Vol A, chap 5.

"""
##########################################################################
import numpy as num
import copy

from xtal._common import atomic_symbols
from xtal import lattice
from xtal import cif_file
from xtal.atom_list import AtomList, _reduce_frac, _expand_frac

##########################################################################
def read_cif(fname):
    """
    Read a cif file and return a UnitCell instance

    Arguments:
    ----------
    * fname: cif file name

    Returns:
    -------
    * UnitCell instance
    """
    cell,labels,atsyms,coords,symops,occ,ox,Uiso,Uaniso = cif_file.read(fname)
    uc = UnitCell(a=cell[0],b=cell[1],c=cell[2],alpha=cell[3],beta=cell[4],gamma=cell[5])
    for j in range(len(labels)):
        if atsyms is not None: atsym = atsyms[j]
        else: atsym = None
        uc.add_site(labels[j],coords[j][0],coords[j][1],coords[j][2],atsym=atsym,
                    occ=occ[j],ox=ox[j],Uiso=Uiso[j],Uaniso=Uaniso[j]) 
    if symops is not None:
        for sym in symops:
            uc.add_sym_op(sym=sym)
    return uc

def write_cif(uc, fname, p1_list=True, na=1, nb=1, nc=1):
    """
    Write unit cell data as a cif file

    Arguments:
    ----------
    * fname: file name for output
    * p1_list: if True list fractional coordinates of the full P1 unit cell
               if False just the assymetric unit is written
    * na,nb,nc: if p1_list is True then list coordinates of expanded cell
                ie. list of coordinates of multiple cells. These should be 
                positive integers (expansion is symmetric about the origin)
    
    Notes:
    ------
    If p1_list is True then symetry operations are not written to the file
    If p1_list is False the na,nb,nc arguments are ignored
    """
    labels = []; atsym  = []; coords = []; occ    = []
    ox     = []; Uiso   = []; Uaniso = []
    if p1_list == True:
        atom_list = uc.atom_list(cartesian=False,na=na,nb=nb,nc=nc)
        atom_list.sort(ascend=True)
        for j in range(atom_list.natoms):
            labels.append(atom_list.labels[j])
            atsym.append(atom_list.atsym[j])
            coords.append(atom_list.coords[j])
            occ.append(atom_list.occ[j])
            ox.append(atom_list.ox[j])
            Uiso.append(atom_list.Uiso[j])
            Uaniso.append(atom_list.Uaniso[j])
    else:
        for j in range(len(uc.sites)):
            labels.append(uc.sites[j].label)
            atsym.append(uc.sites[j].atsym)
            coords.append([uc.sites[j].x, uc.sites[j].y, uc.sites[j].z])
            occ.append(uc.sites[j].occ)
            ox.append(uc.sites[j].ox)
            Uiso.append(uc.sites[j].Uiso)
            Uaniso.append(uc.sites[j].Uaniso)
    if p1_list == False:
        symop = []
        for W in uc.pg.W:
            W_str = uc.pg.W_str(W, condense=True)
            symop.append(W_str[0])
    else:
        symop = None
    cif_file.write(fname, labels, uc.lattice, coords, atsym=atsym, symop=symop, 
                   occ=occ, ox=ox, Uiso=Uiso, Uaniso=Uaniso)

##########################################################################
class UnitCell:
    """
    Unit cell class. 

    Data:
    -----
    self.sites   -> assymetric unit.  list of AtomicSite's
    self.pg      -> symmetry operations. PositionGenerator 
    self.lattice -> unit cell lattice.  lattice.Lattice
    """
    def __init__(self, a=10., b=10., c=10., alpha=90., beta=90., gamma=90., verbose=True):
        """
        Initialize

        Parameters:
        -----------
        * a,b,c in angstroms 
        * alpha, beta, gamma in degrees
        """
        self._description = "Bulk"
        self.verbose = verbose
        self.sites   = [] 
        self.pg      = PositionGenerator() 
        self.lattice = lattice.Lattice(a=a,b=b,c=c,alpha=alpha,beta=beta,gamma=gamma)
    
    def __repr__(self):
        return(self._write(long_fmt=self.verbose))

    def _write(self,long_fmt=False):
        lout = "\nLattice parameters:\n"
        lout = lout + repr(self.lattice)
        if len(self.sites) > 0:
            lout = lout + "\nAssymetric unit:\n"
            lout = lout + "sym    x        y      z    label  occ    ox     Uiso       U11       U12        U13       U22       U23      U33\n"
            for j in range(len(self.sites)):
                lout = lout + "%2s  %4.4f  %4.4f  %4.4f  %4s" % (self.sites[j].atsym, self.sites[j].x,
                                                                 self.sites[j].y, self.sites[j].z, 
                                                                 self.sites[j].label,)
                lout = lout + "  %3.3f  %+3.1f  %6.5f" % (self.sites[j].occ, self.sites[j].ox, self.sites[j].Uiso)
                lout = lout + "   %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n"  % (self.sites[j].Uaniso[0], self.sites[j].Uaniso[1],
                                                                                  self.sites[j].Uaniso[2], self.sites[j].Uaniso[3],
                                                                                  self.sites[j].Uaniso[4], self.sites[j].Uaniso[5])
        if (len(self.pg.W) > 0) and (long_fmt == True):
            lout = lout + "\nSymmetry operations:\n"
            lout = lout + repr(self.pg)
        return lout

    def show(self, long_fmt=False):
        """
        Print to screen

        Arguments:
        ----------
        * long_fmt: if True use long format (include symm ops), if False short format
        """
        print(self._write(long_fmt=long_fmt))

    def write(self, fname="unitcell.out", long_fmt=False):
        """
        write to a file

        Arguments:
        ----------
        * fname: output file 
        * long_fmt: if True use long format (include symm ops), if False short format
        """
        fout = open(fname,'w')
        fout.write(self._write(long_fmt=long_fmt))
        fout.close()

    def write_xyz(self, fname="unitcell.xyz", cartesian=True, na=1, nb=1, nc=1, long_fmt=False):
        """
        Write an xyz file

        Arguments:
        ----------
        * fname: file name for output
        * cartesian: if True output cartesian coordinates, otherwise fractional 
        * na, nb, nc: number of unit cell repeats in each direction. These should be 
                    positive integers (ie expansion is symmetric about the origin)
        * long_fmt:  if True writes long format

        Returns:
        -------
        * If fname is None then this returns an AtomList instance
        * If fname is not None, output is to the file

        Notes:
        ------
        This always lists a full (P1) unit cell
        """
        atom_list = self.atom_list(cartesian=cartesian,na=na,nb=nb,nc=nc)
        atom_list.sort(ascend=True)
        atom_list.write_xyz(fname,long_fmt=long_fmt)

    def add_site(self,label,x,y,z,atsym=None,occ=1.,ox=99,Uiso=0,Uaniso=[0.,0.,0.,0.,0.,0.]):
        """
        Add a site to the assymetric unit

        Parameters:
        -----------
        * label:  site label
        * x,y,z:  fractional coordinates
        * atsym:  atomic symbol
        * ox:     oxidation state
        * occ:    occupancy factor
        * Uiso:   isotropic displacement param
        * Uaniso: anisotropic displacement param

        Notes:
        -----
        If atsym is None we parse the atomic symbol from the site label.
        We assume in this case that the first character(s) of the label
        give the atomic symbol, e.g. C1, Fe2, O1 etc..
        """
        if atsym is None:
            atsym = ''
            if label[0].isupper(): atsym = atsym + label[0]
            if len(label) > 1:
                if label[1].islower(): atsym = atsym + label[1]
        if atsym not in atomic_symbols: atsym = "XX"
        site = AtomicSite(label,x,y,z,atsym=atsym,ox=ox,occ=occ,Uiso=Uiso,Uaniso=Uaniso)
        self.sites.append(site)
    
    def rem_site(self,label):
        """
        Remove a site from the assymetric unit

        Parameters:
        ----------
        * label: site label
        """
        rem = -1
        for j in range(len(self.sites)):
            if self.sites[j].label == label:
                rem = j
                break
        if rem > -1:
            self.sites.pop(rem)

    def add_sym_op(self,sym='x,y,z',shift=''):
        """
        Add a symmetry operator

        Parameters:
        ----------
        * sym and shift are strings with comma delimeted set of
          characters that defines the opertions.  e.g.
          sym = "x,y,z", shift = "0,0,0"
          sym = "-y,z,x+y", shift = "1/2,1/2,0"

        Example:
        --------
        >>unit_cell.add_op(sym="x,y,z",shift="1/2, 1/2, 0")
        """
        self.pg.add_sym_op(sym=sym,shift=shift)

    def atom_list(self,cartesian=False,na=1,nb=1,nc=1,update_labels=True):
        """
        Return listing of atoms in P1 cell

        Arguments:
        ----------
        * cartesian: if True return cartesian coordinates, False is fractional coordinates
        * na,nb,nc: number of cell repeats in each direction (positive integers)
        * update_labels: update the atomic labels with an index number

        Returns:
        --------
        AtomList instance (sorted in ascending order)
        """
        atsym  = []; labels  = []; coords  = []
        occ  = [];  ox = [];  Uiso = [];  Uaniso = []
        for j in range(len(self.sites)):
            copies = self.pg.sym_copy(self.sites[j].x, self.sites[j].y, self.sites[j].z)
            for v in copies:
                coords.append(v)
                labels.append(self.sites[j].label)
                atsym.append(self.sites[j].atsym)
                occ.append(self.sites[j].occ)
                ox.append(self.sites[j].ox)
                Uiso.append(self.sites[j].Uiso)
                Uaniso.append(self.sites[j].Uaniso)
        coords = num.array(coords)
        occ    = num.array(occ)
        ox     = num.array(ox)
        Uiso   = num.array(Uiso)
        Uaniso = num.array(Uaniso)
        # expand cell and create atom list instance 
        atom_list = _expand_frac(labels,atsym,coords,na,nb,nc,occ=occ,ox=ox,Uiso=Uiso,Uaniso=Uaniso)
        atom_list.lattice = self.lattice
        if update_labels==True:
            atom_list.update_labels(ascend=True)
        else:
            atom_list.sort(ascend=True)
        # transform to cartesian
        if cartesian == True:
            return atom_list.cartesian()
        else:
            return atom_list

    def transform(self,Va=None,Vb=None,Vc=None,shift=None):
        """
        Generate a new UnitCell given a set of basis transform vectors.

        Arguments:
        ----------
        * Va,Vb,Vc: Vectors, expressed in the original basis, defining the new 
                    set of basis vectors
        * shift: Vector, expressed in the original basis, defining a shift of 
                the unit cell 

        Returns:
        -------
        * New UnitCell instance

        Notes:
        ------
        If all V's are None, then a cartesian transform is performed
        """
        # create a lattice transform object
        trns = lattice.LatticeTransform(self.lattice, Va=Va, Vb=Vb, Vc=Vc, shift=shift)
        # use cartesian if new basis not specified
        if (Va is None) and (Vb is None) and (Vc is None): trns.cartesian()
        # create a new unit cell
        (a,b,c,alp,bet,gam) = trns.plat_params()
        new_cell = UnitCell(a=a,b=b,c=c,alpha=alp,beta=bet,gamma=gam)
        # transform the assymetric unit
        for site in self.sites:
            label = site.label
            atsym = site.atsym
            ox = site.ox
            occ = site.occ
            Uiso = site.Uiso
            vp = trns.vp([site.x, site.y, site.z])
            Uaniso = num.zeros((3,3))
            Uaniso[0,0] = site.Uaniso[0]; Uaniso[0,1] = site.Uaniso[1]
            Uaniso[0,2] = site.Uaniso[2]; Uaniso[1,1] = site.Uaniso[3]
            Uaniso[1,2] = site.Uaniso[4]; Uaniso[2,2] = site.Uaniso[5]
            Uaniso = num.dot(num.dot(trns.M, Uaniso), trns.G)
            # add atom to the new cell
            new_cell.add_site(label,vp[0],vp[1],vp[2],atsym=atsym,ox=ox,occ=occ,Uiso=Uiso,
                              Uaniso=num.array([Uaniso[0,0], Uaniso[0,1], Uaniso[0,2],
                                                Uaniso[1,1], Uaniso[1,2], Uaniso[2,2]]))
        # transform the symmetry operators (sietz matricies)
        Q = num.zeros((4,4))
        Q[:3,:3] = trns.M; Q[:3,3]  = trns.q; Q[3,3]   = 1.
        P = num.zeros((4,4))
        P[:3,:3] = trns.N; P[:3,3]  = trns.p; P[3,3]   = 1.
        for W in self.pg.W:
            Wp = num.dot(Q, num.dot(W,P))
            new_cell.pg.W.append(Wp)
        return new_cell

class AtomicSite:
    """
    Hold parameters of unit cell atomic sites (asymetric unit)
    coordinates, atomic parameters (symbol, ox state etc),
    occupancy factor, displacement factors
    """
    def __init__(self,label,x,y,z,atsym=None,ox=99,occ=1.,Uiso=0.,Uaniso=[0.,0.,0.,0.,0.,0.]):
        """
        initialize
        """
        self.label  = label    # site label
        self.x      = x        # x fractional coord
        self.y      = y        # y fractional coord
        self.z      = z        # z fractional coord
        self.atsym  = atsym    # atomic symbol
        self.ox     = ox       # oxidation state
        self.occ    = occ      # site occupancy 
        self.Uiso   = Uiso     # isotropic displacement parameter
        #self.Uaniso = Uaniso   # anisotropic displacement tensor
        self.Uaniso = num.array(Uaniso)   # anisotropic displacement tensor

    def __repr__(self,):
        lout = "label = %s, " % self.label
        lout = lout + "symbol = %s\n" % self.atsym
        lout = lout + "x = %6.5f, y = %6.5f, z = %6.5f\n" % (self.x, self.y, self.z)
        lout = lout + "occ = %4.3f,  ox-state = %3.2f,  " % (self.occ, self.ox)
        lout = lout + "Uiso = %6.5f\n"  % self.Uiso
        lout = lout + "U11 = %6.5f, U12 = %6.5f, U13 = %6.5f, " % (self.Uaniso[0], self.Uaniso[1],self.Uaniso[2])
        lout = lout + "U22 = %6.5f, U23 = %6.5f, U33 = %6.5f\n" % (self.Uaniso[3], self.Uaniso[4],self.Uaniso[5])
        return lout

class PositionGenerator:
    """
    Class to generate equivalent positions given a set of symmetry operators
    """
    def __init__(self):
        """ init """
        self.W  = []  # symmetry operations as Sietz (Augmented) matricies

    def __repr__(self):
        lout = ""
        for j in range(len(self.W)):
            lout = lout + "\nR=(%s)  t=(%s)\n" % self.W_str(self.W[j], condense=False)
            lout = lout + num.array2string(self.W[j], precision=3)
            lout = lout + "\n"
        return(lout)

    def W_str(self, W, condense=True):
        """
        Make a string representation of the Sietz matrix

        Arguments:
        ---------
        * W: an augmented (4x4) matrix
        * Condense: if False, shift is returned as a seperate string

        Returns:
        --------
        * (sym, shift): string representation of the symmetry operation
        """
        def _str(vec):
            v = ''
            s = ''
            if vec[0] == 1:    v = v + '+x'
            elif vec[0] == -1: v = v + '-x'
            if vec[1] == 1:    v = v + '+y'
            elif vec[1] == -1: v = v + '-y'
            if vec[2] == 1:    v = v + '+z'
            elif vec[2] == -1: v = v + '-z'
            if vec[3] == 0: 
                if condense == True: s = ""
                else: s = "0"
            else:
                if vec[3] > 0:
                    if condense == True: 
                        s = "+%1.3f" % vec[3]
                    else:
                        s = "%1.3f" % vec[3]
                else:
                    s = "%1.3f" % vec[3]
            if len(v) == 0: return v,s
            if v[0] == "+": v = v[1:]
            return v,s
        x, s_x = _str(W[0,:])
        y, s_y = _str(W[1,:])
        z, s_z = _str(W[2,:])
        if condense == True:
            x = x + s_x
            y = y + s_y
            z = z + s_z
            shift = ""
        else:
            shift = "%s,%s,%s" % (s_x,s_y,s_z)
        sym = "%s,%s,%s" % (x,y,z)
        return sym, shift

    def add_sym_op(self,sym='x,y,z',shift=''):
        """
        Add a new symmetry operator for generating positions

        Parameters:
        ----------
        * sym and shift are strings with comma delimeted set of
          characters that defines the opertions.  e.g.
            sym = "x,y,z", shift = "0,0,0"
            sym = "-y,z,x+y", shift = "1/2,1/2,0"
          The shift may also be included in sym, e.g.
            sym = "1/2-y,1/2+x,x+y", shift=""

        Example:
        --------
        >>sym1 = "x,y,z"
        >>shift1 = "1/2, 1/2, 0"
        >>p.add_sym_op(sym=sym1,shift=shift1)
        """
        #check shift
        if len(shift) > 0:
            shifts = shift.split(',')
            if len(shifts) != 3:
                print("Error parsing shift, should have 3 components: ", shift)
                return None
        else:
            shifts = ['','','']
        # break up sym into x,y,z parts
        syms = sym.split(',')
        if len(syms) != 3:
            print("Error parsing sym, should have 3 components: ", shift)
            return None

        x = syms[0] + shifts[0]
        y = syms[1] + shifts[1]
        z = syms[2] + shifts[2]
        #print 'x=',x,'y=',y,'z=',z
        
        W = self._make_seitz_matrix(x,y,z)
        if W is not None:
            self.W.append(W)

    def _make_seitz_matrix(self,x,y,z):
        """
        Generate augmented (seitz) matrix given
        string symbols for x,y,z, coordinates
        """
        def _vec(sym):
            v = num.array([0.,0.,0.,0.])
            if type(sym) != str:
                print("Error, passed a non-string symbol")
                return None
            sym = sym.replace('+','')
            sym = sym.replace(' ','')
            if '-x' in sym:
                v[0] = -1
                sym = sym.replace('-x','')
            elif 'x' in sym:
                v[0] = 1
                sym = sym.replace('x','')
            if '-y' in sym:
                v[1] = -1
                sym = sym.replace('-y','')
            elif 'y' in sym:
                v[1] = 1
                sym = sym.replace('y','')
            if '-z' in sym:
                v[2] = -1
                sym = sym.replace('-z','')
            elif 'z' in sym:
                v[2] = 1
                sym = sym.replace('z','')
            if len(sym) > 0:
                #sym = sym + '.'
                v[3] = eval(sym)
            return v
        #
        v1 = _vec(x)
        if v1 is None: return None
        v2 = _vec(y)
        if v2 is None: return None
        v3 = _vec(z)
        if v3 is None: return None
        v4 = [0.,0.,0.,1.]
        W = num.array([v1,v2,v3,v4])
        #print W
        return W

    def sym_copy(self,x,y,z,reduce=True,rem_dups=True):
        """
        Calc all sym copies of a position

        Parameters:
        -----------
        * x,y,z are fractional coordinates
        * reduce is flag to indicate that all
          positions must be in bounds  0 to 1
        * rem_dups is a flag to indicate if duplicates
          should be removed

        Outputs:
        --------
        * list of vectors of symmetry copy positions

        Note:
        -----
        * Fractional coords are rounded off to 6 decimals
        * Duplicates are removed if x,y,z copies match within
          3 decimals
        """
        v0 = num.array([float(x),float(y),float(z),1.0])
        vectors = []
        for W in self.W:
            vc = num.dot(W,v0)
            vc = num.around(vc,decimals=6)
            vectors.append(vc[:3])
        # reduce values
        if reduce == True:
            for j in range(len(vectors)):
                if vectors[j][0] >= 1.0 or vectors[j][0] < 0.0:
                    vectors[j][0] = _reduce_frac(vectors[j][0])
                if vectors[j][1] >= 1.0 or vectors[j][1] < 0.0:
                    vectors[j][1] = _reduce_frac(vectors[j][1])
                if vectors[j][2] >= 1.0 or vectors[j][2] < 0.0:
                    vectors[j][2] = _reduce_frac(vectors[j][2])
        # remove duplicates
        def _rem_dups(vectors):
            unique = []
            while len(vectors) > 0:
                v = vectors.pop(0)
                add = True
                for u in unique:
                    #tmp = num.equal(v,u)
                    #tmp = (num.fabs(u-v) < 0.001)
                    tmp = num.equal(num.round(v,decimals=3),num.round(u,decimals=3))
                    if tmp.all() == True:
                        add = False
                        break
                if add == True:
                    unique.append(v)
            return unique
        if rem_dups == True:
            vectors = _rem_dups(vectors)
        return vectors

##########################################################################
##########################################################################
def _test1():
    """
    test PositionGenerator for C2/m
    """
    sym1   = "x,y,z"
    sym2   = "-x,y,-z"
    sym3   = "-x,-y,-z"
    sym4   = "x,-y,z"
    shift0 = "0,0,0"
    shift1 = "1/2, 1/2, 0"
    p = PositionGenerator()
    p.add_sym_op(sym=sym1,shift=shift0)
    p.add_sym_op(sym=sym2,shift=shift0)
    p.add_sym_op(sym=sym3,shift=shift0)
    p.add_sym_op(sym=sym4,shift=shift0)
    p.add_sym_op(sym=sym1,shift=shift1)
    p.add_sym_op(sym=sym2,shift=shift1)
    p.add_sym_op(sym=sym3,shift=shift1)
    p.add_sym_op(sym=sym4,shift=shift1)

    print("0.15,0.0,0.33")
    vecs = p.sym_copy(0.15,0.0,0.33,reduce=True,rem_dups=True)
    for v in vecs: print(v)

    print("0.5,0.11,0.5")
    vecs = p.sym_copy(0.5,0.11,0.5,reduce=True,rem_dups=True)
    for v in vecs: print(v)

    print("0.25,0.25,0.25")
    vecs = p.sym_copy(0.25,0.25,0.25,reduce=True,rem_dups=True)
    for v in vecs: print(v)

    return p

##########################################################################
##########################################################################
if __name__ == "__main__":
    #p = _test1()
    uc = read_cif('COD_Fe2O3.cif')

