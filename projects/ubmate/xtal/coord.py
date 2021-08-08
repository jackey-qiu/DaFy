"""
Coordination calculations

Authors/Modifications:
-----------------------
* Tom Trainor (tptrainor@alaska.edu)

Notes:
-----
* See the bvparm.py file for a list of bond valence parameters


"""
##########################################################################
import numpy as num
import copy
from xtal.lattice import LatticeTransform
from xtal.bvparm import bvparm
from xtal.atom_list import AtomList, _expand_, _reduce_frac, merge_alist

##########################################################################
def coord_calcs(cell,rmax=2.5,labels=None):
    """
    coord calcs

    Arguments:
    ----------
    * cell: a UnitCell, SurfaceCell or AtomList instance
    * rmax: max range for neighbor calculations
    * labels: list of atom labels to calculate
              if None then calcs are performed for all atoms (see below)

    Returns:
    -------
    * Coord instance

    Notes:
    ------
    * if cell is a UnitCell and labels=None calcs are done for 
      each unique site label
    * if cell is a SurfaceCell instance and labels=None calcs are 
      done for all atoms in the P1 termination cell
    * if cell is a SurfaceCell instance you can pass labels from
      both the termination cell and the bulk cell
    * if cell is an AtomList and labels=None calcs are done for all
      atoms.  Note AtomList is not expanded - calcs are done only
      for the structure as passed in.
    """
    if cell._description == "Bulk": ty = 1
    elif cell._description == "Surface": ty = 2
    elif cell._description == "AtomList": ty = 3
    else: return None
    if ty == 1:
        coord = _coord_calc_uc(cell,labels,rmax=rmax)
    elif ty == 2:
        coord = _coord_calc_surf(cell,labels,rmax=rmax)
    elif ty == 3:
        coord = _coord_calc_atom_list(cell,labels,rmax=rmax)
    return coord

def _coord_calc_uc(uc, site_labels, rmax=2.5):
    """
    coordination calcs for unit cell instance
    """
    if site_labels == None:
        site_labels = []
        for site in uc.sites:
            if site.label not in site_labels:
                site_labels.append(site.label)
    trns = LatticeTransform(uc.lattice)
    trns.cartesian()
    coord = Coord()
    atoms = uc.atom_list()
    index = []
    for site in uc.sites:
        if site.label in site_labels:
            j = 0
            v = num.array([site.x, site.y, site.z])
            while j<atoms.natoms:
                if atoms.atsym[j] == site.atsym:
                    if num.all(num.around((atoms.coords[j] - v),decimals=3)==0):
                        index.append(j)
                        break
                j = j+1
    for idx in index:
        (xr,yr,zr) = _box_range(atoms.coords[idx],rmax,trns)
        (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(atoms.labels,atoms.atsym,atoms.coords,xr,Vr=[1.,0,0],
                                                            occ=atoms.occ,ox=atoms.ox)
        (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,coords,yr,Vr=[0,1.,0],occ=occ,ox=ox)
        (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,coords,zr,Vr=[0,0,1.],occ=occ,ox=ox)
        cpoly = _calc_cpoly(idx, rmax, trns, labels, atsym, coords, uc.lattice, occ, ox)
        coord.cpoly.append(cpoly)
    return coord

def _coord_calc_surf(surf, site_labels, rmax=2.5):
    """
    coordination calcs for surface cell instance
    """
    # if None, just get labels for the term cell
    if site_labels == None:
        site_labels = []
        for j in range(surf.p1term.natoms):
            if surf.p1term.labels[j] not in site_labels:
                site_labels.append(surf.p1term.labels[j])
    trns = LatticeTransform(surf.lattice)
    trns.cartesian()
    coord = Coord()
    term = surf.p1term
    bulk = surf.p1bulk
    ### expand bulk atoms in z (in bulk direction)
    # make sure we expand far enough that an atom
    # at the bottom of the bulk cell will have all its
    # neighbors (in bulk z-dir) within rmax
    max_z = num.ceil(rmax/surf.lattice.c)
    zr = list(range(0,int(max_z)+1))
    (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(bulk.labels, bulk.atsym, bulk.coords, zr, 
                                                        Vr=surf.Vr_s, occ=bulk.occ, ox=bulk.ox,
                                                        Uiso=bulk.Uiso, Uaniso=bulk.Uaniso)
    # adjust the coords (x,y) so we have a vertical column... 
    for j in range(len(labels)):
        if coords[j][0] >= 1.0 or coords[j][0] < 0.0:
            coords[j][0] = _reduce_frac(coords[j][0])
        if coords[j][1] >= 1.0 or coords[j][1] < 0.0:
            coords[j][1] = _reduce_frac(coords[j][1])
    bulk = AtomList(labels=labels,atsym=atsym,coords=coords,lattice=surf.lattice,occ=occ,ox=ox,Uiso=Uiso,Uaniso=Uaniso)
    ###
    atoms = merge_alist(term, bulk)
    atoms.sort(ascend=False)
    ### now expand in x/y
    for lbl in site_labels:
        idx = atoms.index(lbl,first=True)
        if len(idx) == 0: break
        (xr,yr,zr) = _box_range(atoms.coords[idx[0],:], rmax, trns)
        (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(atoms.labels,atoms.atsym,atoms.coords,xr,Vr=[1.,0,0],occ=atoms.occ,ox=atoms.ox)
        (labels,atsym,coords,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,coords,yr,Vr=[0,1.,0],occ=occ,ox=ox)
        ###
        temp = AtomList(labels=labels,atsym=atsym,coords=coords,lattice=surf.lattice,occ=occ,ox=ox)
        temp.sort()
        # now find the site again...
        index = temp.index(lbl)
        for j in index:
            if num.all(num.around((temp.coords[j] - atoms.coords[idx]),decimals=3)==0):
                cpoly = _calc_cpoly(j, rmax, trns, temp.labels, temp.atsym, temp.coords, surf.lattice, temp.occ, temp.ox)
                coord.cpoly.append(cpoly)
                break
    return coord

def _coord_calc_atom_list(atoms, site_labels, rmax=2.5):
    """
    coordination calcs for atom_list instance - no expansion
    """
    if site_labels == None:
        site_labels = []
        for lbl in atoms.labels:
            site_labels.append(lbl)
    trns = LatticeTransform(atoms.lattice)
    trns.cartesian()
    coord = Coord()
    index = []
    for lbl in site_labels:
        idx = atoms.index(lbl)
        index.extend(idx)
    for idx in index:
        cpoly = _calc_cpoly(idx, rmax, trns, atoms.labels, atoms.atsym, 
                            atoms.coords, atoms.lattice, atoms.occ, atoms.ox)
        coord.cpoly.append(cpoly)
    return coord

def _calc_cpoly(idx,rmax,trns,labels,atsym,coords,lattice,occ,ox):
    """
    coodination calcs
    """
    # cartesian coords
    nsite = len(labels)
    c_coords = num.zeros((nsite,3))
    for j in range(nsite):
        c_coords[j,:] = trns.vp(coords[j,:])
    # compute the radial distance of all atoms from the center atom
    c_coords_shift = c_coords - c_coords[idx]
    r = num.sqrt(num.sum(c_coords_shift * c_coords_shift, axis = 1))
    # sort by radial distance
    #srt_idx        = num.argsort(r, kind='mergesort')
    #r              = r[srt_idx]
    #coords         = coords[srt_idx,:]
    #c_coords       = c_coords[srt_idx,:]
    #c_coords_shift = c_coords_shift[srt_idx,:]
    #occ            = occ[srt_idx]
    #ox             = ox[srt_idx]
    #labels         = labels[srt_idx]
    #atsym          = atsym[srt_idx]
    # create a coord poly for all atoms within rmax
    #coord_poly = CoordPoly(labels[0],atsym[0],coords[0],lattice,occ=occ[0],ox=ox[0])
    coord_poly = CoordPoly(labels[idx],atsym[idx],coords[idx],lattice,occ=occ[idx],ox=ox[idx])
    j = 0
    while j < nsite:
        if r[j] <= rmax:
            if (r[j] != 0) and (j != idx):
                coord_poly.add_nbr(labels[j],atsym[j],coords[j],r=r[j],ox=ox[j],occ=occ[j],)
        j += 1
    coord_poly.calc_bv()
    return coord_poly

def _box_range(coord,rmax,trns):
    """
    Find corners of a cartesian box surrounding the selected atom
    Note trns.vp() gives cartesian indicies from fractional
    and trns.v() gives fractional indicies from cartesian.
    The scale factors below forces each corner of the box to be 
    at a distance of '1.5*rmax' from the center selected atom.
    The box coordinates are converted to fractional
    """
    scale = 1.5*rmax/num.sqrt(3.)
    box   = num.zeros((8,3))
    rcen  = trns.vp(coord)  # center of box in cart coord
    box[0,:] = trns.v(scale*num.array([1.,1.,1.])    + rcen)
    box[1,:] = trns.v(scale*num.array([1.,-1.,1.])   + rcen)
    box[2,:] = trns.v(scale*num.array([-1.,-1.,1.])  + rcen)
    box[3,:] = trns.v(scale*num.array([-1.,1.,1.])   + rcen)
    box[4,:] = trns.v(scale*num.array([1.,1.,-1.])   + rcen)
    box[5,:] = trns.v(scale*num.array([1.,-1.,-1.])  + rcen)
    box[6,:] = trns.v(scale*num.array([-1.,-1.,-1.]) + rcen)
    box[7,:] = trns.v(scale*num.array([-1.,1.,-1.])  + rcen)
    # Get min/max in x,y,z (frac).  If range is outside (0,1) then 
    # expand the cell.  If its a surface, assume surface is the 
    # +z-direction, so dont expand in that direction.
    min_x = num.min(box[:,0]); max_x = num.max(box[:,0])
    min_y = num.min(box[:,1]); max_y = num.max(box[:,1])
    min_z = num.min(box[:,2]); max_z = num.max(box[:,2])
    # expand in x
    xr = list( range( int(num.floor(min_x)), int(num.ceil(max_x)) ))
    yr = list( range( int(num.floor(min_y)), int(num.ceil(max_y)) ))
    zr = list( range( int(num.floor(min_z)), int(num.ceil(max_z)) ))
    #return box  # for testing (_test1)
    return xr,yr,zr

class Coord:
    """
    Collection of CoordPoly's
    """
    def __init__(self,cpoly=None):
        self.verbose = True
        self.cpoly = []
        if cpoly is not None:
            self.cpoly.append(cpoly)

    def __repr__(self,):
        return self._write(long_fmt=self.verbose)

    def _write(self,long_fmt=True):
        lout = ""
        for cpoly in self.cpoly:
            lout = lout + cpoly._write(long_fmt=long_fmt)
            lout = lout + "\n"
        return lout

    def show(self,long_fmt=True):
        print(self._write(long_fmt=long_fmt))

    def write(self,fname="coord.out",long_fmt=True):
        fout = open(fname,'w')
        fout.write(self._write(long_fmt=long_fmt))
        fout.close()

class CoordPoly:
    """
    For a given central atom keeps track of neighbors,
    distances and angles...
    """
    def __init__(self,label,atsym,coord,lattice,occ=1,ox=9):
        """
        central atom params
        """
        self.verbose = True
        if ox > 9: ox = 9
        self.lattice = lattice   # Lattice instance
        self.label   = label     # site label
        self.atsym   = atsym     # atomic symbol
        self.coord   = coord     # fractional coordinates [x,y,z]
        self.ox      = ox        # oxidation state (9 if not specified)
        self.occ     = occ       # site occupancy
        self.nbr     = []        # list of neighbors
        self.sum_s   = 0.        # bond valence sum

    def __repr__(self):
        return self._write(long_fmt=self.verbose)
        
    def _write(self,long_fmt=True):
        lout = "Central atom: %s, ox_state=%i, occ=%3.3f, "  %  (self.label, self.ox, self.occ)        
        lout = lout + "coords=(%6.3f,%6.3f,%6.3f)\n"  %   (self.coord[0],
                                                           self.coord[1],
                                                           self.coord[2])
        lout = lout + "  Coordination sphere\n"
        n_nbr = len(self.nbr)
        for j in range(n_nbr):
            lout = lout + "    nbr_%i" % (j+1)
            lout = lout + " -> %s(%i), label=%10s," % (self.nbr[j].atsym, self.nbr[j].ox, self.nbr[j].label)
            lout = lout + " coords=(%6.3f,%6.3f,%6.3f)," % (self.nbr[j].coord[0],
                                                            self.nbr[j].coord[1],
                                                            self.nbr[j].coord[2])
            lout = lout + " r=%6.3f, s=%6.3f, occ=%3.2f\n" % (self.nbr[j].r, self.nbr[j].s,self.nbr[j].occ)
        lout = lout + "    Sum_s = %3.3f\n"  % self.sum_s
        if long_fmt==True:
            lout = lout + "  Angles(degrees):\n"
            for j in range(n_nbr):
                for k in range(j+1,n_nbr):
                    angle = self.calc_angle(j,k)
                    lout = lout + "    nbr_%i--x--nbr_%i   %5.2f\n" % (j+1,k+1,angle)
        return lout

    def add_nbr(self,label,atsym,coord,r=None,ox=9,occ=1):
        """
        add a nieghbor atom
        """
        if ox > 9: ox = 9
        self.nbr.append(NbrAtom(label,atsym,coord,r,ox,occ))
        
    def calc_bv(self,):
        """
        compute bond-valence sums
        """
        n_nbr = len(self.nbr)
        self.sum_s = 0.
        for j in range(n_nbr):
            idx = "%s%i%s%i" % (self.atsym,self.ox,self.nbr[j].atsym,self.nbr[j].ox)
            x = bvparm.get(idx)
            if x is None:
                idx = "%s%i%s%i" % (self.nbr[j].atsym,self.nbr[j].ox,self.atsym,self.ox)
                x = bvparm.get(idx)
            if x is None:
                s = 0.
            else:
                ro = x[0]; b = x[1]
                r = self.nbr[j].r
                s = self.nbr[j].occ * num.exp( (ro-r)/b )
                self.nbr[j].s = s
                self.sum_s += s
                
    def calc_angle(self,j,k):
        """
        calculate angles
        """
        v1 = self.nbr[j].coord - self.coord
        v2 = self.nbr[k].coord - self.coord
        return self.lattice.angle(v1,v2)

class NbrAtom:
    """
    Nieghbor atom
    """
    def __init__(self, label, atsym, coord, r, ox, occ):
        self.label = label
        self.atsym = atsym
        self.coord = coord
        self.r     = r
        self.s     = 0.
        self.ox    = ox
        self.occ   = occ

##########################################################################
def _test1():
    """note change _box_range above to return the 'box'..."""
    from pyxrs.xtal.lattice import Lattice
    lattice = Lattice(a=5.0346, b=5.0346, c=13.7473, alpha=90.0,beta=90.0,gamma=120.0)
    coord = num.array([0. , 0.  , 0.35534 ])
    rmax=2.5
    # compute the box and center in cart coords so we can plot it
    trns = LatticeTransform(lattice)
    trns.cartesian()
    fbox =  _box_range(coord,rmax,trns,surf=False)
    cen = trns.vp(coord)
    cbox = num.zeros((8,3))
    for j in range(8):
        cbox[j,:] = trns.vp(fbox[j,:])

    # plot cbox about cen
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx = ax.plot([cen[0]],[cen[1]],[cen[2]],'o')
    for j in range(8):
        xx = ax.plot([cen[0],cbox[j,0]],[cen[1],cbox[j,1]],[cen[2],cbox[j,2]],'*-')

def _test2():
    from pyxrs.xtal.lattice import Lattice
    lattice = Lattice(a=5., b=5., c=5., alpha=90.0,beta=90.0,gamma=90.0)
    coord = num.array([0.5, 0.5, 0.5])
    rmax=1
    # compute the box and center in cart coords so we can plot it
    trns = LatticeTransform(lattice)
    trns.cartesian()
    xr,yr,zr =  _box_range(coord,rmax,trns,surf=True)
    print(xr,yr,zr)

##########################################################################
##########################################################################
if __name__ == "__main__":
    #_test1()
    _test2()

