"""
Surface indexing / unit cell

Authors/Modifications:
-----------------------
* Tom Trainor (tptrainor@alaska.edu)

Todo:
-----
* Test Uansiso transform....
* Include an option in surface_to_xyz to make "terminations" on the 
  top and bottom of the slab.
* Determine plane group of the surface

Notes:
------
* The SurfaceCell class below generates a unit cell in which 
  the as and bs vectors lie in the surface (hkl) plane and 
  the cs vector is parallel to the surface normal with magnitude
  |cs| = nd*d_hkl  (nd=1 by default).  The system is right handed 
  and has the same origin as the bulk unit cell.

  Note that forcing the cs vector to be parrallel to the surface
  normal may result in a non-rational repeat in that direction.  Therefore,
  we define a slab repeat vector (Vr) that describes how surface
  slabs need to be shifted in order to generate space filling 3-d models.  

* Given a bulk UnitCell and surface index (hkl) the SurfaceCell class will 
  attempt to find the best set of bulk lattice vectors used to as and bs axes.  
  This is done by brute force searching over a range of lattice points surrounding
  the origin of the bulk cell.  

  The set of lattice vectors that lie within the surface plane (Vs) are found
  using the law of rational indicies.  That is for a given hkl (which should be  
  a relativley prime set of integers that defines the surface plane), a lattice  
  vector (integer x,y,z) lies within the hkl plane that passese 
  through the origin of the bulk unit cell if the following is true:
    hx + ky + lz = 0

  We can also find the set of possible repeat vectors, terminating on the n'th 
  plane below the plane passing through the bulk cell origin by finding lattice 
  vectors that satisfy:
    hx + ky + lz = -n

  Note that for non-primitive bulk unit cells this algorithm may miss 
  valid surface vectors, since it only searches through integers.  If
  the bulk unit cell is not primitive a matrix that defines the primitive bulk 
  unit cell can be passed to the SurfaceCell class.  The routine will then 
  search for surface/repeat vectors using the primitive setting and then convert
  them back to your original setting.  For transformation matricies see 
  for example International Tables Vol A Table 5.1.3.1.

  To define the surface unit cell the routine choses the two 
  smallest non-colinear vectors from the set of Vs to define the surface
  a_s and b_s basis vectors.  The smallest in-plane lattice vector is
  used to define the a_s axis.  The surface normal (scaled by n) defines the c_s
  axis.  The second smallest in-plane vector (that is non-colinear with a_s) 
  that makes the system right handed is selected as b_s.

  The repeat vector that is closest to the surface normal direction, 
  terminating on the nth plane below the surface plane is used for Vr.  

* The in-plane and (optionally) repeat vectors may also be explicity set by the user.   
  In this case the routine will check to make sure these are in-plane, and then  
  determine the appropriate repeat vector if not specified.  

* The primary use of this module is to generate atomic coordinates of a 
  superstructure (or surface slab) using the surface indexing 
  (see the surface_to_xyz function).  

  The convention used for generating "suface slabs" is as follows (note this
  discussion is in terms of the surface coordinate system, i.e. let a = a_s etc).  
  We define the in-plane directions in terms of the a and b axis.  Therefore
  x,y coordinates are in-plane.  The c axis is parallel to the surface normal, 
  therefore +z coordinates are in the "surface-direction", while -z coordinates
  are in the "bulk-direction".  

  The origin of the system (0,0,0) defines the bottom of the uppermost 
  "bulk-unit-cell".  Therefore, we define the termination of the bulk 
  structure to be at z=+1.  

  When a super-cell (slab) is generated we expand symmetrically about the origin
  (0,0) in the in-plane direction.

  The "bulk" part of the slab (that is z<1) is generated by specifying a 
  number of repeats into the bulk (nbulk).  Setting nbulk = 1 generates
  a super structure that includes 1 "bulk" cell (0<=z<1), nbulk=2 has two 
  layers of bulk cells (-1<z<1) etc.  Setting nbulk =0 gives a structure
  with no "bulk" cells.  
  
  An additional "termination" is added to the top of the superstructure by
  specifying a "term" value.  This parameter is the index of atomic layer at 
  which the "termination-cell" model should be terminated.  If term=0, a termination 
  cell is added that includes all atomic layers of the cell, i.e. all atoms with 
  z-coords of 1<=z<2.  If term = -3, then the termination cell will include 
  all atomic layers up to third atomic layer below the reference surface at z=2.  
  If term = +3 then the termination will be at the third atomic layer above the 
  reference surface at z=1, and therefore will have z-coordinate greater than 2.   
  Setting term=None generates a structure with no "termination" layer. 
  
           term=-1         term=+2
 
                          O   O   O  
                            x   x
                         ------------  z = 2
                          O   O   O  
             x   x          x   x        termination cell
           O   O   O      O   O   O    
             x   x          x   x
          -----------    ------------  z = 1   
           O   O   O      O   O   O  
             x   x          x   x
           O   O   O      O   O   O       bulk cell
             x   x          x   x
          -----------    ------------  z = 0
          O   O   O      O   O   O  
            x   x          x   x
          O   O   O      O   O   O   
            x   x          x   x
          -----------    ------------  z = -1
                      ||
                      ||
                      \/
                     bulk 

References:
-----------
* Trainor, Eng and Robinson (2002) J. Appl. Cryst., 35, 696-701.

"""
##########################################################################
import numpy as num
import numpy as np
import pandas as pd
import copy
from xtal import lattice
from xtal import unitcell
from xtal.atom_list import merge_alist, AtomList, _expand_, _expand_frac, _reduce_frac 

##########################################################################
class SurfaceCell:
    """
    Surface unit cell class
    """
    def __init__(self,bulk_uc,hkl=None,nd=1,term=0,bulk_trns=None):
        """
        * bulk_cell: UnitCell object defining the bulk unit cell
        * hkl: surface indicies (in the bulk unit cell indexing)
        * nd: defines the "thickness" of the surface unit cell as a 
              multiple of d_hkl, ie |c_s| = nd*d_hkl
        * term: atomic layer that defines the surface termination.
        * bulk_trns: transformation matrix to make the bulk unit cell
                    primitive.  Use this so the surface lattice search 
                    finds the smallest set of lattice vectors to define 
                    the surface unit cell (for non-primitive bulk lattice)
        """
        self._description = "Surface"
        self.verbose = True
        self.bulk_uc   = bulk_uc   # bulk UnitCell object
        self.bulk_trns = None      # transform matrix to make bulk cell primitive
        self.hkl       = None      # hkl of surface plane (in bulk indexing)
        self.nd        = 1         # number of d-space repeats to define magnitude of cs
        self.term      = int(term) # atomic layer defining the surface termination
        self.Vs_lst    = None      # list of in-plane bulk lattice vectors (in bulk coords)
        self.Vr_lst    = None      # list of repeat bulk lattice vectors (in bulk coords)
        self.Va        = None      # bulk lattice vector selected to define surface a-axis (in bulk coords)
        self.Vb        = None      # bulk lattice vector selected to define surface b-axis (in bulk coords)
        self.Vc        = None      # bulk vector selected to define surface c-axis (in bulk coords)
        self.Vr        = None      # bulk lattice vector selected to define slab repeat (in bulk coords)
        self.Vr_s      = None      # as above defined in surface coordinate system
        self.transform = None      # bulk-to-surface LatticeTransform object
        self.lattice   = None      # lattice object defined for surface coordinated system
        self.p1bulk    = None      # AtomList with the P1 bulk unit cell atoms (in surface coords)
        self.p1term    = None      # AtomList with the P1 termination unit cell atoms (in surface coords)

        if bulk_trns is not None:
            self.bulk_trns = num.array(bulk_trns)
        if hkl is not None:
            self.find_surf_lattice(hkl,nd=nd)

    def __repr__(self):
        return self._write(long_fmt=self.verbose)

    def _write(self,long_fmt=True):
        lout = "\nBulk unit cell\n"
        lout = lout + repr(self.bulk_uc.lattice)
        #lout = lout + repr(self.bulk_uc)
        lout = lout + "\nSurface plane = %s, nd=%i\n" % (repr(self.hkl), self.nd)
        lout = lout + "  Va = %s\n" % repr(self.Va)
        lout = lout + "  Vb = %s\n" % repr(self.Vb)
        lout = lout + "  Vc = %s\n" % repr(self.Vc)
        lout = lout + "  Vr = %s\n" % repr(self.Vr)
        lout = lout + "  Vr_s = %s\n" % repr(self.Vr_s)
        lout = lout + "\nSurface unit cell\n"
        lout = lout + repr(self.lattice)
        lout = lout + "\nP1 surface cell (surface fractional coords)\n"
        # list P1 cell.  note atoms are stored in 
        # ascending order, here we list them descending order
        lout = lout + "  sym     x       y         z          label"
        if long_fmt==True:
            lout = lout + "   occ     ox    Uiso       U11       U12        U13       U22       U23       U33"
        lout = lout + "\nTermination  (termination layer = %i)\n"  % self.term
        for j in range(self.p1term.natoms-1,-1,-1):
            lout = lout + "  %2s  %6.5f  %6.5f  %6.5f  %12s" % (self.p1term.atsym[j], self.p1term.coords[j][0], 
                                                              self.p1term.coords[j][1], self.p1term.coords[j][2],
                                                              self.p1term.labels[j])
            if long_fmt==True:
                lout = lout + "  %3.3f  %+3.1f  %6.5f" % (self.p1term.occ[j], self.p1term.ox[j], self.p1term.Uiso[j])
                lout = lout + "   %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f"  % (self.p1term.Uaniso[j,0], self.p1term.Uaniso[j,1],
                                                                                self.p1term.Uaniso[j,2], self.p1term.Uaniso[j,3],
                                                                                self.p1term.Uaniso[j,4], self.p1term.Uaniso[j,5])
            lout = lout + "\n"
        lout = lout + "Bulk\n"
        for j in range(self.p1bulk.natoms-1,-1,-1):
            lout = lout + "  %2s  %6.5f  %6.5f  %6.5f  %12s" % (self.p1bulk.atsym[j], self.p1bulk.coords[j][0], 
                                                              self.p1bulk.coords[j][1], self.p1bulk.coords[j][2], 
                                                              self.p1bulk.labels[j])
            if long_fmt==True:
                lout = lout + "  %3.3f  %+3.1f  %6.5f" % (self.p1bulk.occ[j], self.p1bulk.ox[j], self.p1bulk.Uiso[j])
                lout = lout + "   %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f"  % (self.p1bulk.Uaniso[j,0], self.p1bulk.Uaniso[j,1],
                                                                                self.p1bulk.Uaniso[j,2], self.p1bulk.Uaniso[j,3],
                                                                                self.p1bulk.Uaniso[j,4], self.p1bulk.Uaniso[j,5])
            lout = lout + "\n"
        return lout

    def _write_pandas_df(self):
        lout = '<p>{}</p>'.format("Bulk unit cell\n")
        lout = lout + '<p>{}</p>'.format(repr(self.bulk_uc.lattice))
        #lout = lout + repr(self.bulk_uc)
        lout = lout + '<p>{}</p>'.format("\nSurface plane = %s, nd=%i\n" % (repr(self.hkl), self.nd))
        lout = lout + '<p>{}</p>'.format("  Va = %s\n" % repr(self.Va))
        lout = lout + '<p>{}</p>'.format("  Vb = %s\n" % repr(self.Vb))
        lout = lout + '<p>{}</p>'.format("  Vc = %s\n" % repr(self.Vc))
        lout = lout + '<p>{}</p>'.format("  Vr = %s\n" % repr(self.Vr))
        lout = lout + '<p>{}</p>'.format("  Vr_s = %s\n" % repr(self.Vr_s))
        lout = lout + '<p>{}</p>'.format("\nSurface unit cell\n")
        lout = lout + '<p>{}</p>'.format(repr(self.lattice))
        lout = lout + '<p>{}</p>'.format("\nP1 surface cell (surface fractional coords)\n")+'<hr />'

        labels = ['atm_id', 'atm_el','x_frg','y_frg','z_frg','Biso','occ','mult','type']
        results_slab = {'atm_id':list(self.p1term.labels[::-1]) + list(self.p1bulk.labels[::-1]),
                        'atm_el':list(self.p1term.atsym[::-1]) + list(self.p1bulk.atsym[::-1]),
                        'x_frg':list(self.p1term.coords[:,0][::-1]) + list(self.p1bulk.coords[:,0][::-1]),
                        'y_frg':list(self.p1term.coords[:,1][::-1]) + list(self.p1bulk.coords[:,1][::-1]),
                        'z_frg':list(self.p1term.coords[:,2][::-1]) + list(self.p1bulk.coords[:,2][::-1]),
                        'Biso':list(self.p1term.Uiso[::-1]) + list(self.p1bulk.Uiso[::-1]),
                        'occ':list(self.p1term.occ[::-1])+list(self.p1bulk.occ[::-1]),
                        'mult':[1]*(self.p1term.natoms+self.p1bulk.natoms),
                        'type':['surface']*self.p1term.natoms + ['bulk']*self.p1bulk.natoms}
        result_df = pd.DataFrame(results_slab, columns = labels)
        return lout, result_df
        
    def show(self,long_fmt=True):
        """
        write summary to std out

        Arguments:
        ----------
        * long_fmt: if True use long format (include symm ops), if False short format
        """
        print(self._write(long_fmt=long_fmt))

    def write(self,fname="surface.out",long_fmt=True):
        """
        write summary to a file

        Arguments:
        ----------
        * fname: output file (if None, print to stdout)
        * long_fmt: if True use long format (include symm ops), if False short format
        """
        fout = open(fname,'w')
        fout.write(self._write(long_fmt=long_fmt))
        fout.close()

    def write_xyz(self,fname="surfacecell.xyz",cartesian=True,na=1,nb=1,nbulk=1,term=-99,long_fmt=False):
        """
        List surface file

        Parameters:
        -----------
        * fname: output file
        * cartesian: if True output in cartesian, otherwise in fractional coordinates
        * na, nb: number of cell repeats in the in-plane (a,b) directions
        * nbulk: number of repeats of the unit cell in the "bulk" direction (i.e. -c)
                if nbulk=0, then the model will only include the "termination-layer"
                with no unit cell repeats in the "bulk" direction
        * term: atomic layer that defines the surface termination.  
                if term=None the model will only consist of the "bulk-surface-model"
                if term=-99 use the current value of term as already specified in the surf object
                if term=0 then the model terminates at the top of the surface unit cell
                if term=+1 then one atomic layer is added on top
                if term=-1 then one atomic layer is removed from the top 
                etc.
        * long_fmt: output in long format

        Notes:
        ------
        * na, nb, nbulk should all be positive integers
        * na and nb defines the inplane extent of the slab
        * nbulk defines the thickness of the "bulk" part of the slab
        * term, if not None, defines which atomic layer is the termination 
        (and therefore defines how "thick" the terminating layer of the slab is)
        if term is not -99, then surf will be updated with the new term value
        * Note if nbulk=0 and term=None this function returns None (ie no model!)
        """
        atlist = self.atom_list(cartesian=cartesian,na=na,nb=nb,nbulk=nbulk,term=term)
        if atlist is None: return
        fout = open(fname,'w')
        fout.write("%i\n" % atlist.natoms) 
        fout.write(atlist._write(long_fmt=long_fmt,header=True))
        fout.close()

    def set_surf_lattice(self,hkl,Va,Vb,nd=1,Vr=None):
        """
        create the surface cell by setting the surface basis vectors
        
        Arguments:
        ----------
        * hkl: surface indicies (in the bulk unit cell indexing)
        * Va,Vb: in-plane lattice vectors (defined in bulk basis)
        * nd: defines the "thickness" of the surface unit cell as a 
              multiple of d_hkl, ie |c_s| = nd*d_hkl
        * Vr: surface slab repeat vector (defined in bulk basis)
              If None Vr is generated

        Notes
        -----
        This calculates Vc given hkl and n.  It also verifies Va and Vb are in-plane
        and right handed.  
        """
        blat = self.bulk_uc.lattice
        self.hkl = hkl
        self.nd  = int(abs(nd))
        # compute possible vectors
        self._surface_vectors()
        # Vc is surface normal direction
        Vc = self.nd*blat.dvec(self.hkl)
        # make sure Va and Vb are in-plane and right handed
        check = blat.cross(Va,Vb)
        if blat.angle(Vc,check) > 0.001:
            print("Va and Vb do not define a right handed surface system")
            return
        # Vr is repeat with smallest angle
        if Vr is None:
            Vr = self.Vr_lst[0,:3]
        self.Va = Va
        self.Vb = Vb
        self.Vc = Vc
        self.Vr = Vr
        self._calc_transform()
        self.calc_surface_cell(term=self.term)

    def find_surf_lattice(self,hkl=[0,0,1],nd=1):
        """
        find a surface unit cell

        Parameters:
        -----------
        * hkl: surface indicies (in the bulk unit cell indexing)
        * nd: defines the "thickness" of the surface unit cell as a 
              multiple of d_hkl, ie |c_s| = nd*d_hkl
        """
        blat = self.bulk_uc.lattice
        self.hkl = hkl
        self.nd  = int(abs(nd))
        # compute possible vectors
        self._surface_vectors()
        # Vc is surface normal direction
        Vc = self.nd*blat.dvec(self.hkl)
        Vc = num.around(Vc,decimals=5)
        # Va is smallest in-plane
        Va = self.Vs_lst[0,:3]
        # Vb is next smallest in-plane (non-colinear with Va)
        # that makes a right handed system
        for j in range(1,len(self.Vs_lst)):
            ang = blat.angle(self.Vs_lst[j,:3], Va)
            if  (ang > 0.) and (ang < 180.):
                Vb = self.Vs_lst[j,:3]
                # if right handed system
                # the angle btwn (Va x Vb) and Vc
                # should be zero
                check = blat.cross(Va,Vb)
                if blat.angle(Vc,check) < 0.001:
                    break
        # Vr is repeat with smallest angle
        Vr = self.Vr_lst[0,:3]
        self.Va = Va
        self.Vb = Vb
        self.Vc = Vc
        self.Vr = Vr
        self._calc_transform()
        self.calc_surface_cell(term=self.term)

    def _surface_vectors(self):
        """
        Calculate in-plane lattice vectors(Vs), and repeat vectors(Vr)

        Generates:
        ----------
        self.Vs_lst:  The first three column's of the output give coefficients of the
                      in plane vectors. The fourth is the vector length
        
        self.Vr_lst: The first three column's of the output give coefficients of the
                     repeat vectors. The fourth is the magnitude, fifth is the plane
                     below the surface at which it terminates, and sixth is the
                     angle of the vector relative to the surface normal
        
        All the vectors are sorted according to their magnitudes given 
        in column 4 of the output. 
        """
        lat = self.bulk_uc.lattice
        hkl = self.hkl
        nd  = self.nd
        # see if there is a transform for primitive...
        if self.bulk_trns is not None:
            Va = self.bulk_trns[0,:]
            Vb = self.bulk_trns[1,:]
            Vc = self.bulk_trns[2,:]
            trns = lattice.LatticeTransform(lat,Va=Va,Vb=Vb,Vc=Vc)
            lat  = trns.plat()
            hkl  = trns.hp(hkl)
        # calculate vector d in bulk real space basis
        # normal to hkl plane, magnitude = dspacing
        d  = lat.dvec(hkl)
        # find in-plane vectors and repeat vectors
        # note the search range is controlled by vrange below 
        Vs_tmp=[]; Vr_tmp=[]
        #vrange = list(range(-4,5))
        vrange = list(range(-3*nd,3*nd+1))
        for n1 in vrange:
            for n2 in vrange:
                for n3 in vrange:
                    temp = n1*hkl[0] + n2*hkl[1] + n3*hkl[2]
                    # if temp is zero the vector lies in the hkl plane
                    # that passes through the unit cell origin
                    if temp == 0:
                        v_mag = lat.mag([n1, n2, n3])
                        if v_mag > 0.:
                            Vs_tmp.append([n1, n2, n3, v_mag])
                    # if temp == -nd then it terminates on the nd'th
                    # parallel hkl plane below the surface
                    elif temp == -1*nd:
                        v_mag = lat.mag([n1, n2, n3])
                        v_ang = lat.angle([n1, n2, n3],-nd*d)
                        Vr_tmp.append([n1, n2, n3, v_mag, temp, v_ang])
        # sort Vs_tmp according to magnitude
        Vs_tmp = num.array(Vs_tmp,dtype=float)
        idx = num.argsort(Vs_tmp[:,3])
        Vs = []
        for j in range(0,len(idx)):
            x = Vs_tmp[idx[j],:]
            Vs.append(x)
        Vs = num.array(Vs)
        # sort Vr_tmp according to angle they make with -nd*d
        Vr_tmp = num.array(Vr_tmp,dtype=float)
        idx = num.argsort(Vr_tmp[:,5])
        Vr = []
        for j in range(0,len(idx)):
            x = Vr_tmp[idx[j],:]
            Vr.append(x)
        Vr = num.array(Vr)
        # if there was a transform, go back to the original system
        if self.bulk_trns is not None:
            for j in range(len(Vs)):
                Vs[j,:3] = trns.v(Vs[j,:3])
            for j in range(len(Vr)):
                Vr[j,:3] = trns.v(Vr[j,:3])
        # done
        self.Vs_lst = Vs
        self.Vr_lst = Vr

    def _calc_transform(self):
        """
        get transformation matricies for bulk-to-surface (and visa-versa)
        and compute surface lattice params, and Vr in surf coords
        """
        bulk = self.bulk_uc
        Va   = self.Va
        Vb   = self.Vb
        Vc   = self.Vc
        self.transform = lattice.LatticeTransform(bulk.lattice,Va=Va,Vb=Vb,Vc=Vc)
        self.lattice   = self.transform.plat()
        Vr_s           = self.transform.vp(self.Vr)
        self.Vr_s      = num.around(Vr_s, decimals = 6)

    def calc_surface_cell(self,term=0):
        """
        compute coordinates of bulk and termination unit cell
        in surface coordinates
        """
        self._p1_bulk()
        self._p1_term(term=term)

    def update_term(self,term=0):
        """
        recompute the termination unit cell
        """
        self._p1_term(term=term)

    def _p1_bulk(self):
        """
        get bulk P1 cell (in surface coordinates)
        """
        # note expand the bulk so we're sure to fill the surface cell
        Va   = self.Va;  Vb   = self.Vb;  Vc   = self.Vc
        na = int(num.ceil(num.max(num.fabs([Va[0],Vb[0],Vc[0]])))) + 2
        nb = int(num.ceil(num.max(num.fabs([Va[1],Vb[1],Vc[1]])))) + 2
        nc = int(num.ceil(num.max(num.fabs([Va[2],Vb[2],Vc[2]])))) + 2
        batoms = self.bulk_uc.atom_list(cartesian=False,na=na,nb=nb,nc=nc,update_labels=False)
        # get P1 surface unit cell (coords and Uaniso transformed to surf coords)
        labels = []; atsym  = []; coords = []
        occ = []; ox = []; Uiso = []; Uaniso = []
        for j in range(batoms.natoms):
            vs = self.transform.vp(batoms.coords[j])
            vs = num.around(vs, decimals = 5)
            if (vs[0] >= 0) and (vs[0] < 1.) and \
               (vs[1] >= 0) and (vs[1] < 1.) and \
               (vs[2] >= 0) and (vs[2] < 1.):
                U = num.zeros((3,3))
                labels.append(batoms.labels[j])
                atsym.append(batoms.atsym[j])
                coords.append(vs)
                occ.append(batoms.occ[j])
                ox.append(batoms.ox[j])
                Uiso.append(batoms.Uiso[j])
                U[0,0] = batoms.Uaniso[j,0]; U[0,1] = batoms.Uaniso[j,1]
                U[0,2] = batoms.Uaniso[j,2]; U[1,1] = batoms.Uaniso[j,3]
                U[1,2] = batoms.Uaniso[j,4]; U[2,2] = batoms.Uaniso[j,5]
                U = num.dot(num.dot(self.transform.M, U), self.transform.G)
                U = num.concatenate( (U[0,:],U[1,1:],[U[2,2]]) )
                Uaniso.append(U)
        # assign P1 cell to atom_list
        self.p1bulk = AtomList(labels=labels,atsym=atsym,coords=coords,
                               lattice=self.lattice,occ=occ,ox=ox,
                               Uiso=Uiso,Uaniso=Uaniso)
        # Add an index to the atom labels. start counting atoms at bottom...
        self.p1bulk.update_labels(sep=":",pre='b',ascend=True,replace=False)

    def _p1_term(self,term):
        """
        get termination P1 cell (in surface coordinates)
        """
        term = int(term)
        self.term = term
        # make sure bulk coords are in ascending order (should be already).  
        self.p1bulk.sort(ascend=True)
        # find unique atomic layers
        layer_z = []; layer_idx = []
        for j in range(self.p1bulk.natoms):
            if self.p1bulk.coords[j][2] not in layer_z:
                layer_z.append(self.p1bulk.coords[j][2])
                layer_idx.append([j])
            else:
                idx = layer_z.index(self.p1bulk.coords[j][2])
                layer_idx[idx].append(j)
        # figure number of sites in new cell
        nlayer  = len(layer_z) + term
        if nlayer < 1:
            print("Error: term value '%i' is greater than the number of layers '%i'" % (term,len(layer_z)))
            print("Setting term value to '0'")
            term = 0; self.term = term
            nlayer = len(layer_z)
        nsite = 0; idx = 0
        for j in range(nlayer):
            if idx == len(layer_idx): idx=0
            nsite = nsite + len(layer_idx[idx])
            idx += 1
        # build up new cell, applying repeat vector (in positive/surface direction)
        labels = []; atsym  = []
        coords  = num.zeros((nsite,3))
        occ     = num.ones((nsite))
        ox      = 99*num.ones(nsite)
        Uiso    = num.zeros(nsite)
        Uaniso  = num.zeros((nsite,6))
        #
        Vr = -1.*self.Vr_s
        idx1=0; idx2=0; n=1.
        for j in range(nlayer):
            if idx1 == len(layer_idx): idx1=0
            for i in layer_idx[idx1]:
                labels.append(self.p1bulk.labels[i])
                atsym.append(self.p1bulk.atsym[i])
                coords[idx2,:] = self.p1bulk.coords[i,:] + n*Vr
                occ[idx2]      = self.p1bulk.occ[i]
                ox[idx2]       = self.p1bulk.ox[i]
                Uiso[idx2]     = self.p1bulk.Uiso[i]
                Uaniso[idx2,:] = self.p1bulk.Uaniso[i,:]
                idx2 += 1
                n = ( idx2 // self.p1bulk.natoms ) + 1 
            idx1 += 1
        # check x,y in cell bounds
        for j in range(len(coords[:,0])):
            if coords[j][0] >= 1.0 or coords[j][0] < 0.0:
                coords[j][0] = _reduce_frac(coords[j][0])
            if coords[j][1] >= 1.0 or coords[j][1] < 0.0:
                coords[j][1] = _reduce_frac(coords[j][1])
        # assign P1 cell to AtomList
        self.p1term = AtomList(labels=labels,atsym=atsym,coords=coords,
                                 lattice=self.lattice,occ=occ,ox=ox,
                                 Uiso=Uiso,Uaniso=Uaniso)
        # Add an index to the atom labels. start counting atoms at bottom...
        self.p1term.update_labels(sep=":",pre='t',ascend=True,replace=True)

    def atom_list(self,cartesian=False,na=1,nb=1,nbulk=1,term=-99):
        """
        output listing of atom coordinates 
        (in surface coordinate system)

        Parameters:
        -----------
        * cartesian: if True output in cartesian, otherwise in fractional coordinates
        * na, nb: number of cell repeats in the in-plane (a,b) directions
        * nbulk: number of repeats of the unit cell in the "bulk" direction (i.e. -c)
                 if nbulk=0, then the model will only include the "termination-layer"
                 with no unit cell repeats in the "bulk" direction
        * term: atomic layer that defines the surface termination.  If None
                then the model will only consist of the "bulk-surface-model"
                if term=-99 use the current value of term as already specified in the surf object
                if term=0 then the model terminates at the top of the surface unit cell
                if term=+1 then one atomic layer is added on top
                if term=-1 then one atomic layer is removed from the top 
                etc.
        * long_fmt: output in long format

        Output:
        -------
        * AtomList object

        Notes:
        ------
        * na, nb, nbulk should all be positive integers
        * na and nb defines the inplane extent of the slab
        * nbulk defines the thickness of the "bulk" part of the slab
        * term, if not None, defines which atomic layer is the termination 
         (and therefore defines how "thick" the terminating layer of the slab is)
         if term is not -99, then surf will be updated with the new term value
        * Note if nbulk=0 and term=None this function returns None (ie no model!)
        * coordinates are output in descending order with respect to z
        """
        # check there is something to return
        if (nbulk==0) and (term is None): 
            print("Warning: nbulk is zero and term is None in surface.atoms_list")
            return None
        ## get the expanded bulk unit cell
        # these should be positive integers
        na = int(abs(na)); nb = int(abs(nb)); nbulk = int(abs(nbulk))
        if nbulk <= 0:
            batoms = None
        elif nbulk == 1:
            batoms = self.p1bulk
        elif nbulk > 1:
            # expand bulk cell in the bulk direction
            rng = list(range(0,nbulk))
            blk = self.p1bulk
            (labels,atsym,frac,occ,ox,Uiso,Uaniso) = _expand_(blk.labels, blk.atsym, blk.coords, 
                                                              rng, Vr=self.Vr_s, occ=blk.occ, 
                                                              ox=blk.ox, Uiso=blk.Uiso, 
                                                              Uaniso=blk.Uaniso)
            for j in range(len(labels)):
                if frac[j][0] >= 1.0 or frac[j][0] < 0.0:
                    frac[j][0] = _reduce_frac(frac[j][0])
                if frac[j][1] >= 1.0 or frac[j][1] < 0.0:
                    frac[j][1] = _reduce_frac(frac[j][1])
            batoms = AtomList(labels=labels,atsym=atsym,coords=frac,lattice=self.lattice,
                              occ=occ,ox=ox,Uiso=Uiso,Uaniso=Uaniso)
        # check the termination
        if term is not None:
            if term != -99: self._p1_term(term=term)
            tatoms = self.p1term
        else:
            tatoms = None
        # make combo AtomList
        if batoms is None:
            atlist = copy.copy(tatoms)
        elif tatoms is None:
            atlist = copy.copy(batoms)
        else:
            atlist = merge_alist(tatoms,batoms)
        # expand in x-y
        if na>1 or nb>1:
            atlist = _expand_frac(atlist.labels, atlist.atsym, atlist.coords, 
                                  na=na, nb=nb, nc=1, occ=atlist.occ, ox=atlist.ox, 
                                  Uiso=atlist.Uiso, Uaniso=atlist.Uaniso, atom_list=True)
            atlist.lattice = self.lattice
        atlist.sort(ascend=False)
        # transform to cartesian
        if cartesian == True:
            return atlist.cartesian()
        else:
            return atlist

##########################################################################


