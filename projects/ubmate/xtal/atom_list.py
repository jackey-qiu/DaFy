"""
Atom List

Authors/Modifications:
-----------------------
* Tom Trainor (tptrainor@alaska.edu)


Notes:
------
* Store and manipulate lists of atoms, coords etc... 

"""
##########################################################################
import numpy as num
import copy
from xtal import lattice

##########################################################################
def merge_alist(atlst1, atlst2):
    """
    Merge 2 AtomList instances
    """
    natoms  = atlst1.natoms + atlst2.natoms
    labels  = []
    atsym   = []
    coords  = num.zeros((natoms,3))
    occ     = num.zeros(natoms)
    ox      = num.zeros(natoms)
    Uiso    = num.zeros(natoms)
    Uaniso  = num.zeros((natoms,6))
    #
    if atlst1.lattice == atlst2.lattice:
        latt = atlst1.lattice
    else:
        print("Warning, atom list lattices are not equivalent")
        latt = None
    # 
    def _updt(atlst,st):
        k = st
        for j in range(atlst.natoms):
            labels.append(atlst.labels[j])
            atsym.append(atlst.atsym[j])
            coords[k,:]  = atlst.coords[j,:]
            occ[k]       = atlst.occ[j]
            ox[k]        = atlst.ox[j]
            Uiso[k]      = atlst.Uiso[j]
            Uaniso[k,:]  = atlst.Uaniso[j,:]
            k = k+1
    _updt(atlst1,0)
    _updt(atlst2,atlst1.natoms)
    return AtomList(labels=labels,atsym=atsym,coords=coords,lattice=latt,occ=occ,ox=ox,Uiso=Uiso,Uaniso=Uaniso)

##########################################################################
class AtomList:
    """
    List of atoms in a unit cell and associated parameters
    This is used when generating a P1 listing of atoms
    """
    def __init__(self,labels=None,atsym=None,coords=None,lattice=None,occ=None,ox=None,Uiso=None,Uaniso=None):
        """
        * labels: array of site labels
        * atsym:  array of atomic symbols
        * coords: array of coordinates
        * lattice: lattice instance (use None for cartesian)
        * occ: array of occupancies  (default to ones)
        * ox: array of oxidation states (default to 99)
        * Uiso: array of isotropic displacement factors  (default to zeros)
        * Uiso: array of anisotropic dispacement factors 
                [[U11, U12, U13, U22, U23, U33]...],  default to zeros
        """
        self.verbose = True
        self._description = "AtomList"
        if labels is None:
            self.natoms = 0
            self.labels  = num.array([],dtype='U')
            self.atsym   = num.array([],dtype='U')
            self.coords  = num.array([],dtype='double')
            self.occ     = num.array([],dtype='double')
            self.ox      = num.array([],dtype='double')
            self.Uiso    = num.array([],dtype='double')
            self.Uaniso  = num.array([],dtype='double')
        else:
            self.natoms  = len(labels)
            self.labels  = num.array(labels)
            self.atsym   = num.array(atsym)
            self.coords  = num.array(coords)
            if occ is None: self.occ = num.ones(self.natoms)
            else: self.occ = num.array(occ)
            if ox is None: self.ox  = 99*num.ones(self.natoms)
            else: self.ox = num.array(ox)
            if Uiso is None: self.Uiso  = num.zeros(self.natoms)
            else: self.Uiso = num.array(Uiso)
            if Uaniso is None: self.Uaniso  = num.zeros((self.natoms,6))
            else: self.Uaniso = num.array(Uaniso)
        self.lattice = lattice

    def __repr__(self,):
        return self._write(long_fmt=self.verbose)

    def _write(self,long_fmt=True,header=True):
        """
        generate a string for output
        """
        if header==True:
            if self.lattice is not None:
                out =  "%6.5f  %6.5f  %6.5f  %6.5f  %6.5f  %6.5f \n" % (self.lattice.a, self.lattice.b,
                                                                        self.lattice.c, self.lattice.alpha,
                                                                        self.lattice.beta, self.lattice.gamma)
            else:
                out = ""
            lbls = "# sym    x        y         z"
            if long_fmt == True:
                lbls = lbls + "          label    occ    ox       Uiso       U11       U12        U13       U22       U23       U33"
            lbls = lbls + "\n"
            out = out + lbls
        else:
            out = ""
        for j in range(self.natoms):
            out = out + "%2s  %8.5f  %8.5f  %8.5f" % (self.atsym[j], self.coords[j,0], self.coords[j,1], self.coords[j,2])
            if long_fmt == True:
                out = out + "   %10s   %3.3f   %+3.1f   %8.5f"  % (self.labels[j], self.occ[j], self.ox[j], self.Uiso[j])
                out = out + "   %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f"  % (self.Uaniso[j,0], self.Uaniso[j,1],
                                                                              self.Uaniso[j,2], self.Uaniso[j,3],
                                                                              self.Uaniso[j,4], self.Uaniso[j,5])
            out = out + "\n"
        return out

    def show(self, long_fmt=True, header=True):
        """
        display to std out

        Arguments:
        ----------
        * long_fmt: if True use long format
        * header: include lattice params and column labels as a header
        """
        print(self._write(long_fmt=long_fmt,header=header))

    def write_xyz(self,fname="atomlist.xyz",long_fmt=True,header=True):
        """
        write to a file

        Arguments:
        ----------
        * fname: output file
        * long_fmt: if True use long format
        * header: include lattice params and column labels as a header
        """
        fout = open(fname,'w')
        fout.write("%i\n" % self.natoms)
        fout.write(self._write(long_fmt=long_fmt,header=header))
        fout.close()

    def add_atom(self,label,atsym,coord,occ=1.,ox=99,Uiso=0,Uaniso=[0.,0.,0.,0.,0.,0.]):
        """
        * label: site label string
        * atsym:  atomic symbol string
        * coord: list or numpy array of coordinates
        * occ:  occupancy  (default to one)
        * ox: oxidation state (default to 99)
        * Uiso: isotropic displacement factor  (default to zero)
        * Uiso: array or list of anisotropic dispacement factors 
                [U11, U12, U13, U22, U23, U33],  default to zeros
        """
        self.labels = num.concatenate( (self.labels, [label]) )
        self.natoms = len(self.labels)
        self.atsym  = num.concatenate( (self.atsym, [atsym]) )
        self.coords = num.concatenate( (self.coords.ravel(), num.array(coord) ))
        self.coords.shape = (self.natoms,3)
        self.occ    = num.concatenate( (self.occ, [occ]) )
        self.ox     = num.concatenate( (self.ox, [ox]) )
        self.Uiso   = num.concatenate( (self.Uiso, [Uiso]) )
        self.Uaniso = num.concatenate( (self.Uaniso.ravel(), num.array(Uaniso)) )
        self.Uaniso.shape = (self.natoms,6)

    def index(self,label,first=False):
        """
        Get the index of an atom

        Arguments:
        ----------
        * label: string label of atom
        * first: if True only return index of first instance

        Returns:
        --------
        * list containing indicies of atoms matching label
        """
        index = []
        for j in range(self.natoms): 
            if self.labels[j] == label:
                index.append(j)
                if first == True:
                    return index
        return index

    def sort(self,ascend=True):
        """
        sort cell contents by x then z

        Arguments:
        ----------
        * ascend: if True sort in ascending order
                  if False sort descending order

        Notes:
        ------
        The default sort is low to high (ascending order), first sorting
        by the x-coordinate, then by the z-coordinate.  If ascend = False 
        the sort order is high to low (descending order) again by x, then z.
        """
        def _srt(c):
            idx = num.argsort(self.coords[:,c],kind='mergesort')
            # note [::-1] reverses the array order
            if ascend == False:
                self.coords = self.coords[idx[::-1],:]
                self.labels = self.labels[idx[::-1]]
                self.atsym  = self.atsym[idx[::-1]]
                self.occ     = self.occ[idx[::-1]]
                self.ox      = self.ox[idx[::-1]]
                self.Uiso    = self.Uiso[idx[::-1]]
                self.Uaniso  = self.Uaniso[idx[::-1],:]
            else:
                self.coords = self.coords[idx,:]
                self.labels = self.labels[idx]
                self.atsym  = self.atsym[idx]
                self.occ     = self.occ[idx]
                self.ox      = self.ox[idx]
                self.Uiso    = self.Uiso[idx]
                self.Uaniso  = self.Uaniso[idx,:]
        _srt(0)
        _srt(2)

    def update_labels(self,sep=":",pre='',ascend=True,replace=False):
        """
        append a suffix to the labels with the atom index
        counting in order given by ascend

        Arguments:
        ----------
        * sep: seperator between label and index
        * pre: string to prepend to the sfx
        * ascend: if True sort in ascending order
                  if False sort descending order
        * replace: if True this will check each label to see if 
                   there was already an existing suffix, if so will 
                   replace it
        Notes:
        -----
        * the new labels will be of the form:  old_label+sep+pre+index
        * this leaves the atom list sorted according to ascend arg
        * you only need the replace arg if this method was called, 
          then you want to replace the suffix.  Note that 'sep' 
          needs to be the same in each call
        """
        self.sort(ascend=ascend)
        labels = list(self.labels)
        for j in range(self.natoms):
            lbl = labels[j]
            if replace==True:
                ii = lbl.rfind(sep)
                if ii > -1: lbl = lbl[0:ii]
            labels[j] = lbl + sep + pre + "%i" % (j+1)
        self.labels = num.array(labels)

    def cartesian(self):
        """
        return a copy of the atom list transformed to cartesian coordinates
        """
        if self.lattice is None: return None
        atom_list = AtomList()
        trns = lattice.LatticeTransform(self.lattice)
        trns.cartesian()
        for j in range(self.natoms):
            cart = trns.vp(self.coords[j,:])
            # Uaniso (see above about line 235)
            Uaniso = num.zeros((3,3))
            Uaniso[0,0] = self.Uaniso[j,0]; Uaniso[0,1] = self.Uaniso[j,1]
            Uaniso[0,2] = self.Uaniso[j,2]; Uaniso[1,1] = self.Uaniso[j,3]
            Uaniso[1,2] = self.Uaniso[j,4]; Uaniso[2,2] = self.Uaniso[j,5]
            Uaniso = num.dot(num.dot(trns.M, Uaniso), trns.G)
            Uaniso = num.concatenate((Uaniso[0,:],Uaniso[1,1:],[Uaniso[2,2]]))
            atom_list.add_atom(self.labels[j],self.atsym[j],cart,
                               occ=self.occ[j],ox=self.ox[j],
                               Uiso=self.Uiso[j],Uaniso=Uaniso)
        return atom_list

##########################################################################
def _reduce_frac(x):
    """
    Reduce a fractional coordinate so its btwn 0 and 1

    note could probably improve performance using divmod 
    instead of the while loop
    """
    while 1:
        if x >= 1.0:
            x = x-1.0
        if x < 0.0:
            x = x + 1.0
        #if num.fabs(x) < 1.0:
        if (x >= 0.) and (x < 1.0): 
            return x

def _rng(n):
    """
    Given a positive integer n, get a list of integers covering the range,
    inclusive of zero, that specify the cell repeats.  E.g 
      _rng(0) = [0]
      _rng(1) = [0]
      _rng(2) = [0, 1]
      _rng(3) = [-1, 0, 1]
    """
    n = int(abs(n))
    if n % 2 == 0:
        en = int(n/2 + 1)
        st = -1*int(n/2 - 1)
    else: 
        en = int((n-1)/2 + 1)
        st = -1*int((n-1)/2)
    rng = [j for j in range(st,en)]
    if len(rng) == 0: rng = [0]
    return rng

def _expand_frac(labels,atsym,frac,na=1,nb=1,nc=1,occ=None,ox=None,Uiso=None,Uaniso=None,atom_list=True):
    """
    Expand a set of fractional coordinates
    
    Parameters:
    -----------
    * labels: a list or array of site labels
    * atsym: a list or array of atomic symbols
    * frac: a numpy array of fractional coordinates (all values should be
            between 0 and < 1.)
    * na,nb,nc: number of cell repeats in each direction (positive integers)
    * occ: array or list of occupancies
    * ox: array or list of oxidation states
    * Uiso: array or list of isotropic displacement factors
    * Uaniso: array or list of anisotropic dispacement factors [[U11, U12, U13, U22, U23, U33]...]
    * atom_list: if this flag is True an AtomList object is returned.  otherwise we return a tuple
      of lists
      
    Returns:
    --------
    * AtomList instance

    Notes:
    ------
    * na,nb,nc should be positive integers.  
      The _rng fcn generates a list of integers, inclusive of zero, that
      specify the cell repeats in each direction.  E.g na=3, nb=2, nc=1
          xr = [-1, 0, 1], yr = [-1, 0], zr = [0]
      will generate 3x2x1 expansion of the input fractional coords
    * The initial frac coords will be placed at the top of the coord array, len(rng)
      followed by expansion in x, then y, then z 
      (ie. results are not sorted, but coords[0:nsite,:] are the initial 
      set of fractional coordinates --> the (0,0,0) cell).
    * If any of occ, ox, Uiso, Uaniso are None, default arrays will be generated 
    """
    na = int(abs(na)); nb = int(abs(nb)); nc = int(abs(nc))
    check = False
    if na==1 and nb==1 and nc==1: check = True
    if na > 1 or check == True:
        (labels,atsym,frac,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,frac,_rng(na),Vr=[1.,0,0],occ=occ,
                                                          ox=ox,Uiso=Uiso,Uaniso=Uaniso)
    if nb > 1:
        (labels,atsym,frac,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,frac,_rng(nb),Vr=[0,1.,0],occ=occ,
                                                          ox=ox,Uiso=Uiso,Uaniso=Uaniso)
    if nc > 1:
        (labels,atsym,frac,occ,ox,Uiso,Uaniso) = _expand_(labels,atsym,frac,_rng(nc),Vr=[0,0,1.],occ=occ,
                                                          ox=ox,Uiso=Uiso,Uaniso=Uaniso)
    if atom_list == True:
        return AtomList(labels=labels,atsym=atsym,coords=frac,occ=occ,ox=ox,Uiso=Uiso,Uaniso=Uaniso)
    else:
        return (labels,atsym,frac,occ,ox,Uiso,Uaniso)

def _expand_(labels,atsym,frac,rng,Vr=[0,0,1],occ=None,ox=None,Uiso=None,Uaniso=None):
    """
    Expand a set of fractional coordinates
    
    Parameters:
    -----------
    * labels: a list or array of site labels
    * atsym: a list or array of atomic symbols
    * frac: a numpy array of fractional coordinates (all values should be
            between 0 and < 1.)
    * rng: array of cell repeats (integers inclusive of zero --> the origin cell)
    * Vr expansion vector
    * occ: array or list of occupancies
    * ox: array or list of oxidation states
    * Uiso: array or list of isotropic displacement factors
    * Uaniso: array or list of anisotropic dispacement factors [[U11, U12, U13, U22, U23, U33]...]
      
    Returns:
    --------
    * a tuple of lists/arrays

    Notes:
    ------
    * If any of occ, ox, Uiso, Uaniso are None, default arrays will be generated 
    """
    nsite   = len(labels)
    ncell   = len(rng)
    #
    labels  = list(copy.copy(labels))
    atsym   = list(copy.copy(atsym))
    frac    = num.array(frac)
    #
    coords     = num.zeros((nsite*ncell,3))
    occ_ex     = num.ones(nsite*ncell)
    ox_ex      = 99*num.ones(nsite*ncell)
    Uiso_ex    = num.zeros(nsite*ncell)
    Uaniso_ex  = num.zeros((nsite*ncell,6))
    #
    coords[0:nsite,:] = frac
    if occ is not None: occ_ex[0:nsite] = num.array(occ)[:]
    if ox is not None: ox_ex[0:nsite] = num.array(ox)[:]
    if Uiso is not None: Uiso_ex[0:nsite] = num.array(Uiso)[:]
    if Uaniso is not None: Uaniso_ex[0:nsite,:] = num.array(Uaniso)[:,:]
    #
    lbls  = copy.copy(labels)
    at    = copy.copy(atsym)
    Vr    = num.array(Vr,dtype='float')
    shift = num.zeros((nsite,3))
    n     = 1
    for j in rng:
        if j != 0:
            st = (n)*nsite
            en = (n)*nsite + nsite
            #
            shift              = j*Vr
            coords[st:en,:]    = frac + shift
            #
            occ_ex[st:en]      = occ_ex[0:nsite]
            ox_ex[st:en]       = ox_ex[0:nsite]
            Uiso_ex[st:en]     = Uiso_ex[0:nsite]
            Uaniso_ex[st:en,:] = Uaniso_ex[0:nsite,:]
            #
            labels.extend(lbls)
            atsym.extend(at)
            n += 1
    #return
    labels = num.array(labels)
    atsym  = num.array(atsym)
    return (labels,atsym,coords,occ_ex,ox_ex,Uiso_ex,Uaniso_ex)

##########################################################################
##########################################################################
if __name__ == "__main__":
    print(_rng(0))
    print(_rng(-1))
    print(_rng(-2))
    print(_rng(-3))


