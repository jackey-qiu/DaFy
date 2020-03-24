'''<h1>Library for surface x-ray diffraction simulations</h1>
<p> The problem of modelling the sample is divided to four
different classes: Sample, Slab, UnitCell and Instrument.
A Slab is the basic unit that builds up a sample and can
be seen as a quasi-unitcell for the sxrd problem.
Stricitly it is a 2D unitcell with a finite extension
out-of-plane. The Sample is then built from these Slabs one slab for
the bulk and a list of slabs for the surface strucutre.

<p> The unitcell consists of parameters for  the unitcell and the
instrument contains instrument variables. See below for a full list.

<h2>Classes</h2>
<h3>Slab</h3>
<code> Slab(c = 1.0, slab_oc = 1.0)</code><br>
    <dl>
    <dt><code><b>c</b></code></dt>
    <dd> A scale factor for ou-of-plane extension of the
    Slab. All z-positions will be scaled with this factor.</dd>
    <dt><code><b>slab_oc</b></code></dt>
    <dd> A global scaling of the occupancy of all atoms in the
    slab.</dd>
    </dl>
<code> [Slab].add_atom(id, el, x, y, z, u = 0, oc = 1.0, m = 1.0)</code><br>
    <dl>
    <dt><code><b>id</b></code></dt>
    <dd>A unique string identifier </dd>
    <dt><code><b>el</b></code></dt>
    <dd>The element described in a string. Note that
    ions is denoted as "Sr2p" and "O2m" where 2 is the oxidation number and
    p and m denoted plus and minus charge.</dd>
    <dt><code><b>x</b></code></dt>
    <dd> The x-position in Slab unit cell coords (same as given by the
    UnitCell)</dd>
    <dt><code><b>y</b></code></dt>
    <dd> The y-position in Slab unit cell coords (same as given by the
    UnitCell)</dd>
    <dt><code><b>z</b></code></dt>
    <dd> The z-position in Slab unit cell coords (The Unitcell c scaled by
    a factor of the c-value for the slab)</dd>
    <dt><code><b>u</b></code></dt>
    <dd> The mean-square displacement for the atom</dd>
    <dt><code><b>oc</b></code></dt>
    <dd> The occupancy of the atom</dd>
    <dt><code><b>m</b></code></dt>
    <dd> The multiplicity of the site, defined as in the international tables
    of crystallogrphy. Note that it is plane goups and NOT space groups that
    will produce valid results.</dd>
    </dl>
<code> [Slab].copy()</code><br>
    Creates a copy of object [Slab]. This decouples the new object
    returned by copy from the original [Slab].
<code> [Slab].find_atoms(expression)</code><br>
    Function to locate atoms in a slab in order to connect parameters
    between them. Returns an AtomGroup.
    <dl>
    <dt><code><b>expression</b></code></dt>
    <dd> Either a list of the same length as the number of atoms or
    a string that will evaluate to true or false for each atom.
    Allowed variables are: <code>x, y, z, id, el, u, ov, m,/code></dd>
    </dl>
<code> [Slab].all_atoms()</code><br>
    Yields all atoms inside a slab as an AtomGroup.
    Returns an AtomGroup.
<code> [Slab][id]</code><br>
    Locates atom that has id <code>id</code>. Returns an AtomGroup
    <dl>
    <dt><code><b>id</b></code></dt>
    <dd>Uniqe string identifer for one atom </dd>
    </dl>
<h3>Sample</h3>
<code> Sample(inst, bulk_slab, slabs, unit_cell, surface_sym = [],
bulk_sym = []) </code><br>
    <dl>
    <dt><code><b>inst</b></code></dt>
    <dd> Instrument object for the sample
    </dd>
    <dt><code><b>bulk_slab</b></code></dt>
    <dd>The Slab that describes the bulk strucutre
    </dd>
    <dt><code><b>slabs</b></code></dt>
    <dd>A list ([]) of slabs for the surface structure
    </dd>
    <dt><code><b>unit_cell</b></code></dt>
    <dd>A UnitCell object
    </dd>
    <dt><code><b>surface_sym</b></code></dt>
    <dd>A list ([]) of SymTrans objects describing the surface symmetry.
    Default value - an empty list will implement a p1 symmetry, that is no
    symmetry operations at all.
    </dd>
    <dt><code><b>bulk_sym</b></code></dt>
    <dd>A list ([]) of SymTrans objects describing the bulk symmetry.
    Default value - an empty list will implement a p1 symmetry, that is
    no symetry operations at all.
    </dd>
    </dl>
<code>[Sample].calc_f(h, k, l)</code><br>
Calculates the total structure factor (complex number) from the
the surface and bulk strucutre. Returns an array of the same size
as h, k, l. (h, k, l should be of the same legth and is given in
coordinates of the reciprocal lattice as defnined by the uit_cell coords)
<code>[Sample].turbo_calc_f(h, k, l)</code><br>
A faster version of <code>calc_f</code> which uses inline c code to increase
the speed. Can be more unstable than <code>calc_f</code> use on your own risk.
<code>[Sample].calc_rhos(x, y, z, sb)</code><br>
Calculate the the surface electron density of a model. The parameter sb is a Gaussian convolution factor given the width of the Gaussian in reciprocal space.
Used mainly for comparison with direct methods, i.e. DCAF.
NOTE that the transformation from the width of the window function given
in <code>dimes.py</code> is <code>sqrt(2)*pi*[]</code>
'''
#updated by Jackey Qiu 12/19/2011
"""
change log:
class Sample has been updated to consider domains in the frame of surface unit cell
    the argument of slabs is a library of domains in form of {'domain_name':{'slab':slab_class,'wt':0.1}}, and therefore
    the vertical stacking operation will be disable.
    argument 'surface_parms' is a library of delta1 and delta2 which are used to define surface unitcell, if no coordinate
    system change, just set them 0.
    argument 'coherence'is a flag for operation to add up struction factor for different domains, True means adding up
    in coherence, False means adding up incoherence
class Slab was updated as follows:
    dx was changed to three parmeter dx1,dx2 and dx3, the same thing to dy dz, it is changed in this way to fit into the operation
    for operations in AtomGroup, refer to AtomGroup part for detail. function _extract_value was changed accordingly
    argument T_factor is a switch to different interpretation of termal factor, which can be either 'u' or 'B'
    some bugs in function of del_atom was fixed, now it works well
AtomGroup was changed to consider moving atoms on symmetrical basis
    In the original version, dx/dy/dz shift set in AtomGoup will be set in the exactely the same way for the member atoms
    After considering symmetry operation, operation of dx shift in AtomGroup will make shift of dx1, dy1 and dz1 simultaneously
    dy shift will corresond to dx2,dy2 and dz2 (dz to dx3 dy3 and dz3)in the associated slab, that't why Slab has three set of
    dxdydz which has been customized in the slab class
    argument id_in_sym_file is a list of ids with their orders corresponding to their row orders appearing symmetry operation datafile
    use_sym is a switch to use symmetry or not
    filename is the file name of symmetry operations (txt file, data in form of n by 9)
    set_par is scaling parameters for code extention in future
    _set_func and _get_func was changed accordingly
"""
##in version3##
#the symmetry related domains are always added up together incoherencely. If the coherence is true, actually domainA's added up coherencely,
#domainB's added up coherencely.
##in version4##
#take away the argument of sym_file in the Atom_group, each time you add a new group member, you must specify the matrix list to define the symmetry relationship

import numpy as np
from .utils import f, rho
import time,os
import pickle,copy

try:
    from scipy import weave
    _turbo_sim = True
except:
    print('Info: Could not import weave, turbo off')
    _turb_sim = False

__pars__ = ['Sample', 'UnitCell', 'Slab', 'AtomGroup', 'Instrument']

class Sample:
    def __init__(self, inst, bulk_slab, slabs,unit_cell,surface_parms={'delta1':0,'delta2':0},
                 surface_sym = [], bulk_sym = [],coherence=True):
        self.set_bulk_slab(bulk_slab)
        #self.set_slabs(slabs)
        self.set_surface_sym(surface_sym)
        self.set_bulk_sym(bulk_sym)
        self.inst = inst
        self.set_unit_cell(unit_cell)
        self.delta1=surface_parms['delta1']
        self.delta2=surface_parms['delta2']
        self.domain=slabs
        self.coherence=coherence

    def set_bulk_slab(self, bulk_slab):
        '''Set the bulk unit cell to bulk_slab
        '''
        if type(bulk_slab) != type(Slab()):
            raise TypeError("The bulk slab has to be a member of class Slab")
        self.bulk_slab = bulk_slab

    def set_slabs(self, slabs):
        '''Set the slabs of the sample.

        slabs should be a list of objects from the class Slab
        '''
        if type(slabs) != type([]):
            raise TypeError("The surface slabs has to contained in a list")
        if min([type(slab) == type(Slab()) for slab in slabs]) == 0:
            raise TypeError("All members in the slabs list has to be a memeber of class Slab")
        self.slabs = slabs

    def set_surface_sym(self, sym_list):
        '''Sets the list of symmetry operations for the surface.

        sym_list has to be a list ([]) of symmetry elements from the
        class SymTrans
        '''
        # Type checking
        if type(sym_list) != type([]):
            raise TypeError("The surface symmetries has to contained in a list")

        if sym_list == []:
            sym_list = [SymTrans()]

        if min([type(sym) == type(SymTrans()) for sym in sym_list]) == 0:
            raise TypeError("All members in the symmetry list has to be a memeber of class SymTrans")

        self.surface_sym = sym_list

    def set_bulk_sym(self, sym_list):
        '''Sets the list of allowed symmetry operations for the bulk

        sym_list has to be a list ([]) of symmetry elements from the
        class SymTrans
        '''
        # Type checking
        if type(sym_list) != type([]):
            raise TypeError("The surface symmetries has to contained in a list")

        if sym_list == []:
            sym_list = [SymTrans()]

        if min([type(sym) == type(SymTrans()) for sym in sym_list]) == 0:
            raise TypeError("All members in the symmetry list has to be a memeber of class SymTrans")

        self.bulk_sym = sym_list

    def set_unit_cell(self, unit_cell):
        '''Sets the unitcell of the sample
        '''
        if type(unit_cell) != type(UnitCell(1.0, 1.0, 1.0)):
            raise TypeError("The bulk slab has to be a member of class UnitCell")
        if unit_cell == None:
            unit_cell = UnitCell(1.0, 1,.0, 1.0)
        self.unit_cell = unit_cell



    def calc_f(self, h, k, l):
        '''Calculate the structure factors for the sample
        '''
        #here the chemically equivalent domains will be added up in-coherently always
        ftot=0
        ftot_A=0
        ftot_B=0
        keys_domainA=[]
        keys_domainB=[]
        fb = self.calc_fb(h, k, l)
        for i in self.domain.keys():
            if "A" in i:keys_domainA.append(i)
            if "B" in i:keys_domainB.append(i)

        if self.coherence==True:
            for i in keys_domainA:
                if self.domain[i]['wt']!=0:
                    ftot_A=ftot_A+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    pass
            for i in keys_domainB:
                if self.domain[i]['wt']!=0:
                    ftot_B=ftot_B+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    pass
        else:
            for i in keys_domainA:
                if self.domain[i]['wt']!=0:
                    ftot_A=ftot_A+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    pass
            for i in keys_domainB:
                if self.domain[i]['wt']!=0:
                    ftot_B=ftot_B+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    pass
        ftot=abs(ftot_A)+abs(ftot_B)
        return abs(ftot)*self.inst.inten


    def calc_f2(self, h, k, l):
        #here incoherence means add up all domains in-coherently, and coherence means adding up all coherently
        ftot=0
        fb = self.calc_fb(h, k, l)
        if self.coherence==True:
            for i in self.domain.keys():
                ftot=ftot+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
        else:
            for i in self.domain.keys():
                ftot=ftot+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
        return abs(ftot)*self.inst.inten

    def calc_f3(self, h, k, l):
        #here self.coherence is a list of True (add up coherently) or False (add up in-coherently)
        ftot=0
        ftot_A_C, ftot_A_IC=0,0
        ftot_B_C, ftot_B_IC=0,0
        keys_domainA=[]
        keys_domainB=[]
        fb = self.calc_fb(h, k, l)
        for i in self.domain.keys():
            if "A" in i:keys_domainA.append(i)
            if "B" in i:keys_domainB.append(i)
        for i in keys_domainA:
            j=int(i[-2])-1
            if self.coherence[j]:
                ftot_A_C=ftot_A_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            else:
                ftot_A_IC=ftot_A_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
        for i in keys_domainB:
            j=int(i[-2])-1
            if self.coherence[j]:
                ftot_B_C=ftot_B_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            else:
                ftot_B_IC=ftot_B_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
        ftot=abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)

        return abs(ftot)*self.inst.inten

    def calc_f4(self, h, k, l):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs
        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten

    def calc_f4_test_coherence(self, h, k, l,coherence_symmetry=False):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs
        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            if coherence_symmetry:
                ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
            else:
                ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten

    def calc_f4_specular(self, h, k, l,raxr_el):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
            for i in keys_domainB:
                f_layered_water=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
            ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten


    def calc_f4_specular_test_coherence(self, h, k, l,raxr_el,coherence_symmetry=False):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
            for i in keys_domainB:
                f_layered_water=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
            if coherence_symmetry:
                ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
            else:
                ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
            #ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten

    def calculate_structure_factor(self,h,k,x,y,index=None,fit_mode='MD',height_offset=0,version=1):
        if x[0]<100:#CTR data
            return self.calc_f4_muscovite_CTR(h,k,x,height_offset,version)
        else:#RAXR data
            if fit_mode=='MI':
                return self.calc_f4_muscovite_RAXR_MI(h,k,x,y,index,height_offset,version)
            elif fit_mode=='MD':
                return self.calc_f4_muscovite_RAXR_MD(h,k,x,y,index,height_offset,version)

    def calc_f4_muscovite_CTR(self, h, k, l,height_offset=0,version=1):
        #now the coherence is either true or force corresponding to coherent and incoherent summation of structure factor
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        if version==1:
            f_surface=self.calc_fs
        elif version>=1.1:
            f_surface=self.calc_fs_muscovite
        f_layered_water=self.calc_f_layered_water_muscovite(h,k,l,self.domain['layered_water_pars'],height_offset)
        if self.domain['freeze']:#the raxs el has no effect on the structure factor
            f_layered_sorbate=0
        else:
            f_layered_sorbate=self.calc_f_layered_sorbate_muscovite(h,k,l,self.domain['layered_sorbate_pars'],height_offset)
        domains=self.domain['domains']
        if coherence:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*(fb+f_surface(h,k,l,[domains[i]])+f_layered_water+f_layered_sorbate)
        else:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*abs(fb+f_surface(h,k,l,[domains[i]])+f_layered_water+f_layered_sorbate)
        return abs(ftot)*self.inst.inten

    def calc_f4_muscovite_RAXR_MI(self,h,k,x,y,index,height_offset=0,version=1):
        h, k, l, E, E0, f1f2, a, b, c, resonant_el=h,k,y,x,self.domain['E0'],self.domain['F1F2'],self.domain['raxs_vars']['a'+str(index)],self.domain['raxs_vars']['b'+str(index)],self.domain['raxs_vars']['c'+str(index)],self.domain['el']
        ftot=0

        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        f1f2=_extract_f1f2(f1f2,E)

        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        if version==1:
            f_surface=self.calc_fs
        elif version>=1.1:
            f_surface=self.calc_fs_muscovite
        f_layered_water=self.calc_f_layered_water_muscovite(h,k,l,self.domain['layered_water_pars'],height_offset)
        if self.domain['freeze']:
            f_layered_sorbate=0
        else:
            f_layered_sorbate=self.calc_f_layered_sorbate_muscovite_RAXR(h,k,l,self.domain['layered_sorbate_pars'],height_offset,f1f2)
        #only consider one set of Fourier components in the whole strucutre
        A_list=[self.domain['raxs_vars']['A'+str(index)+'_D'+str(i+1)] for i in range(1)]
        P_list=[self.domain['raxs_vars']['P'+str(index)+'_D'+str(i+1)] for i in range(1)]


        domains=self.domain['domains']
        if coherence:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*(fb+f_surface(h,k,l,[domains[i]])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[0]*np.exp(1.0J*np.pi*2*P_list[0]))
        else:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*abs(fb+f_surface(h,k,l,[domains[i]])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[0]*np.exp(1.0J*np.pi*2*P_list[0]))
        ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*abs(ftot)
        #linear attenuation of resonant element from liquid film
        if version>=1.2:
            #electron radius
            _re=self.domain['exp_factors'][2]
            #film thickness
            _mu=self.domain['exp_factors'][1]
            #ra element concentration
            _ra_con=self.domain['exp_factors'][4]
            _q=np.pi*2*self.unit_cell.abs_hkl(h,k,l)
            L_attn=np.array(16*np.pi*(_re/1e7)*6.02e23*_ra_con/1e6*(_mu/1e3)/(_q*1e7))
            li_attn=(np.exp(-L_attn*f1f2[:,1]))**0.5
            return ftot/li_attn*self.inst.inten
        else:
            return ftot*self.inst.inten

    def calc_f4_muscovite_RAXR_MD(self,h,k,x,y,index,height_offset=0,version=1):
        h, k, l, E, E0, f1f2, a, b, c, resonant_el=h,k,y,x,self.domain['E0'],self.domain['F1F2'],self.domain['raxs_vars']['a'+str(index)],self.domain['raxs_vars']['b'+str(index)],self.domain['raxs_vars']['c'+str(index)],self.domain['el']
        ftot=0

        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        if len(f1f2)!=len(E):
            f1f2=_extract_f1f2(f1f2,E)

        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        if version==1:
            f_surface=self.calc_fs_RAXR
        elif version>=1.1:
            f_surface=self.calc_fs_RAXR_muscovite
        f_layered_water=self.calc_f_layered_water_muscovite(h,k,l,self.domain['layered_water_pars'],height_offset)
        f_layered_sorbate=self.calc_f_layered_sorbate_muscovite_RAXR(h,k,l,self.domain['layered_sorbate_pars'],height_offset,f1f2)
        domains=self.domain['domains']

        if coherence:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*(fb+f_surface(h, k, l,[domains[i]],f1f2,resonant_el)+f_layered_water+f_layered_sorbate)
        else:
            for i in range(len(domains)):
                ftot=ftot+getattr(self.domain['global_vars'],'wt'+str(i+1))*abs(fb+f_surface(h, k, l,[domains[i]],f1f2,resonant_el)+f_layered_water+f_layered_sorbate)
        ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*abs(ftot)
        #linear attenuation of resonant element from liquid film
        if version>=1.2:
            #electron radius
            _re=self.domain['exp_factors'][2]
            #film thickness
            _mu=self.domain['exp_factors'][1]
            #ra element concentration
            _ra_con=self.domain['exp_factors'][4]
            _q=np.pi*2*self.unit_cell.abs_hkl(h,k,l)
            L_attn=np.array(16*np.pi*(_re/1e7)*6.02e23*_ra_con/1e6*(_mu/1e3)/(_q*1e7))
            li_attn=(np.exp(-L_attn*f1f2[:,1]))**0.5
            return ftot/li_attn*self.inst.inten
        else:
            return ftot*self.inst.inten

    def cal_structure_factor_hematite_RAXR(self,i,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h, k, y, x, E0, F1F2,SCALES,rough):
        a=getattr(VARS['rgh_raxr'],'a'+str(i+1))
        b=getattr(VARS['rgh_raxr'],'b'+str(i+1))
        c=getattr(VARS['rgh_raxr'],'c'+str(i+1))
        if RAXR_FIT_MODE=='MI':
            A_list,P_list=[],[]
            for index_resonant_el in range(len(RESONANT_EL_LIST)):
                A_list_domain=0
                P_list_domain=0
                if RESONANT_EL_LIST[index_resonant_el]!=0:
                    A_list_domain=getattr(VARS['rgh_raxr'],'A_D'+str(index_resonant_el+1)+'_'+str(i+1))
                    P_list_domain=getattr(VARS['rgh_raxr'],'P_D'+str(index_resonant_el+1)+'_'+str(i+1))
                A_list.append(A_list_domain)
                P_list.append(P_list_domain)
            if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                f = SCALES[0]*rough*self.calc_f4_specular_hematite_RAXR_MI(h, k, y, x, E0, F1F2, a, b, c, A_list, P_list, RESONANT_EL_LIST,RAXR_EL)
            else:
                f = rough*self.calc_f4_nonspecular_hematite_RAXR_MI(h, k, y, x, E0, F1F2, a, b, c, A_list, P_list, RESONANT_EL_LIST)
        elif RAXR_FIT_MODE=='MD':
                if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                    f = SCALES[0]*rough*self.calc_f4_specular_hematite_RAXR_MD(h, k, y, x, E0, F1F2, a, b, c, RESONANT_EL_LIST,RAXR_EL)
                else:
                    f = rough*self.calc_f4_nonspecular_hematite_RAXR_MD(h, k, y, x, E0, F1F2, a, b, c, RESONANT_EL_LIST)
        return f
    def calc_f4_specular_hematite_RAXR_MD(self, h, k, l,E,E0,f1f2,a,b,c,resonant_els=[1,0,0],raxr_el=''):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"

        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs_hematite_RAXR_MD
        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        if len(f1f2)!=len(E):
            f1f2=_extract_f1f2(f1f2,E)
        #(h, k, l,slabs,f1f2,raxr_el)
        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:#consider layered water?
                    f_layered_water=self.calc_f_layered_water_hematite(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():#consider layered sorbate?
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_hematite_RAXR_MD(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water)*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_hematite_RAXR_MD(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el)+f_layered_water)*self.domain[i]['wt']

            ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_nonspecular_hematite_RAXR_MD(self, h, k, l,E,E0,f1f2,a,b,c,resonant_els=[1,0,0],raxr_el=''):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"

        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs_hematite_RAXR_MD
        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        if len(f1f2)!=len(E):
            f1f2=_extract_f1f2(f1f2,E)
        #(h, k, l,slabs,f1f2,raxr_el)
        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el))*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el))*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el))*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,raxr_el))*self.domain[i]['wt']
            ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_specular_hematite_RAXR_MI(self, h, k, l,E,E0,f1f2,a,b,c,A_list=[],P_list=[],resonant_els=[1,0,0],raxr_el=''):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"

        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        if len(f1f2)!=len(E):
            f1f2=_extract_f1f2(f1f2,E)

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:#consider layered water?
                    f_layered_water=self.calc_f_layered_water_hematite(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():#consider layered sorbate?
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_hematite(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_hematite(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']

            ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_nonspecular_hematite_RAXR_MI(self, h, k, l,E,E0,f1f2,a,b,c,A_list=[],P_list=[],resonant_els=[1,0,0],raxr_el=''):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"

        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        def _extract_f1f2(f1f2,E):
            E_f1f2=np.around(f1f2[:,2],0)#make sure E in eV
            E=np.around(E,0)
            index=[]
            for each_E in E_f1f2:
                if each_E in E:
                    index.append(np.where(E_f1f2==each_E)[0][0])
            return f1f2[index,:]

        if len(f1f2)!=len(E):
            f1f2=_extract_f1f2(f1f2,E)

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']

            ftot=np.exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_specular_RAXR(self, h, k, l,E,E0,f1f2,a,b,A_list=[],P_list=[],resonant_els=[1,0,0],raxr_el=''):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"

        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:#consider layered water?
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():#consider layered sorbate?
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,raxr_el,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']

            ftot=(a+b*(E-E0))*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten


    def calc_f4_specular_RAXR_MI(self, h, k, l,E,E0,f1f2,A_list=[],P_list=[],resonant_els=[1,0,0],**abc):
        #calculate the structure factor in the process of model-independent RAXR fitting
        #Use linear background function (abc.keys=['a','b']), or Victoreen background function (abc.keys=['a','b','c'])
        #Linear func: slope{n} = (a(n)*(E{n}-Eo)+1)*b(n)*norm_offset*1/q(n)^2;
        #Victoreen func: slope{n} = exp(-a(n)*(E{n}-Eo).^2/Eo^2 + b(n)*(E{n}-Eo)/Eo) * c(n);
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"
        if len(abc.keys())==3:
            a,b,c=abc['a'],abc['b'],abc['c']
        elif len(abc.keys())==2:
            a,b=abc['a'],abc['b']
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:#consider layered water?
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():#consider layered sorbate?
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water+f_layered_sorbate+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']

            if len(abc.keys())==2:
                ftot=(a*(E-E0)+1)*b*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            else:
                ftot=exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_offspecular_RAXR_MI(self, h, k, l,E,E0,f1f2,A_list=[],P_list=[],resonant_els=[1,0,0],**abc):
        #calculate the structure factor in the process of model-independent RAXR fitting for offspecular rods (no influence from layered water and sorbates)
        #Use linear background function (abc.keys=['a','b']), or Victoreen background function (abc.keys=['a','b','c'])
        #Linear func: slope{n} = (a(n)*(E{n}-Eo)+1)*b(n)*norm_offset*1/q(n)^2;
        #Victoreen func: slope{n} = exp(-a(n)*(E{n}-Eo).^2/Eo^2 + b(n)*(E{n}-Eo)/Eo) * c(n);
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"
        if len(abc.keys())==3:
            a,b,c=abc['a'],abc['b'],abc['c']
        elif len(abc.keys())==2:
            a,b=abc['a'],abc['b']
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*P_list[ii]))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[:,1])*A_list[ii]*np.exp(1.0J*np.pi*2*(P_list[ii]-0.5*l[0])))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']

            if len(abc.keys())==2:
                ftot=(a*(E-E0)+1)*b*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            else:
                ftot=exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_f4_specular_RAXR_for_test_purpose(self, h, k, l,f1f2,res_el='Pb'):
        #this function is used to generate an arbitrary raxr dataset for testing purpose
        #hkl is a list of hkl values
        #f1f2 is in form of [[f1_1,f2_1],[f1_2,f2_2]]
        #The return value is in form of [[],[]] with length =len(f1f2) and the length of each item=len(hkl)
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"


        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs_test_purpose
        f_total_container=[]
        for each_f1f2 in f1f2:
            ftot=0
            for n in range(len(coherence)):
                ftot_A_C, ftot_A_IC=0,0
                ftot_B_C, ftot_B_IC=0,0
                keys_domainA=[]
                keys_domainB=[]

                for i in coherence[n].values()[0]:
                    keys_domainA.append('domain'+str(i+1)+'A')
                    keys_domainB.append('domain'+str(i+1)+'B')
                for i in keys_domainA:
                    ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                    f_layered_water=0
                    if self.domain[i]['layered_water']!=[]:
                        f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                    if coherence[n].keys()[0]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],each_f1f2,res_el)+f_layered_water)*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],each_f1f2,res_el)+f_layered_water)*self.domain[i]['wt']
                for i in keys_domainB:
                    #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                    #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                    ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                    f_layered_water=0
                    if self.domain[i]['layered_water']!=[]:
                        f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                    if coherence[n].keys()[0]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],each_f1f2,res_el)+f_layered_water)*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],each_f1f2,res_el)+f_layered_water)*self.domain[i]['wt']
                ftot=ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C)
                #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
            f_total_container.append(abs(ftot)*self.inst.inten)
        return f_total_container

    def calc_f4_specular_RAXR_MD(self, h, k, l,E,E0,f1f2,resonant_els=[1,0,0],res_el='Zr',**abc):
        #calculate the structure factor in the process of model-dependent RAXR fitting
        #Use linear background function (abc.keys=['a','b']), or Victoreen background function (abc.keys=['a','b','c'])
        #Linear func: slope{n} = (a(n)*(E{n}-Eo)+1)*b(n)*norm_offset*1/q(n)^2;
        #Victoreen func: slope{n} = exp(-a(n)*(E{n}-Eo).^2/Eo^2 + b(n)*(E{n}-Eo)/Eo) * c(n);
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"
        if len(abc.keys())==3:
            a,b,c=abc['a'],abc['b'],abc['c']
        elif len(abc.keys())==2:
            a,b=abc['a'],abc['b']
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs_RAXR

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:#consider layered water?
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():#consider layered sorbate?
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                f_layered_water=0
                f_layered_sorbate=0
                if self.domain[i]['layered_water']!=[]:
                    f_layered_water=self.calc_f_layered_water(h,k,l,*self.domain[i]['layered_water'])
                if 'layered_sorbate' in self.domain[i].keys():
                    if self.domain[i]['layered_sorbate']!=[]:
                        f_layered_sorbate=self.calc_f_layered_sorbate_RAXR(h,k,l,*self.domain[i]['layered_sorbate'])
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el)+f_layered_water+f_layered_sorbate)*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']])+f_layered_water)*self.domain[i]['wt']

            if len(abc.keys())==2:
                ftot=(a*(E-E0)+1)*b*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            else:
                ftot=exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten


    def calc_f4_offspecular_RAXR_MD(self, h, k, l,E,E0,f1f2,resonant_els=[1,0,0],res_el='Zr',**abc):
        #calculate the structure factor in the process of model-dependent RAXR fitting for offspecular rods (no influence from layered water and sorbates)
        #Use linear background function (abc.keys=['a','b']), or Victoreen background function (abc.keys=['a','b','c'])
        #Linear func: slope{n} = (a(n)*(E{n}-Eo)+1)*b(n)*norm_offset*1/q(n)^2;
        #Victoreen func: slope{n} = exp(-a(n)*(E{n}-Eo).^2/Eo^2 + b(n)*(E{n}-Eo)/Eo) * c(n);
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of True or False specifying whether or not considering the resonant scattering in each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.Note in P or A_list, the 0 item means no resonant element
        #                  so len(P_list)==len(resonant_els)
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"
        if len(abc.keys())==3:
            a,b,c=abc['a'],abc['b'],abc['c']
        elif len(abc.keys())==2:
            a,b=abc['a'],abc['b']
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs_RAXR

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']],f1f2,res_el))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+self.calc_fs(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']

            if len(abc.keys())==2:
                ftot=(a*(E-E0)+1)*b*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            else:
                ftot=exp(-a*(E-E0)**2/E0**2+b*(E-E0)/E0)*c*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
        return abs(ftot)*self.inst.inten

    def calc_fs_test_purpose(self, h, k, l,slabs,single_f1f2,res_el):
        '''Calculate the structure factors from the surface
        '''
        #print single_f1f2
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        f=self._get_f(el, dinv)
        shape=f.shape
        f_offset=np.zeros(shape=shape)+0J
        for i in range(shape[0]):
            for j in range(shape[1]):
                if res_el==el[j]:
                    f_offset[i][j]=single_f1f2[0]+1.0J*single_f1f2[1]
        f=f+f_offset
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)

        return fs

    #calculate the Fourier components
    #this function will only consider the specular rod
    #it will calculate Fourier components only for the domainA's
    def find_A_P(self,l,res_el,print_AP=False):
        keys=self.domain.keys()
        keys_domainA=[key for key in keys if "A" in key]
        keys_domainA.sort()
        A,P={},{}
        for each_key in keys_domainA:
            single_domain=self.domain[each_key]
            slabs=[single_domain['slab']]
            domain_wt=single_domain['wt']
            dinv = self.unit_cell.abs_hkl(np.zeros(len(l)), np.zeros(len(l)), np.array(l))
            x, y, z, u, oc, el = self._surf_pars(slabs)
            sorbate_index=[i for i in range(len(el)) if el[i]==res_el]
            A_container,P_container=[],[]

            for each_l in l:
                q=each_l*2*np.pi/self.unit_cell.c
                complex_sum=0.+1.0J*0.
                for i in sorbate_index:
                    complex_sum+=oc[i]*np.exp(-q**2*u[i]**2/2)*np.exp(1.0J*2*np.pi*each_l*(z[i]+1))#z should be plus 1 to account for the fact that surface slab sitting on top of bulk slab
                A_container.append(domain_wt*abs(complex_sum))
                img_complex_sum, real_complex_sum=np.imag(complex_sum),np.real(complex_sum)
                if img_complex_sum==0.:
                    P_container.append(0)
                elif real_complex_sum==0 and img_complex_sum==1:
                    P_container.append(0.25)#1/2pi/2pi
                elif real_complex_sum==0 and img_complex_sum==-1:
                    P_container.append(0.75)#3/2pi/2pi
                else:#adjustment is needed since the return of np.arctan is ranging from -1/2pi to 1/2pi
                    if real_complex_sum>0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.)
                    elif real_complex_sum>0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+1.)
                    elif real_complex_sum<0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
                    elif real_complex_sum<0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
            A[each_key]=A_container
            P[each_key]=P_container
        if print_AP:
            print("l list=",l)
            for each_key in keys_domainA:
                print('\n',each_key)
                print("A list=",['%.4f' % each_A for each_A in A[each_key]])
                print("P list=",['%.4f' % each_P for each_P in P[each_key]])
        return l,A,P

    #calculate the Fourier components
    #this function will only consider the specular rod
    #it will calculate Fourier components only for the domainA's
    def find_A_P_hematite(self,h,k,l,res_el):
        keys=self.domain.keys()
        keys_domainA=[key for key in keys if "A" in key]
        keys_domainA.sort()
        dinv = self.unit_cell.abs_hkl(np.array(h), np.array(k), np.array(l))
        Q=np.pi*2*dinv
        A,P={},{}
        for each_key in keys_domainA:
            single_domain=self.domain[each_key]
            slabs=[single_domain['slab']]
            #domain_wt=single_domain['wt']
            domain_wt=1
            dinv = self.unit_cell.abs_hkl(np.zeros(len(l)), np.zeros(len(l)), np.array(l))
            x, y, z, u, oc, el = self._surf_pars(slabs)
            sorbate_index=[i for i in range(len(el)) if el[i]==res_el]
            A_container,P_container=[],[]

            for q_index in range(len(Q)):
                q=Q[q_index]
                complex_sum=0.+1.0J*0.
                for i in sorbate_index:
                    complex_sum+=oc[i]*np.exp(-q**2*u[i]**2/2)*np.exp(1.0J*q*(z[i]+1)*self.unit_cell.c) #z should be plus 1 to account for the fact that surface slab sitting on top of bulk slab
                A_container.append(domain_wt*abs(complex_sum))
                img_complex_sum, real_complex_sum=np.imag(complex_sum),np.real(complex_sum)
                if img_complex_sum==0.:
                    P_container.append(0)
                elif real_complex_sum==0 and img_complex_sum==1:
                    P_container.append(0.25)#1/2pi/2pi
                elif real_complex_sum==0 and img_complex_sum==-1:
                    P_container.append(0.75)#3/2pi/2pi
                else:#adjustment is needed since the return of np.arctan is ranging from -1/2pi to 1/2pi
                    if real_complex_sum>0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.)
                    elif real_complex_sum>0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+1.)
                    elif real_complex_sum<0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
                    elif real_complex_sum<0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
            A[each_key]=A_container
            P[each_key]=P_container

        return A,P,Q

    def find_A_P_muscovite_original(self,h,k,l):
        A,P=[],[]
        hs,ks,ls=np.array([h]*100),np.array([k]*100),np.arange(0,l,l/100.)
        dinv = self.unit_cell.abs_hkl(hs, ks, ls)
        Q=np.pi*2*dinv
        for i in range(len(self.domain['domains'])):
            single_domain=self.domain['domains'][i]
            slabs=[single_domain]
            x, y, z, u, oc, el = self._surf_pars(slabs)
            res_el=self.domain['el']
            sorbate_index=[ii for ii in range(len(el)) if el[ii]==res_el]
            A_container,P_container=[],[]

            for q_index in range(len(Q)):
                q=Q[q_index]
                h_single,k_single,l_single=hs[q_index],ks[q_index],ls[q_index]
                complex_sum=0.+1.0J*0.
                for j in sorbate_index:
                    complex_sum+=oc[j]*np.exp(-q**2*u[j]**2/2)*np.exp(1.0J*2*np.pi*(h_single*x[j]+k_single*y[j]+l_single*(z[j]+1)))#z should be plus 1 to account for the fact that surface slab sitting on top of bulk slab
                A_container.append(abs(complex_sum))
                img_complex_sum, real_complex_sum=np.imag(complex_sum),np.real(complex_sum)
                if img_complex_sum==0.:
                    P_container.append(0)
                elif real_complex_sum==0 and img_complex_sum==1:
                    P_container.append(0.25)#1/2pi/2pi
                elif real_complex_sum==0 and img_complex_sum==-1:
                    P_container.append(0.75)#3/2pi/2pi
                else:#adjustment is needed since the return of np.arctan is ranging from -1/2pi to 1/2pi
                    if real_complex_sum>0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.)
                    elif real_complex_sum>0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+1.)
                    elif real_complex_sum<0 and img_complex_sum>0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
                    elif real_complex_sum<0 and img_complex_sum<0:
                        P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
            A.append(A_container)
            P.append(P_container)

        return np.transpose(A),np.transpose(P),Q

    def find_A_P_muscovite(self,h,k,l):
        if type(h)==type([]):
            hs,ks,ls=h,k,l
        else:
            hs,ks,ls=np.array([h]*100),np.array([k]*100),np.arange(0,l,l/100.)
        dinv = self.unit_cell.abs_hkl(np.array(hs), np.array(ks), np.array(ls))
        Q=np.pi*2*dinv
        A_container,P_container=[],[]
        for q_index in range(len(Q)):
            q=Q[q_index]
            h_single,k_single,l_single=hs[q_index],ks[q_index],ls[q_index]
            complex_sum=0.+1.0J*0.
            for i in range(len(self.domain['domains'])):
                single_domain=self.domain['domains'][i]
                slabs=[single_domain]
                x, y, z, u, oc, el = self._surf_pars(slabs)
                res_el=self.domain['el']
                sorbate_index=[ii for ii in range(len(el)) if el[ii]==res_el]
                for j in sorbate_index:
                    #complex_sum+=getattr(self.domain['global_vars'],'wt'+str(i+1))*oc[j]*np.exp(-q**2*u[j]**2/2)*np.exp(1.0J*2*np.pi*(h_single*x[j]+k_single*y[j]+l_single*(z[j]+1)))#z should be plus 1 to account for the fact that surface slab sitting on top of bulk slab
                    #l is not necessary perpendicular to z direction
                    complex_sum+=getattr(self.domain['global_vars'],'wt'+str(i+1))*oc[j]*np.exp(-q**2*u[j]**2/2)*np.exp(1.0J*q*(z[j]+1)*self.unit_cell.c)
            A_container.append(abs(complex_sum))
            img_complex_sum, real_complex_sum=np.imag(complex_sum),np.real(complex_sum)
            if img_complex_sum==0.:
                P_container.append(0)
            elif real_complex_sum==0 and img_complex_sum==1:
                P_container.append(0.25)#1/2pi/2pi
            elif real_complex_sum==0 and img_complex_sum==-1:
                P_container.append(0.75)#3/2pi/2pi
            else:#adjustment is needed since the return of np.arctan is ranging from -1/2pi to 1/2pi
                if real_complex_sum>0 and img_complex_sum>0:
                    P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.)
                elif real_complex_sum>0 and img_complex_sum<0:
                    P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+1.)
                elif real_complex_sum<0 and img_complex_sum>0:
                    P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
                elif real_complex_sum<0 and img_complex_sum<0:
                    P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)

        return np.array(A_container),np.array(P_container),Q

    def calc_f4_nonspecular_RAXR(self, h, k, l,E,E0,f1f2,a,b,A_list=[],P_list=[],resonant_els=[1,1,0]):
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]
        #resonant_els:a list of integer numbers (>=0) specifying whether or not considering the resonant scattering in each domain, and how many species on each domain
        #             so the len(resonant_els) is equal to the total domain numbers
        #E is the energy scan list, and make sure items in E is one-to-one corresponding to those in f1f2
        #E0 is the center of the range of energy scan
        #f1f2 numpy array of anomalous correction items (n*2 shape) with the first column as f' and the second as f''
        #a,b are fitting parameters for extrinsic factors
        #P_list and A_list are two lists of Fourier components. Depending on the total domains, you can consider different Fourier
        #                  components for chemically different domains.
        #Resonant structure factor is calculated using equation (9) presented in paper of "Park, Changyong and Fenter, Paul A.(2007) J. Appl. Cryst.40, 290-301"
        ftot=0
        coherence=self.coherence
        fb = self.calc_fb(h, k, l)
        f_surface=self.calc_fs

        for n in range(len(coherence)):
            ftot_A_C, ftot_A_IC=0,0
            ftot_B_C, ftot_B_IC=0,0
            keys_domainA=[]
            keys_domainB=[]

            for i in coherence[n].values()[0]:
                keys_domainA.append('domain'+str(i+1)+'A')
                keys_domainB.append('domain'+str(i+1)+'B')
            for i in keys_domainA:
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[0:,1])*np.sum(np.array(A_list[ii])*np.exp(1.0J*np.pi*2*np.array(P_list[ii]))))*self.domain[i]['wt']
                    else:
                        ftot_A_C=ftot_A_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_A_C=ftot_A_C+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[0:,1])*np.sum(np.array(A_list[ii])*np.exp(1.0J*np.pi*2*np.array(P_list[ii]))))*self.domain[i]['wt']
                    else:
                        ftot_A_IC=ftot_A_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
            for i in keys_domainB:
                #in this specific case (rcut hematite, domainB is symmetricaly related to domainA with half unit cell step lower)
                #in light of that, the Fourier component A(amplitude) is same as that for the associated domainA, but the other one (phase) should be 0.5 off
                ii=int(i[6:-1])-1#extract the domain index from the domain key, eg for "domain10A" will have a 9 as the domain index
                if coherence[n].keys()[0]:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[0:,1])*np.sum(np.array(A_list[ii])*np.exp(1.0J*np.pi*2*(np.array(P_list[ii])-0.5))))*self.domain[i]['wt']
                    else:
                        ftot_B_C=ftot_B_C+(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']
                else:
                    if resonant_els[ii]:
                        ftot_B_C=ftot_B_C+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']])+(f1f2[:,0]+1.0J*f1f2[0:,1])*np.sum(np.array(A_list[ii])*np.exp(1.0J*np.pi*2*(np.array(P_list[ii])-0.5))))*self.domain[i]['wt']
                    else:
                        ftot_B_IC=ftot_B_IC+abs(fb+f_surface(h, k, l,[self.domain[i]['slab']]))*self.domain[i]['wt']

            ftot=(a+b*(E-E0))*(ftot+abs(ftot_A_C)+ftot_A_IC+ftot_B_IC+abs(ftot_B_C))
            #ftot=ftot+ftot_A_C+ftot_A_IC+ftot_B_IC+ftot_B_C
        return abs(ftot)*self.inst.inten

    def calc_f_layered_water(self,h,k,l,u0,ubar,d_w,first_layer_height,density_w=0.033):
        #contribution of layered water calculated as equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height. and the corrections were done accordingly
        #In addition, the occupancy of layered water molecules was correctly calculated here by Auc*d_w*density_w
        #the u0 and ubar here are in A
        dinv = self.unit_cell.abs_hkl(h*0, k*0, l)
        f=self._get_f(np.array(['O']), dinv)[:,0]
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_water=f*(Auc*d_w*density_w)*np.exp(-0.5*q**2*u0**2)*np.exp(q*first_layer_height*1.0J)\
                        /(1-np.exp(-0.5*q**2*ubar**2)*np.exp(q*d_w*1.0J))
        return F_layered_water

    def calc_f_layered_water_muscovite(self,h,k,l,args,height_offset=0):
        #contribution of layered water calculated as equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height. and the corrections were done accordingly
        #In addition, the occupancy of layered water molecules was correctly calculated here by Auc*d_w*density_w
        #the u0 and ubar here are in A
        if h[0]==0 and k[0]==0:#layered structure has effect only on specular rod
            u0,ubar,d_w,first_layer_height,density_w=args['u0_w'],args['ubar_w'],args['d_w'],args['first_layer_height_w'],args['density_w']
            dinv = self.unit_cell.abs_hkl(h, k, l)
            f=self._get_f(np.array(['O']), dinv)[:,0]
            f_H=self._get_f(np.array(['H']), dinv)[:,0]
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            q=2*np.pi*dinv
            #here the first layer height is referenced to 0, which is the height of top most atom layer before relaxation in the surface slab
            F_layered_water=(f+2*f_H)*(Auc*d_w*density_w)*np.exp(-0.5*q**2*u0**2)*np.exp(q*(first_layer_height)*1.0J)\
                            /(1-np.exp(-0.5*q**2*ubar**2)*np.exp(q*d_w*1.0J))#54.3=20.1058*(1+1.6) offset height accouting for bulk and surface slab
            return F_layered_water
        else:
            return 0

    def calc_f_layered_sorbate(self,h,k,l,el,u0_s,ubar_s,d_s,first_layer_height_s,density_s,oc_damping_factor,f1f2=None):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*1*density_s
        #the u0_s and ubar_s here are in A
        #note f1f2 is not used in the function, it serves as a purpose for easy pasting arguments in script
        dinv = self.unit_cell.abs_hkl(h, k, l)
        f=self._get_f(np.array([el]), dinv)[:,0]
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*first_layer_height_s*1.0J)\
                        /(1-np.exp(-oc_damping_factor)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))

        return F_layered_sorbate

    def calc_f_layered_water_hematite(self,h,k,l,u0,ubar,d_w,first_layer_height,density_w=0.033):
        #contribution of layered water calculated as equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height. and the corrections were done accordingly
        #In addition, the occupancy of layered water molecules was correctly calculated here by Auc*d_w*density_w
        #the u0 and ubar here are in A
        dinv = self.unit_cell.abs_hkl(h, k, l)
        f=self._get_f(np.array(['O']), dinv)[:,0]
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_water=f*(Auc*d_w*density_w)*np.exp(-0.5*q**2*u0**2)*np.exp(q*first_layer_height*1.0J)\
                        /(1-np.exp(-0.5*q**2*ubar**2)*np.exp(q*d_w*1.0J))
        return F_layered_water

    def calc_f_layered_sorbate_hematite(self,h,k,l,el,u0_s,ubar_s,d_s,first_layer_height_s,density_s,oc_damping_factor,f1f2=None):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*1*density_s
        #the u0_s and ubar_s here are in A
        #note f1f2 is not used in the function, it serves as a purpose for easy pasting arguments in script
        dinv = self.unit_cell.abs_hkl(h, k, l)
        f=self._get_f(np.array([el]), dinv)[:,0]
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*first_layer_height_s*1.0J)\
                        /(1-np.exp(-oc_damping_factor)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))

        return F_layered_sorbate

    def calc_f_layered_sorbate_hematite_RAXR_MD(self,h,k,l,el,u0_s,ubar_s,d_s,first_layer_height_s,density_s,oc_damping_factor,f1f2):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*d_s*density_s
        #the u0_s and ubar here are in A
        dinv = self.unit_cell.abs_hkl(h, k, l)
        f=self._get_f(np.array([el]), dinv)[:,0]+(f1f2[:,0]+1.0J*f1f2[0:,1])#atomic form factor corrected by the f1f2 correction items
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*first_layer_height_s*1.0J)\
                        /(1-np.exp(-oc_damping_factor)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))
        return F_layered_sorbate

    def calc_f_layered_sorbate_muscovite(self,h,k,l,args,height_offset=0):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*d_s*density_s
        #the u0_s and ubar_s here are in A
        #note f1f2 is not used in the function, it serves as a purpose for easy pasting arguments in script
        if h[0]==0 and k[0]==0:#layered structure has effect only on specular rod
            el,u0_s,ubar_s,d_s,first_layer_height_s,density_s=self.domain['el'],args['u0_s'],args['ubar_s'],args['d_s'],args['first_layer_height_s'],args['density_s']
            try:
                oc_bar=args['oc_damping_factor']
            except:
                oc_bar=0
            dinv = self.unit_cell.abs_hkl(h, k, l)
            f=self._get_f(np.array([el]), dinv)[:,0]
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            q=2*np.pi*dinv
            F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*(first_layer_height_s)*1.0J)\
                            /(1-np.exp(-oc_bar)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))
            return F_layered_sorbate
        else:
            return 0

    def calc_f_layered_sorbate_RAXR(self,h,k,l,el,u0_s,ubar_s,d_s,first_layer_height_s,density_s,oc_damping_factor,f1f2):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*d_s*density_s
        #the u0_s and ubar here are in A
        dinv = self.unit_cell.abs_hkl(h, k, l)
        f=self._get_f(np.array([el]), dinv)[:,0]+(f1f2[:,0]+1.0J*f1f2[0:,1])#atomic form factor corrected by the f1f2 correction items
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        q=2*np.pi*dinv
        F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*first_layer_height_s*1.0J)\
                        /(1-np.exp(-oc_damping_factor)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))
        return F_layered_sorbate

    def calc_f_layered_sorbate_muscovite_RAXR(self,h,k,l,args,height_offset=0,f1f2=None):
        #contribution of layered sorbate calculated based on a function modified from equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        #note here the height of first atom layer is not at 0 as in that equation but is specified by the first_layer_height_s. and the corrections were done accordingly
        #In addition, the occupancy of layered sorbate molecules was correctly calculated here by Auc*d_s*density_s
        #the u0_s and ubar here are in A
        if h[0]==0 and k[0]==0:#layered structure has effect only on specular rod
            el,u0_s,ubar_s,d_s,first_layer_height_s,density_s=self.domain['el'],args['u0_s'],args['ubar_s'],args['d_s'],args['first_layer_height_s'],args['density_s']
            try:
                oc_bar=args['oc_damping_factor']
            except:
                oc_bar=0
            try:
                if f1f2==None:
                    f1f2=self.domain['F1F2']
                else:
                    pass
            except:
                pass

            dinv = self.unit_cell.abs_hkl(h, k, l)
            f=self._get_f(np.array([el]), dinv)[:,0]+(f1f2[:,0]+1.0J*f1f2[:,1])#atomic form factor corrected by the f1f2 correction items
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            q=2*np.pi*dinv
            F_layered_sorbate=f*(Auc*1*density_s)*np.exp(-0.5*q**2*u0_s**2)*np.exp(q*(first_layer_height_s)*1.0J)\
                            /(1-np.exp(-oc_bar)*np.exp(-0.5*q**2*ubar_s**2)*np.exp(q*d_s*1.0J))
            return F_layered_sorbate
        else:
            return 0

    def turbo_calc_f(self, h, k, l):
        '''Calculate the structure factors for the sample with
        inline c code for the surface.
        '''
        fs = self.turbo_calc_fs(h, k, l)
        fb = self.calc_fb(h, k, l)
        ftot = fs + fb
        return ftot*self.inst.inten

    def fourier_synthesis(self,HKL_list,P_list,A_list,z_min=0.,z_max=20.,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'Zr':40,"Th":90,"Rb":37},resonant_el='Pb',resolution=1000,water_scaling=1):

        ZR=el_lib[resonant_el]
        q_list = self.unit_cell.abs_hkl(np.array(HKL_list[0]), np.array(HKL_list[1]), np.array(HKL_list[2]))#a list of 1/d for each hkl set
        q_list_sorted=copy.copy(q_list)
        q_list_sorted.sort()
        q_list_sorted=np.array(q_list_sorted)*np.pi*2#note that q=2pi/d
        delta_q=np.average([q_list_sorted[i+1]-q_list_sorted[i] for i in range(len(q_list_sorted)-1)])
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        z_plot=[]
        eden_plot=[]
        eden_domain_plot=[]
        for i in range(resolution):
            z_each=float(z_max-z_min)/resolution*i+z_min
            z_plot.append(z_each)
            eden=0
            eden_domains=[]
            eden_each_domain=ZR/Auc/np.pi/2*np.sum(A_list*np.cos(2*np.pi*P_list-np.array(q_list_sorted)*z_each)*delta_q)/water_scaling
            eden_domains.append(eden_each_domain)
            eden+=eden_each_domain
            eden_plot.append(eden)
            eden_domain_plot.append(eden_domains)
        return z_plot,eden_plot,eden_domain_plot

    def fourier_synthesis_hematite(self,HKL_list,P_list,A_list,z_min=0.,z_max=20.,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'Zr':40,"Th":90,"Rb":37,'Zn':30},resonant_el='Pb',resolution=1000,water_scaling=1):

        ZR=el_lib[resonant_el]
        q_list = self.unit_cell.abs_hkl(np.array(HKL_list[0]), np.array(HKL_list[1]), np.array(HKL_list[2]))#a list of 1/d for each hkl set
        q_list_sorted=copy.copy(q_list)
        q_list_sorted.sort()
        q_list_sorted=np.array(q_list_sorted)*np.pi*2#note that q=2pi/d
        delta_q=np.average([q_list_sorted[i+1]-q_list_sorted[i] for i in range(len(q_list_sorted)-1)])
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        z_plot=[]
        eden_plot=[]
        eden_domain_plot=[]
        keys=P_list.keys()
        keys.sort()
        for i in range(resolution):
            z_each=float(z_max-z_min)/resolution*i+z_min
            z_plot.append(z_each)
            eden=0
            eden_domains=[]
            for key in keys:
                eden_each_domain=ZR/Auc/np.pi/2*np.sum(np.array([A_list[key]])*np.cos(2*np.pi*np.array([P_list[key]])-np.array(q_list_sorted)*z_each)*delta_q)/water_scaling
                eden_domains.append(eden_each_domain)
                eden+=eden_each_domain
            eden_plot.append(eden)
            eden_domain_plot.append(eden_domains)
        return z_plot,eden_plot,eden_domain_plot

    def fourier_synthesis_original(self,HKL_list,P_list,A_list,z_min=0.,z_max=20.,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'Zr':40},resonant_el='Pb',resolution=1000):
        ZR=el_lib[resonant_el]
        q_list = self.unit_cell.abs_hkl(np.array(HKL_list[0]), np.array(HKL_list[1]), np.array(HKL_list[2]))#a list of 1/d for each hkl set
        q_list_sorted=copy.copy(q_list)
        q_list_sorted.sort()
        q_list_sorted=np.array(q_list_sorted)*np.pi*2#note that q=2pi/d
        delta_q=np.average([q_list_sorted[i+1]-q_list_sorted[i] for i in range(len(q_list_sorted)-1)])
        Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
        z_plot=[]
        eden_plot=[]
        eden_domain_plot=[]
        for i in range(resolution):
            z_each=float(z_max-z_min)/resolution*i+z_min
            z_plot.append(z_each)
            eden=0
            eden_domains=[]
            for j in range(len(P_list)):
                eden_each_domain=ZR/Auc/np.pi*np.sum(A_list[j]*np.cos(2*np.pi*P_list[j]-np.array(q_list_sorted)*z_each)*delta_q)
                eden_domains.append(eden_each_domain)
                eden+=eden_each_domain
            eden_plot.append(eden)
            eden_domain_plot.append(eden_domains)
        return z_plot,eden_plot,eden_domain_plot

    def plot_electron_density(self,slabs,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'P':15,'Cr':24,'Cd':48,'Cu':29,'Zn':30,'Al':13,'Si':14,'K':19},z_min=0.,z_max=28.,N_layered_water=10,resolution=1000,file_path="D:\\"):
        #print dinv
        e_data=[]
        labels=[]
        e_total=np.zeros(resolution)
        keys_sorted=[each for each in slabs.keys() if "A" in each]
        keys_sorted.sort()
        for key in keys_sorted:
            slab=[slabs[key]['slab']]
            x, y, z, u, oc, el = self._surf_pars(slab)
            z=(z+1.)*self.unit_cell.c#z is offseted by 1 unit since such offset is explicitly considered in the calculatino of structure factor
            f=np.array([el_lib[each] for each in el])
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            z_min,z_max=z_min,z_max
            eden=[]
            z_plot=[]
            layered_water,z_layered_water,sigma_layered_water,d_w,water_density=None,[],[],None,None
            if slabs[key]['layered_water']!=[]:
                #the items for the layered water is [u0,ubar,d_w(in A),first_layer_height(in fractional),density_w (in # of waters/A^3)]
                layered_water=slabs[key]['layered_water']
                d_w=layered_water[2]
                water_density=layered_water[-1]
                for i in range(N_layered_water):
                    z_layered_water.append((layered_water[3]+1.)*self.unit_cell.c+i*layered_water[2])#first layer is offseted by 1 accordingly
                    sigma_layered_water.append((layered_water[0]**2+i*layered_water[1]**2)**0.5)
            #consider the e density of layered sorbate
            layered_sorbate,z_layered_sorbate,sigma_layered_sorbate,d_s,sorbate_density=None,[],[],None,None
            if 'layered_sorbate' in slabs[key].keys():
                if slabs[key]['layered_sorbate']!=[]:
                    #the items for the layered sorbate is [el,u0,ubar,d_s(in A),first_layer_height(in fractional),density_s (in # of waters/A^3)]
                    layered_sorbate=slabs[key]['layered_sorbate']
                    d_s=layered_sorbate[3]
                    sorbate_density=layered_sorbate[-2]
                    for i in range(N_layered_water):#assume the number of sorbate layer equal to that for water layers
                        z_layered_sorbate.append((layered_sorbate[4]+1.)*self.unit_cell.c+i*layered_sorbate[3])#first layer is offseted by 1 accordingly
                        sigma_layered_sorbate.append((layered_sorbate[1]**2+i*layered_sorbate[2]**2)**0.5)
            #print u,f,z
            for i in range(resolution):
                z_each=float(z_max-z_min)/resolution*i+z_min
                z_plot.append(z_each)
                #normalized with occupancy and weight factor (manually scaled by a factor 2 to consider the half half of domainA and domainB)
                #here considering the e density for each atom layer will be distributed within a volume of Auc*1, so the unit here is e/A3
                eden.append(np.sum(slabs[key]['wt']*2*oc*f/Auc*(2*np.pi*u**2)**-0.5*np.exp(-0.5/u**2*(z_each-z)**2)))
                if slabs[key]['layered_water']!=[]:
                    #eden[-1]=eden[-1]+np.sum(8*slabs[key]['wt']*2*Auc*d_w*water_density*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2))
                    eden[-1]=eden[-1]+np.sum(8*slabs[key]['wt']*2*water_density*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2))
                if 'layered_sorbate' in slabs[key].keys():
                    if slabs[key]['layered_sorbate']!=[]:
                        eden[-1]=eden[-1]+np.sum(el_lib[slabs[key]['layered_sorbate'][0]]*slabs[key]['wt']*2*sorbate_density*(2*np.pi*np.array(sigma_layered_sorbate)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_sorbate)**2*(z_each-np.array(z_layered_sorbate))**2))

            labels.append(key)
            e_data.append(np.array([z_plot,eden]))
            e_total=e_total+np.array(eden)
        labels.append('Total electron density')
        e_data.append(np.array([list(e_data[0])[0],e_total]))
        pickle.dump([e_data,labels],open(os.path.join(file_path,"temp_plot_eden"),"wb"))

    def plot_electron_density_hematite(self,slabs,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'P':15,'Cr':24,'Cd':48,'Cu':29,'Zn':30,'Al':13,'Si':14,'K':19,'Zr':40,"Th":90,"Rb":37},z_min=0.,z_max=28.,N_layered_water=100,resolution=1000,file_path="D:\\",height_offset=0,version=1.0,freeze=False,raxs_el='Pb'):
        #print dinv
        z_min=z_min
        z_max=z_max
        e_data=[]
        labels=[]
        e_total=np.zeros(resolution)
        e_total_raxs=np.zeros(resolution)
        e_total_layer_water=np.zeros(resolution)
        keys_sorted=[each for each in slabs.keys() if "A" in each]
        keys_sorted.sort()
        for key in keys_sorted:
            slab=[slabs[key]['slab']]
            wt=slabs[key]['wt']
            raxs_el=raxs_el
            x, y, z, u, oc, el = self._surf_pars(slab)
            try:
                sig_eff=slabs[key]['sig_eff']
            except:
                sig_eff=0.203
            u=(u**2+sig_eff**2)**0.5

            index_raxs=np.where(np.array(el)==raxs_el)[0]
            z_raxs=np.array([(z[i]+1.)*self.unit_cell.c for i in index_raxs])
            u_raxs=np.array([u[i] for i in index_raxs])
            oc_raxs=np.array([oc[i] for i in index_raxs])
            f_raxs=el_lib[raxs_el]
            eden_raxs=[]
            eden_layer_water=[]

            z=(z+1.)*self.unit_cell.c#z is offseted by 1 unit since such offset is explicitly considered in the calculatino of structure factor
            f=np.array([el_lib[each] for each in el])
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            z_min,z_max=z_min,z_max
            eden=[]
            z_plot=[]
            layered_water,z_layered_water,sigma_layered_water,d_w,water_density=None,[],[],None,None
            layered_water=slabs[key]['layered_water']
            if layered_water!=[]:
                d_w=layered_water[2]
                water_density=layered_water[-1]
                for i in range(N_layered_water):
                    #z_layered_water.append(layered_water[3]+54.3+height_offset+i*layered_water[2])#first layer is offseted by 1 accordingly
                    z_layered_water.append(7.3707+layered_water[3]+i*layered_water[2])#offset by one unit cell length (7.3707)
                    sigma_layered_water.append((layered_water[0]**2+i*layered_water[1]**2+sig_eff**2)**0.5)
            #consider the e density of layered sorbate
            layered_sorbate,z_layered_sorbate,sigma_layered_sorbate,sorbate_damping_factors,d_s,sorbate_density=None,[],[],[],None,None
            layered_sorbate_keys=['u0_s','ubar_s','d_s','first_layer_height_s','density_s','oc_damping_factor','F1F2']
            layered_sorbate=slabs[key]['layered_sorbate']
            if layered_sorbate!=[]:
                d_s=layered_sorbate[2]
                sorbate_density=layered_sorbate[4]
                damping_factor=layered_sorbate[5]
                for i in range(N_layered_water):#assume the number of sorbate layer equal to that for water layers
                    z_layered_sorbate.append(7.3707+layered_sorbate[3]+i*layered_sorbate[2])#first layer is offseted by 1 unit cell (7.3707 A) accordingly
                    sigma_layered_sorbate.append((layered_sorbate[0]**2+i*layered_sorbate[1]**2+sig_eff**2)**0.5)
                    sorbate_damping_factors.append(damping_factor*i)#first layer no damping, second will be damped with a factor of exp(-damping_factor), third will exp(-2*damping_factor) and so on.
            #print u,f,z
            for i in range(resolution):
                z_each=float(z_max-z_min)/resolution*i+z_min
                z_plot.append(z_each)
                #normalized with occupancy and weight factor (thus normalized to the whole surface area containing multiple domains)
                #here considering the e density for each atom layer will be distributed within a volume of Auc*1, so the unit here is e/A3
                eden.append(np.sum(wt*2*oc*f/Auc*(2*np.pi*u**2)**-0.5*np.exp(-0.5/u**2*(z_each-z)**2)))
                eden_raxs.append(np.sum(wt*2*oc_raxs*f_raxs/Auc*(2*np.pi*u_raxs**2)**-0.5*np.exp(-0.5/u_raxs**2*(z_each-z_raxs)**2)))
                if layered_water!=[]:
                    eden[-1]=eden[-1]+np.sum(10*wt*2*water_density*layered_water[2]*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2))
                    eden_layer_water.append(np.sum(10*wt*2*water_density*layered_water[2]*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2)))
                if layered_sorbate!=[]:
                    eden[-1]=eden[-1]+np.sum(el_lib[raxs_el]*wt*2*sorbate_density*np.exp(-np.array(sorbate_damping_factors))*(2*np.pi*np.array(sigma_layered_sorbate)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_sorbate)**2*(z_each-np.array(z_layered_sorbate))**2))
                    eden_raxs[-1]=eden_raxs[-1]+np.sum(el_lib[raxs_el]*wt*2*sorbate_density*np.exp(-np.array(sorbate_damping_factors))*(2*np.pi*np.array(sigma_layered_sorbate)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_sorbate)**2*(z_each-np.array(z_layered_sorbate))**2))

            labels.append(key)
            #e_data.append(np.array([z_plot,eden,eden_raxs,eden_layer_water]))
            normalized_factor=3.03#3.03:1 electron per 3.03 cubic A
            e_data.append(np.array([np.array(z_plot)-height_offset,np.array(eden)*normalized_factor,np.array(eden_raxs)*normalized_factor,np.array(eden_layer_water)*normalized_factor]))
            e_total=e_total+np.array(eden)
            #if layered_sorbate!=[]:
            e_total_raxs=e_total_raxs+np.array(eden_raxs)
            #if layered_water!=[]:
            e_total_layer_water=e_total_layer_water+np.array(eden_layer_water)
        labels.append('Total electron density')
        #e_data.append(np.array([list(e_data[0])[0],e_total,e_total_raxs,e_total_layer_water]))
        e_data.append(np.array([list(e_data[0])[0],e_total*normalized_factor,e_total_raxs*normalized_factor,e_total_layer_water*normalized_factor]))

        water_scaling=0.33
        pickle.dump([e_data,labels],open(os.path.join(file_path,"temp_plot_eden"),"wb"))
        return water_scaling

    def plot_electron_density_muscovite(self,slabs,el_lib={'O':8,'Fe':26,'As':33,'Pb':82,'Sb':51,'P':15,'Cr':24,'Cd':48,'Cu':29,'Zn':30,'Al':13,'Si':14,'K':19,'Zr':40,"Th":90,"Rb":37},z_min=0.,z_max=28.,N_layered_water=10,resolution=1000,file_path="D:\\",height_offset=0,version=1.0,freeze=False):

        #print dinv
        e_data=[]
        labels=[]
        e_total=np.zeros(resolution)
        e_total_raxs=np.zeros(resolution)
        e_total_layer_water=np.zeros(resolution)

        for domain_index in range(len(slabs['domains'])):
            wt=getattr(slabs['global_vars'],'wt'+str(domain_index+1))
            raxs_el=slabs['el']
            slab=[slabs['domains'][domain_index]]
            x, y, z, u, oc, el = self._surf_pars(slab)
            try:
                sig_eff=slabs['sig_eff']
            except:
                sig_eff=0.203
            u=(u**2+sig_eff**2)**0.5

            index_raxs=np.where(np.array(el)==raxs_el)[0]
            z_raxs=np.array([(z[i]+1.)*self.unit_cell.c for i in index_raxs])
            u_raxs=np.array([u[i] for i in index_raxs])
            oc_raxs=np.array([oc[i] for i in index_raxs])
            f_raxs=el_lib[raxs_el]
            eden_raxs=[]
            eden_layer_water=[]

            z=(z+1.)*self.unit_cell.c#z is offseted by 1 unit since such offset is explicitly considered in the calculatino of structure factor
            f=np.array([el_lib[each] for each in el])
            Auc=self.unit_cell.a*self.unit_cell.b*np.sin(self.unit_cell.gamma)
            z_min,z_max=z_min,z_max
            eden=[]
            z_plot=[]
            layered_water,z_layered_water,sigma_layered_water,d_w,water_density=None,[],[],None,None
            layered_water_keys=['u0_w','ubar_w','d_w','first_layer_height_w','density_w']
            layered_water=[slabs['layered_water_pars'][each_key] for each_key in layered_water_keys]
            d_w=layered_water[2]
            water_density=layered_water[-1]
            for i in range(N_layered_water):
                #z_layered_water.append(layered_water[3]+54.3+height_offset+i*layered_water[2])#first layer is offseted by 1 accordingly
                z_layered_water.append(layered_water[3]+i*layered_water[2])
                sigma_layered_water.append((layered_water[0]**2+i*layered_water[1]**2+sig_eff**2)**0.5)
            #consider the e density of layered sorbate
            layered_sorbate,z_layered_sorbate,sigma_layered_sorbate,sorbate_damping_factors,d_s,sorbate_density=None,[],[],[],None,None
            layered_sorbate_keys=['u0_s','ubar_s','d_s','first_layer_height_s','density_s','oc_damping_factor']
            layered_sorbate=[slabs['layered_sorbate_pars'][each_key] for each_key in layered_sorbate_keys]
            d_s=layered_sorbate[2]
            sorbate_density=layered_sorbate[-2]
            damping_factor=layered_sorbate[-1]
            for i in range(N_layered_water):#assume the number of sorbate layer equal to that for water layers
                z_layered_sorbate.append(layered_sorbate[3]+i*layered_sorbate[2])#first layer is offseted by 1 accordingly
                sigma_layered_sorbate.append((layered_sorbate[0]**2+i*layered_sorbate[1]**2+sig_eff**2)**0.5)
                sorbate_damping_factors.append(damping_factor*i)#first layer no damping, second will be damped with a factor of exp(-damping_factor), third will exp(-2*damping_factor) and so on.
            #print u,f,z
            for i in range(resolution):
                z_each=float(z_max-z_min)/resolution*i+z_min
                z_plot.append(z_each)
                #normalized with occupancy and weight factor (thus normalized to the whole surface area containing multiple domains)
                #here considering the e density for each atom layer will be distributed within a volume of Auc*1, so the unit here is e/A3
                eden.append(np.sum(wt*oc*f/Auc*(2*np.pi*u**2)**-0.5*np.exp(-0.5/u**2*(z_each-z)**2)))
                eden_raxs.append(np.sum(wt*oc_raxs*f_raxs/Auc*(2*np.pi*u_raxs**2)**-0.5*np.exp(-0.5/u_raxs**2*(z_each-z_raxs)**2)))
                bulk_water=0
                if z_each>0:
                    bulk_water=1
                eden[-1]=eden[-1]+np.sum(10*wt*water_density*layered_water[2]*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2))+(.33-0.16233394)*wt*bulk_water*0
                eden_layer_water.append(np.sum(10*wt*water_density*layered_water[2]*(2*np.pi*np.array(sigma_layered_water)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2))+(.33-0.16233394)*wt*bulk_water*0)
                #eden[-1]=eden[-1]+np.sum(10*wt*water_density*(np.exp(-0.5/np.array(sigma_layered_water)**2*(z_each-np.array(z_layered_water))**2)))
                eden[-1]=eden[-1]+np.sum(el_lib[raxs_el]*wt*sorbate_density*np.exp(-np.array(sorbate_damping_factors))*(2*np.pi*np.array(sigma_layered_sorbate)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_sorbate)**2*(z_each-np.array(z_layered_sorbate))**2))

                eden_raxs[-1]=eden_raxs[-1]+np.sum(el_lib[raxs_el]*wt*sorbate_density*np.exp(-np.array(sorbate_damping_factors))*(2*np.pi*np.array(sigma_layered_sorbate)**2)**-0.5*np.exp(-0.5/np.array(sigma_layered_sorbate)**2*(z_each-np.array(z_layered_sorbate))**2))

            labels.append('Domain'+str(domain_index+1))
            #e_data.append(np.array([z_plot,eden,eden_raxs,eden_layer_water]))
            normalized_factor=3.03#3.03:1 electron per 3.03 cubic A
            if domain_index==0:#domain1 has a 0.25 weighting factor
                e_data.append(np.array([z_plot,np.array(eden)*normalized_factor,np.array(eden_raxs)*normalized_factor,np.array(eden_layer_water)*normalized_factor]))
            elif domain_index==1:#domain2 has a 0.75 weighting factor
                e_data.append(np.array([z_plot,np.array(eden)*normalized_factor,np.array(eden_raxs)*normalized_factor,np.array(eden_layer_water)*normalized_factor]))
            if version==1.0:
                e_total=e_total+np.array(eden)
            elif version>=1.1:
                if freeze:
                    e_total=e_total+np.array(np.array(eden)-np.array(eden_raxs))
                else:
                    e_total=e_total+np.array(eden)
            e_total_raxs=e_total_raxs+np.array(eden_raxs)
            e_total_layer_water=e_total_layer_water+np.array(eden_layer_water)
        labels.append('Total electron density')
        #e_data.append(np.array([list(e_data[0])[0],e_total,e_total_raxs,e_total_layer_water]))
        e_data.append(np.array([list(e_data[0])[0],e_total*normalized_factor,e_total_raxs*normalized_factor,e_total_layer_water*normalized_factor]))

        water_scaling=0.33
        pickle.dump([e_data,labels],open(os.path.join(file_path,"temp_plot_eden"),"wb"))
        return water_scaling

    def calc_fs(self, h, k, l,slabs):
        '''Calculate the structure factors from the surface
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        #Note that the u here has been recalculated to represent for the Gaussian distribution width of the thermal vibration (ie sigma in Angstrom)
        f=self._get_f(el, dinv)
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def calc_fs_hematite_RAXR_MD(self, h, k, l,slabs,f1f2,res_el='Pb'):
        '''Calculate the structure factors from the surface
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        #Note that the u here has been recalculated to represent for the Gaussian distribution width of the thermal vibration (ie sigma in Angstrom)
        f=self._get_f(el, dinv)
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        shape=f.shape
        f_offset=np.zeros(shape=shape)+0J
        for i in range(shape[0]):
            for j in range(shape[1]):
                if res_el==el[j]:
                    f_offset[i][j]=f1f2[i][0]+1.0J*f1f2[i][1]
        f=f+f_offset

        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def calc_fs_muscovite(self, h, k, l,slabs):
        '''Calculate the structure factors from the surface
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        try:
            if self.domain['freeze']:
                sub_space_index=[i for i in range(len(slabs[0].id)) if slabs[0].id[i][0:11]!='Freezed_el_']
                x,y,z,u,oc,el=x[sub_space_index],y[sub_space_index],z[sub_space_index],u[sub_space_index],oc[sub_space_index],el[sub_space_index]
        except:
            pass
        #Note that the u here has been recalculated to represent for the Gaussian distribution width of the thermal vibration (ie sigma in Angstrom)
        f=self._get_f(el, dinv)
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def calc_fs_RAXR_muscovite(self, h, k, l,slabs,f1f2,res_el='Zr'):
        '''Calculate the structure factors from the surface with resonant element
           In the normal case, hkl will be an array of same number (eg h=[1]*10,k=[1]*10,l=[1.3]*10,f1f2 has the same length as hkl, but it changes as a function of E)
           Atomic form factor for the res_el will be corrected by those two correction items (f1 and f2)
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        sub_space_index=[i for i in range(len(slabs[0].id)) if slabs[0].id[i][0:11]=='Freezed_el_']
        #Note that the u here has been recalculated to represent for the Gaussian distribution width of the thermal vibration (ie sigma in Angstrom)
        f=self._get_f(el, dinv)
        shape=f.shape
        f_offset=np.zeros(shape=shape)+0J
        for i in range(shape[0]):
            for j in range(shape[1]):
                if res_el==el[j]:
                    try:
                        if j in sub_space_index and self.domain['freeze']:
                            f[:,j]=f[:,j]*0#set resonant element have no effect on the non-resonant structure factor
                            f_offset[i][j]=f1f2[i][0]+1.0J*f1f2[i][1]
                        else:
                            f_offset[i][j]=f1f2[i][0]+1.0J*f1f2[i][1]
                    except:
                        f_offset[i][j]=f1f2[i][0]+1.0J*f1f2[i][1]
        f=f+f_offset
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def calc_fs_RAXR(self, h, k, l,slabs,f1f2,res_el='Zr'):
        '''Calculate the structure factors from the surface with resonant element
           In the normal case, hkl will be an array of same number (eg h=[1]*10,k=[1]*10,l=[1.3]*10,f1f2 has the same length as hkl, but it changes as a function of E)
           Atomic form factor for the res_el will be corrected by those two correction items (f1 and f2)
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars(slabs)
        #Note that the u here has been recalculated to represent for the Gaussian distribution width of the thermal vibration (ie sigma in Angstrom)
        f=self._get_f(el, dinv)
        shape=f.shape
        f_offset=np.zeros(shape=shape)+0J
        for i in range(shape[0]):
            for j in range(shape[1]):
                if res_el==el[j]:
                    f_offset[i][j]=f1f2[i][0]+1.0J*f1f2[i][1]
        f=f+f_offset
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def calc_fs_offspecular(self, h, k, l,slabs):
        '''Calculate the structure factors from the surface
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars_offspecular(slabs)
        f=self._get_f(el, dinv)
        #print x, y,z
        # Create all the atomic structure factors
        #print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape,el.shape
        #change mark 3
        #delta_l=1
        #if self.delta1==[]:delta_l=0
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:,np.newaxis]**2)\
            *np.sum([np.exp(2.0*np.pi*1.0J*(
                 h[:,np.newaxis]*(sym_op.trans_x(x, y)+self.delta1) +
                 k[:,np.newaxis]*(sym_op.trans_y(x, y)+self.delta2) +
                 l[:,np.newaxis]*(z[np.newaxis, :]+1)))
              for sym_op in self.surface_sym], 0)
                    ,1)
        """
        for id in slabs[0].id:
            if "Pb" in str(id):

                print id, np.sum([np.exp(2.0*np.pi*1.0J*(\
                    1*(sym_op.trans_x(x, y)+self.delta1) +\
                    1*(sym_op.trans_y(x, y)+self.delta2) +\
                    1.3*(z[np.newaxis, :]+1)))\
                    for sym_op in self.surface_sym][0][0])#[np.where(slabs[0].id==id)[0][0]]
        """
        return fs

    def turbo_calc_fs(self, h, k, l):
        '''Calculate the structure factors with weave (inline c code)
        Produces faster simulations of large strucutres.
        '''
        h = h.astype(np.float64)
        k = k.astype(np.float64)
        l = l.astype(np.float64)
        #t1 = time.time()
        dinv = self.unit_cell.abs_hkl(h, k, l)
        #t2 = time.time()
        #print 'dinv: %f'%(t2-t1)
        x, y, z, u, oc, el = self._surf_pars()
        #x = np.array(x); y = np.array(y); z= np.array(z)
        f = self._get_f(el, dinv)
        #print f.shape
        Pt = np.array([np.c_[so.P, so.t] for so in self.surface_sym])
        # Setup other stuff needed ...
        im = np.array([1.0J], dtype = np.complex128)
        fs = np.zeros(h.shape, dtype = np.complex128)
        tmp = np.array([0.0J], dtype = np.complex128)
        # Inline c-code goes here..
        code = '''
        double pi = 3.14159265358979311599796346854418516159057617187500;
        int ij = 0;
        int offset = 0;
        //printf("Atoms: %d, Points: %d, Symmetries: %d\\n", Noc[0], Nh[0], NPt[0]);
        // Loop over all data points
        for(int i = 0; i < Nh[0]; i++){
           // Loop over all atoms
           //printf("l = %f\\n", l[i]);
           for(int j = 0; j < Noc[0]; j++){
              ij = i  + j*Nh[0];
              //printf("   x = %f, y = %f, z = %f, u = %f, oc = %f \\n", x[j], y[j], z[j], u[j], oc[j]);
              // Loop over symmetry operations
              tmp[0] = 0.0*tmp[0];
              for(int m = 0; m < NPt[0]; m++){
                 offset = m*6;
                 tmp[0] += exp(2.0*pi*im[0]*(h[i]*(
                          Pt[0 + offset]*x[j] + Pt[1 + offset]*y[j] +
                              Pt[2 + offset])+
                          k[i]*(Pt[3+offset]*x[j] + Pt[4+offset]*y[j]+
                              Pt[5 + offset]) +
                          l[i]*z[j]));
                  if(i == 0 && j == 0 && false){
                     printf("P = [%d, %d] [%d, %d]",
                     Pt[0 + offset], Pt[1 + offset],
                     Pt[3 + offset], Pt[4 + offset]);
                     printf(", t = [%d, %d]\\n", Pt[2 + offset], Pt[5+offset]);

                  } // End if statement
              } // End symmetry loop index m
              fs[i] += oc[j]*f[ij]*exp(-2.0*pow(pi*dinv[i],2.0)*u[j])*tmp[0];
           } // End atom loop index j
        } // End data point (h,k,l) loop
        '''
        #t1 = time.time()
        weave.inline(code, ['x', 'y', 'z', 'h', 'k', 'l', 'u', 'oc', 'f',
                            'Pt', 'im', 'fs', 'dinv', 'tmp'],
                     compiler = 'gcc')
        #t2 = time.time()
        #print t2-t1
        return fs

    def calc_fb(self, h, k, l):
        '''Calculate the structure factors from the bulk
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, el, u, oc, c = self.bulk_slab._extract_values()
        oc = oc/float(len(self.bulk_sym))
        f = self._get_f(el, dinv)
        # Calculate the "shape factor" for the CTRs
        eff_thick = self.unit_cell.c/np.sin(self.inst.alpha*np.pi/180.0)
        alpha = (2.82e-5*self.inst.wavel*eff_thick/self.unit_cell.vol()*
                                              np.sum(f.imag,1))
        #change mark 1,l was changed to zeta
        denom = 1.0-np.exp(-2.0*np.pi*1.0J*(self.delta1*h+self.delta2*k+l))*np.exp(-alpha)
        # Delta functions to remove finite size effect in hk plane
        delta_funcs=(abs(h - np.round(h)) < 1e-12)*(
            abs(k - np.round(k)) < 1e-12)
        # Sum up the uc struct factors
        f_u = np.sum(oc*f*np.exp(-2*np.pi**2*u**2*dinv[:, np.newaxis]**2)*
                     np.sum([np.exp(2.0*np.pi*1.0J*(
                            h[:,np.newaxis]*sym_op.trans_x(x, y) +
                            k[:,np.newaxis]*sym_op.trans_y(x, y) +
                            l[:,np.newaxis]*z [np.newaxis, :]))
                     for sym_op in self.bulk_sym], 0)
                    ,1)
        # Putting it all togheter
        fb = f_u/denom*delta_funcs

        return fb

    def calc_rhos(self, x, y, z, sb = 0.8):
        '''Calcualte the electron density of the unitcell
           Not working yet
        '''
        px, py, pz, u, oc, el = self._surf_pars([self.domain['domain1A']['slab']])
        rhos = self._get_rho(el)


        rho = np.sum([np.sum([rho(self.unit_cell.dist(x, y, z,
                                                      sym_op.trans_x(xat, yat)%1.0,
                                                      sym_op.trans_y(xat, yat)%1.0,
                                                      zat),
                                  0.5*uat+0.5/sb**2, ocat)
                              for rho, xat, yat, zat, uat, ocat in
                              zip(rhos, px, py, pz, u, oc)], 0)
                      for sym_op in self.surface_sym], 0)
        return rho


    def _surf_pars(self,slabs):
        '''Extracts the necessary parameters for simulating the surface part
        '''
        # Extract the parameters we need
        # the star in zip(*... transform the list elements to arguments
        xt, yt, zt, elt, ut, oct, ct = zip(*[slab._extract_values()
                                  for slab in slabs])
        #x1 = np. r_[xt]
        #y1 = np.r_[yt]
        # scale and shift the slabs with respect to each other
        cn = np.cumsum(np.r_[0, ct])[:-1]
        z = np.concatenate([zs*c_s + c_cum
                            for zs, c_cum, c_s in zip(zt, cn, ct)])
        x = np.concatenate([xs + c_cum*self.delta1
                            for xs, c_cum, c_s in zip(xt, cn, ct)])
        y = np.concatenate([ys + c_cum*self.delta2
                            for ys, c_cum, c_s in zip(yt, cn, ct)])
        el = np.r_[elt]
        u = np.r_[ut]
        # Account for overlapping atoms
        oc = np.r_[oct]/float(len(self.surface_sym))
        #print x,y,z, u
        #print y-y1

        return x, y, z, u, oc, el

    def _surf_pars_offspecular(self,slabs):
        '''Extracts the necessary parameters for simulating the surface part
        '''
        #the effect of interfacial molecules wont be included for the calculation of structure factor for offspecular rods
        # Extract the parameters we need
        # the star in zip(*... transform the list elements to arguments

        xt, yt, zt, elt, ut, oct, ct = zip(*[slab._extract_values_offspecular()
                                  for slab in slabs])

        #x1 = np. r_[xt]
        #y1 = np.r_[yt]
        # scale and shift the slabs with respect to each other
        cn = np.cumsum(np.r_[0, ct])[:-1]
        z = np.concatenate([zs*c_s + c_cum
                            for zs, c_cum, c_s in zip(zt, cn, ct)])
        x = np.concatenate([xs + c_cum*self.delta1
                            for xs, c_cum, c_s in zip(xt, cn, ct)])
        y = np.concatenate([ys + c_cum*self.delta2
                            for ys, c_cum, c_s in zip(yt, cn, ct)])
        el = np.r_[elt]
        u = np.r_[ut]
        # Account for overlapping atoms
        oc = np.r_[oct]/float(len(self.surface_sym))
        #print x,y,z, u
        #print y-y1

        return x, y, z, u, oc, el

    def create_uc_output(self):
        ''' Create atomic positions and such for output '''
        x, y, z, u, oc, el = self._surf_pars()
        ids = []
        [ids.extend(slab._extract_ids()) for slab in self.slabs]
        xout = np.array([])
        yout = np.array([])
        zout = np.array([])
        uout = np.array([])
        ocout = np.array([])
        elout = el[0:0].copy()
        idsout = []
        for sym_op in self.surface_sym:
            xout = np.r_[xout, sym_op.trans_x(x, y)]
            yout = np.r_[yout, sym_op.trans_y(x, y)]
            zout = np.r_[zout, z]
            uout = np.r_[uout, u]
            ocout = np.r_[ocout, oc]
            elout = np.r_[elout, el]
        idsout.extend(ids)

        return xout, yout, zout, uout, ocout, elout, idsout

    def _get_f(self, el, dinv):
        '''from the elements extract an array with atomic structure factors
        '''
        return _get_f(self.inst, el, dinv)

    def _get_rho(self, el):
        '''Returns the rho functions for all atoms in el
        '''
        return _get_rho(self.inst, el)

    def _fatom_eval(self, f, element, s):
        '''Smart (fast) evaluation of f_atom. Only evaluates f if not
        evaluated before.

        element - element string
        f - dictonary for lookup
        s - sintheta_over_lambda array
        '''
        return _fatom_eval(inst, f, element, s)

class UnitCell:
    '''Class containing the  unitcell.
    This also allows for simple crystalloraphic computing of different
    properties.
    '''
    def __init__(self, a, b, c, alpha = 90,
                 beta = 90, gamma = 90):
        self.set_a(a)
        self.set_b(b)
        self.set_c(c)
        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_gamma(gamma)

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b

    def set_c(self, c):
        self.c = c

    def set_alpha(self, alpha):
        self.alpha = alpha*np.pi/180.

    def set_beta(self, beta):
        self.beta = beta*np.pi/180.

    def set_gamma(self, gamma):
        self.gamma = gamma*np.pi/180.

    def vol(self):
        '''Calculate the volume of the unit cell in AA**3
        '''
        vol = self.a*self.b*self.c*np.sqrt(1 - np.cos(self.alpha)**2 -
                np.cos(self.beta)**2  - np.cos(self.gamma)**2 +
                2*np.cos(self.alpha)*np.cos(self.beta)*np.cos(self.gamma))
        return vol

    def cart_coords(self, uc_x, uc_y, uc_z):
        '''Transform the uc coors uc_x, uc_y, uc_z to cartesian
        coordinates expressed in AA
        '''
        return (cart_coord_x(uc_x, uc_y, uc_z), cart_coord_y(uc_x, uc_y, uc_z),
                cart_coord_z(uc_x, uc_y, uc_z))

    def cart_coord_x(self, uc_x, uc_y, uc_z):
        '''Get the x-coord in the cart system
        '''
        return uc_x*self.a

    def cart_coord_y(self, uc_x, uc_y, uc_z):
        '''Get the y-coord in the cart system
        '''
        return uc_y*self.b

    def cart_coord_z(self, uc_x, uc_y, uc_z):
        '''Get the y-coord in the cart system
        '''
        return uc_z*self.c

    def dist(self, x1, y1, z1, x2, y2, z2):
        '''Calculate the distance in AA between the points
        (x1, y1, z1) and (x2, y2, z2). The coords has to be unit cell
        coordinates.
        '''
        #print 'Warning works only with orth cryst systems!'
        return np.sqrt(((x1 - x2)*self.a)**2 + ((y1 - y2)*self.b)**2 +
                       ((z1 - z2)*self.c)**2)

    def abs_hkl(self, h, k, l):
        '''Returns the absolute value of (h,k,l) vector in units of
        AA.

        This is equal to the inverse lattice spacing 1/d_hkl.
        '''
        dinv = np.sqrt(((h/self.a*np.sin(self.alpha))**2 +
                         (k/self.b*np.sin(self.beta))**2  +
                         (l/self.c*np.sin(self.gamma))**2 +
                        2*k*l/self.b/self.c*(np.cos(self.beta)*
                                             np.cos(self.gamma) -
                                             np.cos(self.alpha)) +
                        2*l*h/self.c/self.a*(np.cos(self.gamma)*
                                             np.cos(self.alpha) -
                                             np.cos(self.beta)) +
                        2*h*k/self.a/self.b*(np.cos(self.alpha)*
                                             np.cos(self.beta) -
                                             np.cos(self.gamma)))
                        /(1 - np.cos(self.alpha)**2 - np.cos(self.beta)**2
                          - np.cos(self.gamma)**2 + 2*np.cos(self.alpha)
                          *np.cos(self.beta)*np.cos(self.gamma)))
        return dinv

class Slab:
    par_names = ['dx1','dx2','dx3','dx4','dy1','dy2','dy3','dy4','dz1','dz2','dz3','dz4',\
                          'u', 'du','oc','doc', 'm']
    def __init__(self, name = '', c = 1.0, slab_oc = 1.0, T_factor='u'):
        try:
            self.c = float(c)
        except:
            raise ValueError("Parameter c has to be a valid floating point number")
        try:
            self.slab_oc = float(slab_oc)
        except:
            raise ValueError("Parameter slab_oc has to be a valid floating point number")
        # Set the arrays to their default values
        self.x = np.array([], dtype = np.float64)
        self.y = np.array([], dtype = np.float64)
        self.z = np.array([], dtype = np.float64)
        self.dx1 = np.array([], dtype = np.float64)
        self.dx2 = np.array([], dtype = np.float64)
        self.dx3 = np.array([], dtype = np.float64)
        self.dx4 = np.array([], dtype = np.float64)
        self.dy1 = np.array([], dtype = np.float64)
        self.dy2 = np.array([], dtype = np.float64)
        self.dy3 = np.array([], dtype = np.float64)
        self.dy4 = np.array([], dtype = np.float64)
        self.dz1 = np.array([], dtype = np.float64)
        self.dz2 = np.array([], dtype = np.float64)
        self.dz3 = np.array([], dtype = np.float64)
        self.dz4 = np.array([], dtype = np.float64)
        self.u = np.array([], dtype = np.float64)
        self.oc = np.array([], dtype = np.float64)
        self.du = np.array([], dtype = np.float64)
        self.doc = np.array([], dtype = np.float64)
        self.m = np.array([], dtype = np.float64)
        self.id = np.array([], dtype = np.str)
        self.el = np.array([], dtype = np.str)
        self.T_factor=T_factor

        # TODO: Type checking and defaults!
        #self.inst = inst
        self.name = str(name)

    def copy(self):
        '''Returns a copy of the object.
        '''
        #T_factor must be 'u', not matter what's that for the original one, since they have been transfered to u already.
        self.id = list(self.id)
        cpy = Slab(c = self.c, slab_oc = self.slab_oc,T_factor=self.T_factor)
        for i in range(len(self.id)):
            cpy.add_atom(str(self.id[i]), str(self.el[i]),
                         self.x[i], self.y[i],
                         self.z[i], self.u[i], self.oc[i], self.m[i])
            cpy.dz1[-1] = self.dz1[i]
            cpy.dz2[-1] = self.dz2[i]
            cpy.dz3[-1] = self.dz3[i]
            cpy.dz4[-1] = self.dz4[i]
            cpy.dx1[-1] = self.dx1[i]
            cpy.dx2[-1] = self.dx2[i]
            cpy.dx3[-1] = self.dx3[i]
            cpy.dx4[-1] = self.dx4[i]
            cpy.dy1[-1] = self.dy1[i]
            cpy.dy2[-1] = self.dy2[i]
            cpy.dy3[-1] = self.dy3[i]
            cpy.dy4[-1] = self.dy4[i]
            cpy.du[-1] = self.du[i]
            cpy.doc[-1] = self.doc[i]
        return cpy

    def add_atom(self,id,  element, x, y, z, u = 0.0, oc = 1.0, m = 1.0):
        '''Add an atom to the slab.

        id - a unique id for this atom (string)
        element - the element of this atom has to be found
        within the scatteringlength table.
        x, y, z - position in the assymetricv unit cell (floats)
        u - debye-waller parameter for the atom
        oc - occupancy of the atomic site
        '''
        if id in self.id:
            raise ValueError('The id %s is already defined in the'
                             'slab'%(id))
        # TODO: Check the element as well...
        self.x = np.append(self.x, x)
        self.dx1 = np.append(self.dx1, 0.)
        self.dx2 = np.append(self.dx2, 0.)
        self.dx3 = np.append(self.dx3, 0.)
        self.dx4 = np.append(self.dx4, 0.)
        self.y = np.append(self.y, y)
        self.dy1 = np.append(self.dy1, 0.)
        self.dy2 = np.append(self.dy2, 0.)
        self.dy3 = np.append(self.dy3, 0.)
        self.dy4 = np.append(self.dy4, 0.)
        self.z = np.append(self.z, z)
        self.dz1 = np.append(self.dz1, 0.)
        self.dz2 = np.append(self.dz2, 0.)
        self.dz3 = np.append(self.dz3, 0.)
        self.dz4 = np.append(self.dz4, 0.)
        self.du = np.append(self.du, 0.)
        self.doc = np.append(self.doc, 0.)
        self.u = np.append(self.u, u)
        self.oc = np.append(self.oc, oc)
        self.m = np.append(self.m, m)
        self.id = np.append(self.id, id)
        self.el = np.append(self.el, str(element))
        item = len(self.id) - 1
        # Create the set and get functions dynamically
        for par in self.par_names:
            p = par
            setattr(self, 'set' + id + par, self._make_set_func(par, item))
            setattr(self, 'get' + id + par, self._make_get_func(par, item))

        return AtomGroup(self, id)

    def insert_atom(self,index,id,element, x, y, z, u = 0.0, oc = 1.0, m = 1.0):
        '''Add an atom to the slab.

        id - a unique id for this atom (string)
        element - the element of this atom has to be found
        within the scatteringlength table.
        x, y, z - position in the assymetricv unit cell (floats)
        u - debye-waller parameter for the atom
        oc - occupancy of the atomic site
        '''
        if id in self.id:
            raise ValueError('The id %s is already defined in the'
                             'slab'%(id))
        # TODO: Check the element as well...
        self.x = np.insert(self.x,[index+1], x)
        self.dx1 = np.insert(self.dx1, [index+1],0.)
        self.dx2 = np.insert(self.dx2, [index+1],0.)
        self.dx3 = np.insert(self.dx3, [index+1],0.)
        self.dx4 = np.insert(self.dx4, [index+1],0.)
        self.y = np.insert(self.y, [index+1],y)
        self.dy1 = np.insert(self.dy1,[index+1], 0.)
        self.dy2 = np.insert(self.dy2, [index+1],0.)
        self.dy3 = np.insert(self.dy3, [index+1],0.)
        self.dy4 = np.insert(self.dy4, [index+1],0.)
        self.z = np.insert(self.z, [index+1],z)
        self.dz1 = np.insert(self.dz1, [index+1],0.)
        self.dz2 = np.insert(self.dz2, [index+1],0.)
        self.dz3 = np.insert(self.dz3, [index+1],0.)
        self.dz4 = np.insert(self.dz4, [index+1],0.)
        self.du = np.insert(self.du, [index+1],0.)
        self.doc = np.insert(self.doc, [index+1],0.)
        self.u = np.insert(self.u,[index+1],u)
        self.oc = np.insert(self.oc,[index+1],oc)
        self.m = np.insert(self.m,[index+1],m)
        self.id = np.insert(self.id,[index+1],id)
        self.el = np.insert(self.el,[index+1],str(element))
        item = len(self.id) - 1
        # Create the set and get functions dynamically
        for par in self.par_names:
            p = par
            setattr(self, 'set' + id + par, self._make_set_func(par, item))
            setattr(self, 'get' + id + par, self._make_get_func(par, item))
        return AtomGroup(self, id)

    def del_atom(self, id):
        '''Remove atom identified with id
        '''
        if not id in self.id:
            raise ValueError('Can not remove atom with id %s -'
                             'namedoes not exist')
        item = np.argwhere(self.id == id)[0][0]

        for par in self.par_names:
                for id in self.id:
                    delattr(self, 'set' + id + par)
                    delattr(self, 'get' + id + par)

        if item < len(self.x) - 1:
            ar = getattr(self, 'id')
            setattr(self, 'id', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'el')
            setattr(self, 'el', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'x')
            setattr(self, 'x', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'y')
            setattr(self, 'y', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'z')
            setattr(self, 'z', np.r_[ar[:item], ar[item+1:]])

            for par in self.par_names:
                ar = getattr(self, par)
                setattr(self, par, np.r_[ar[:item], ar[item+1:]])
            #when you delete one atom, you must reset the set_get function, since the order of parameter values will change.
            for par in self.par_names:
                for id in self.id:
                    setattr(self, 'set' + id + par, self._make_set_func(par, np.where(self.id==id)[0][0]))
                    setattr(self, 'get' + id + par, self._make_get_func(par, np.where(self.id==id)[0][0]))
        else:
            ar = getattr(self, 'id')
            setattr(self, 'id', ar[:-1])
            ar = getattr(self, 'el')
            setattr(self, 'el', ar[:-1])
            ar = getattr(self, 'x')
            setattr(self, 'x', ar[:-1])
            ar = getattr(self, 'y')
            setattr(self, 'y', ar[:-1])
            ar = getattr(self, 'z')
            setattr(self, 'z', ar[:-1])

            for par in self.par_names:
                ar = getattr(self, par)
                setattr(self, par, ar[:-1])

            for par in self.par_names:
                for id in self.id:
                    setattr(self, 'set' + id + par, self._make_set_func(par, np.where(self.id==id)[0][0]))
                    setattr(self, 'get' + id + par, self._make_get_func(par, np.where(self.id==id)[0][0]))



    def find_atoms(self, expression):
        '''Find the atoms that satisfy the logical expression given in the
        string expression. Expression can also be a list or array of the
        same length as the number of atoms in the slab.

        Allowed variables in expression are:
        x, y, z, u, occ, id, el
        returns an AtomGroup
        '''
        if (type(expression) == type(np.array([])) or
            type(expression) == type(list([]))):
            if len(expression) != len(self.id):
                raise ValueError('The length of experssion is wrong'
                                 ', it should match the number of atoms')
            ag = AtomGroup()
            [ag.add_atom(self, str(id)) for id, add in
             zip(self.id, expression) if add]
            return ag
        elif type(expression) == type(''):
            choose_list = [eval(expression) for x,y,z,u,oc,el,id in
                           zip(self.x, self.y, self.z, self.u,
                               self.oc, self.el, self.id)]
            #print choose_list
            ag = AtomGroup()
            [ag.add_atom(self, str(name)) for name, add
             in zip(self.id, choose_list) if add]
            return ag
        else:
            raise ValueError('Expression has to be a string, array or list')

    def all_atoms(self):
        '''Puts all atoms in the slab to an AtomGroup.

        returns: AtomGroup
        '''
        return self.find_atoms([True]*len(self.id))


    def set_c(self, c):
        '''Set the out-of-plane extension of the slab.
        Note that this is in the defined UC coords given in
        the corresponding sample
        '''
        self.c = float(c)

    def get_c(self):
        '''Get the out-of-plane extension of the slab in UC coord.
        '''
        return self.c

    def set_oc(self, oc):
        '''Set a global occupation parameter for the entire slab.
        should be between 0 and 1. To create the real occupancy this
        value is multiplied with the occupancy for that atom.
        '''
        self.slab_oc = oc

    def get_oc(self):
        '''Get the global occupancy of the slab
        '''
        return self.slab_oc

    def __getitem__(self, id):
        '''Locate id in slab with a dictonary style.
        Returns a AtomGroup instance
        '''
        return AtomGroup(self, id)

    def __contains__(self, id):
        '''Makes it possible to check if id exist in this Slab by using
        the in operator. It is also possible if all atoms in an AtomGroup
        belongs to the slab.

        returns True or False
        '''
        if type(id) == type(''):
            return id in self.id
        elif type(id) == type(AtomGroup):
            return np.all([atid in self.id for atid in id.ids])
        else:
            raise ValueError('Can only check for mebership for Atom groups'
                             'or string ids.')

    def _set_in(self, arr, pos, value):
        '''Sets a value in an array or list
        '''
        arr[pos]=value

    def _make_set_func(self, par, pos):
        ''' Creates a set functions for parameter par and at pos.
        Returns a function
        '''
        def set_par(val):
            getattr(self, par)[pos] = val

        return set_par

    def _make_get_func(self, par, pos):
        '''Cerates a set function for member par at pos.
        Returns a function.
        '''
        def get_par(scale=1.):
            return getattr(self, par)[pos]/scale

        return get_par

    def _extract_values(self):
        #B=8*pi*pi*u*u in A2
        #u in A
        if self.T_factor=='B':
            return  self.x + self.dx1+self.dx2+self.dx3+self.dx4, self.y + self.dy1+self.dy2+self.dy3+self.dy4, self.z + self.dz1+ self.dz2+ self.dz3+self.dz4,\
                    self.el, (self.u/(8*np.pi**2))**0.5+self.du, (self.oc+self.doc)*self.m*self.slab_oc, self.c
        elif self.T_factor=='u':
            return  self.x + self.dx1+self.dx2+self.dx3+self.dx4, self.y + self.dy1+self.dy2+self.dy3+self.dy4, self.z + self.dz1+ self.dz2+ self.dz3+self.dz4,\
                   self.el, (self.u)**0.5+self.du, (self.oc+self.doc)*self.m*self.slab_oc, self.c

    def _extract_values_offspecular(self):
        ids=self.id
        ii=None#index for first water molecule
        for i in range(1,30):#water molecules will be added at the very end and wont exceed 10 usually
            if 'Os' not in ids[-i]:
                ii=len(ids)-i+1
                break
            else:
                pass
        if self.T_factor=='B':
            return  self.x[0:ii] + self.dx1[0:ii]+self.dx2[0:ii]+self.dx3[0:ii]+self.dx4[0:ii], self.y[0:ii] + self.dy1[0:ii]+self.dy2[0:ii]+self.dy3[0:ii]+self.dy4[0:ii], self.z[0:ii] + self.dz1[0:ii]+ self.dz2[0:ii]+ self.dz3[0:ii]+self.dz4[0:ii],\
                    self.el[0:ii], (self.u[0:ii]/(8*np.pi**2))**0.5+self.du[0:ii], (self.oc[0:ii]+self.doc[0:ii])*self.m[0:ii]*self.slab_oc, self.c
        elif self.T_factor=='u':
            return  self.x[0:ii] + self.dx1[0:ii]+self.dx2[0:ii]+self.dx3[0:ii]+self.dx4[0:ii], self.y[0:ii] + self.dy1[0:ii]+self.dy2[0:ii]+self.dy3[0:ii]+self.dy4[0:ii], self.z[0:ii] + self.dz1[0:ii]+ self.dz2[0:ii]+ self.dz3[0:ii]+self.dz4[0:ii],\
                   self.el[0:ii], self.u[0:ii]+self.du[0:ii], (self.oc[0:ii]+self.doc[0:ii])*self.m[0:ii]*self.slab_oc, self.c

    def _extract_values2(self):
        return  self.x + self.dx1+self.dx2+self.dx3+self.dx4, self.y + self.dy1+self.dy2+self.dy3+self.dy4, self.z + self.dz1+ self.dz2+ self.dz3+self.dz4,\
                   self.el, self.u+self.du, (self.oc+self.doc)*self.m*self.slab_oc, self.c

    def _extract_ids(self):
        'Extract the ids of the atoms'
        return [self.name + '.' + str(id) for id in self.id]

class AtomGroup:
    par_names = ['dx', 'dy', 'dz', 'u', 'oc']
    def __init__(self, slab = None, id = None,matrix=[1,0,0,0,1,0,0,0,1]):

        self.ids = []
        self.slabs = []
        # Variable for composition ...
        self.comp = 1.0
        self.oc = 1.0
        self.sym=[]
        if slab != None and  id != None:
            self.add_atom(slab, id, matrix)

    def _set_func(self, par):
        '''create a function that sets all atom paramater par'''
        #id_=list(np.copy(self.ids))
        #id_.sort()
        #print id_
        funcs=[]
        #here you must make sure the id is different even for different slab
        for i in range(len(self.ids)):
        #the change of dx,dy or dz will accordingly change dx,dy and dz at the same time
        #to eliminate the overwriting, the changes go to temp dxn,dyn and dzn. At the time of calculating structure factor
        #sum of dxn will be added to x, sum of dyn will be added up to y, and sum of dzn will be added up to z
            id=self.ids[i]
            if (par=='dx'):
                funcx=getattr(self.slabs[i], 'set'+ id  + 'dx1')
                funcy=getattr(self.slabs[i], 'set'+ id  + 'dy1')
                funcz=getattr(self.slabs[i], 'set'+ id  + 'dz1')
                funcs.append([funcx,funcy,funcz])
            elif (par=='dy'):
                funcx=getattr(self.slabs[i], 'set'+ id  + 'dx2')
                funcy=getattr(self.slabs[i], 'set'+ id  + 'dy2')
                funcz=getattr(self.slabs[i], 'set'+ id  + 'dz2')
                funcs.append([funcx,funcy,funcz])
            elif (par=='dz'):
                funcx=getattr(self.slabs[i], 'set'+ id  + 'dx3')
                funcy=getattr(self.slabs[i], 'set'+ id  + 'dy3')
                funcz=getattr(self.slabs[i], 'set'+ id  + 'dz3')
                funcs.append([funcx,funcy,funcz])
            else:funcs.append(getattr(self.slabs[i], 'set'+ id  +  par))
        def set_pars(val):
            #print self.sym_file.shape
            for i in range(len(funcs)):
                #the corresponding infomation stored in sym_row, id_order_in_sym_file is the ids of atoms with its order
                #appearing the same as that in sym files, say, if I have a id1 at the first place, then the order is defined as 0
                #which is order of id1's symmetry operations in sym file, thus the first row is the associated sym opts.
                if par=='dx':
                    funcs[i][0](val*self.sym[i][0])
                    funcs[i][1](val*self.sym[i][1])
                    funcs[i][2](val*self.sym[i][2])
                    #print i,'dx',val
                elif par=='dy':
                    funcs[i][0](val*self.sym[i][3])
                    funcs[i][1](val*self.sym[i][4])
                    funcs[i][2](val*self.sym[i][5])
                    #i,'dy',val
                elif par=='dz':
                    # try:
                        # print(self.ids[i])
                        # print(dir(self))
                        # print(list(val))
                    # except:
                        # pass
                    funcs[i][0](val*self.sym[i][6])
                    funcs[i][1](val*self.sym[i][7])
                    funcs[i][2](val*self.sym[i][8])
                    #i,'dz',val
                else: funcs[i](val)
        return set_pars

    def _get_func(self, par):
        '''create a function that gets all atom paramater par'''
        funcs = []
        for id, slab in zip(self.ids, self.slabs):
            if par=='dx':
                funcs.append(getattr(slab, 'get' + id + 'dx1'))
            elif par=='dy':
                funcs.append(getattr(slab, 'get' + id + 'dy2'))
            elif par=='dz':
                funcs.append(getattr(slab, 'get' + id + 'dz3'))
            else:funcs.append(getattr(slab, 'get' + id + par))

        def get_pars():
            if par=='dx':
                return np.mean([func(self.sym[funcs.index(func)][0]+1.0e-30)  for func in funcs])
            elif par=='dy':
                return np.mean([func(self.sym[funcs.index(func)][4]+1.0e-30)  for func in funcs])
            elif par=='dz':
                return np.mean([func(self.sym[funcs.index(func)][8]+1.0e-30)  for func in funcs])
            else: return np.mean([func() for func in funcs])
        return get_pars

    def update_setget_funcs(self,matrix):
        '''Update all the atomic set and get functions
        '''
        for par in self.par_names:
            setattr(self, 'set' + par, self._set_func(par))
            setattr(self, 'get' + par, self._get_func(par))

    def add_atom(self, slab, id,matrix=[1,0,0,0,1,0,0,0,1]):
        '''Add an atom to the group.
        '''

        if not id in slab:
            raise ValueError('The id %s is not a member of the slab'%id)

        self.ids.append(id)

        #print self.sym_file
        self.slabs.append(slab)
        self.sym.append(matrix)
        self.update_setget_funcs(matrix)

    def _copy(self):
        '''Creates a copy of self And looses all connection to the
        previously created compositions conenctions
        '''
        cpy = AtomGroup()
        cpy.ids = self.ids[:]
        cpy.slabs = self.slabs[:]
        cpy.update_setget_funcs()
        return cpy

    def comp_coupl(self, other, self_copy = False, exclusive = True):
        '''Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. If self_copy is True the
        returned value will be a copy of self.
        If exculive is true reomves all methods from the
        previous AtomGroups that are coupled.
        '''
        if not type(self) == type(other):
            raise TypeError('To create a composition function both objects'
                            ' has to be of the type AtomGroup')
        if hasattr(other, '_setoc_'):
            raise AttributeError('The right hand side AtomicGroup has already'
                                 'been coupled to another one before.'
                                 ' Only one connection'
                                 'is allowed')
        if hasattr(self, '_setoc'):
            raise AttributeError('The left hand side AtomicGroup has already'
                                 'been coupled to another one before.'
                                 ' Only one connection'
                                 'is allowed')
        if self_copy:
            s = self._copy()
        else:
            s = self

        def set_comp(comp):
            #print "Executing comp function"
            s.comp = float(comp)
            s._setoc(comp*s.oc)
            other._setoc_((1.0 - comp)*s.oc)

        def set_oc(oc):
            #print "Executing oc function"
            s.oc = float(oc)
            s._setoc(s.comp*s.oc)
            other._setoc_((1 - s.comp)*s.oc)

        def get_comp():
            return s.comp

        def get_oc():
            return s.oc

        # Functions to couple the other parameters, set
        def create_set_func(par):
            sf_set = getattr(s, 'set' + par)
            of_set = getattr(other, 'set' + par)
            def _set_func(val):
                p = str(par)
                #print 'Setting %s to %s'%(p, val)
                sf_set(val)
                of_set(val)
            return _set_func

        # Functions to couple the other parameters, set
        def create_get_func(par):
            sf_get = getattr(s, 'get' + par)
            of_get = getattr(other, 'get' + par)
            def _get_func():
                p = str(par)
                return (sf_get() + of_get())/2
            return _get_func

        # Do it (couple) for all parameters except the occupations
        if exclusive:
            for par in s.par_names:
                if not str(par) == 'oc':
                    #print par
                    setattr(s, 'set' + par, create_set_func(par))
                    setattr(s, 'get' + par, create_get_func(par))

        # Create new set and get methods for the composition
        setattr(s, 'setcomp', set_comp)
        setattr(s, 'getcomp', get_comp)

        # Store the original setoc for future use safely
        setattr(s, '_setoc', s.setoc)
        setattr(other, '_setoc_', getattr(other, 'setoc'))

        setattr(s, 'setoc', set_oc)
        setattr(s, 'getoc', get_oc)

        # Now remove all the coupled attribute from other.
        if exclusive:
            for par in s.par_names:
                delattr(other, 'set' + par)

        s.setcomp(1.0)

        return s



    def __xor__(self, other):
        '''Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. Note that the
        first element (left hand side of ^) will be copied
        and loose all its previous connections.
        Note that all the move methods that are not coupled will
        be removed.
        '''
        return self.comp_coupl(other, self_copy = True, exclusive = True)

    def __ixor__(self, other):
        '''Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy.
        Note that all the move methods that are not coupled will
        be removed.
        '''
        self.comp_coupl(other, exclusive = True)

    def __or__(self, other):
        '''Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. Note that the
        first element (left hand side of |) will be copied
        and loose all its previous connections.
        '''
        return self.comp_coupl(other, self_copy = True, exclusive = False)

    def __ior__(self, other):
        '''Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy.
        '''
        self.comp_coupl(other, exclusive = False)

    def __add__(self, other):
        '''Adds two Atomic groups togheter
        '''
        if not type(other) == type(self):
            raise TyepError('Adding wrong type to an AtomGroup has to be an'
                            'AtomGroup')
        ids = self.ids + other.ids
        slabs = self.slabs + other.slabs
        out = AtomGroup()
        [out.add_atom(slab, id) for slab, id in zip(slabs, ids)]

        s = self

        def set_oc(oc):
            #print "Executing oc function"
            s.oc = float(oc)
            s.setoc(s.oc)
            other.setoc(s.oc)

        def get_oc():
            return s.oc

        setattr(out, 'setoc', set_oc)
        setattr(out, 'getoc', get_oc)

        return out

class Instrument:
    '''Class that keeps tracks of instrument settings.
    '''
    geometries = ['alpha_in fixed', 'alpha_in eq alpha_out',
                  'alpha_out fixed']
    def __init__(self, wavel, alpha, geom = 'alpha_in fixed',
                  flib = f, rholib = rho):
        '''Inits the instrument with default parameters
        '''
        self.flib = f
        self.rholib = rho
        self.set_wavel(wavel)
        self.set_geometry(geom)
        self.alpha = alpha
        self.inten = 1.0

    def set_inten(self, inten):
        '''Set the incomming intensity
        '''
        self.inten = inten

    def get_inten(self):
        '''retrieves the intensity
        '''
        return self.inten

    def set_wavel(self, wavel):
        '''Set the wavelength in AA
        '''
        try:
            self.wavel = float(wavel)
            self.flib.set_wavelength(wavel)
            self.rholib.set_wavelength(wavel)
        except ValueError:
            raise ValueError('%s is not a valid float number needed for the'
                             'wavelength'%(wavel))

    def get_wavel(self, wavel):
        '''Returns the wavelength in AA
        '''
        return self.wavel

    def set_energy(self, energy):
        '''Set the energy in keV
        '''
        try:
            self.set_wavel(12.39842/float(energy))
        except ValueError:
            raise ValueErrror('%s is not a valid float number needed for the'
                             'energy'%(wavel))
    def get_energy(self, energy):
        '''Returns the photon energy in keV
        '''
        return 12.39842/self.wavel

    def set_alpha(self, alpha):
        '''Sets the freezed angle. The meaning of this angle varies depening
        of the geometry parameter.

        geo =  "alpha_in fixed", alpha = alpha_in
        geo = "alpha_in eq alpha_out", alpha = alpha_in = alpha_out
        geo = "alpha_out fixed", alpha = alpha_out
        '''
        self.alpha = alpha

    def get_alpha(self):
        '''Gets the freexed angle. See set_alpha.
        '''
        return self.alpha

    def set_geometry(self, geom):
        '''Set the measurement geometry

        Should be one of the items in Instrument.geometry
        '''
        try:
            self.geom = self.geometries.index(geom)
        except ValueError:
            raise ValueError('The geometry  %s does not exist please choose'
                             'one of the following:\n%s'%(geom,
                                                          self.geomeries))
    def set_flib(self, flib):
        '''Set the structure factor library
        '''
        self.flib = flib

    def set_rholib(self, rholib):
        '''Set the rho library (electron density shape of the atoms)
        '''
        self.rholib = rholib

class SymTrans:
    def __init__(self, P = [[1,0],[0,1]], t = [0,0]):
        # TODO: Check size of arrays!
        self.P = np.array(P)
        self.t = np.array(t)

    def trans_x(self, x, y):
        '''transformed x coord
        '''
        #print self.P[0][0]*x + self.P[0][1]*y + self.t[0]
        return self.P[0][0]*x + self.P[0][1]*y + self.t[0]

    def trans_y(self, x, y):
        '''transformed x coord
        '''
        #print self.P[1][0]*x + self.P[1][1]*y + self.t[1]
        return self.P[1][0]*x + self.P[1][1]*y + self.t[1]

    def apply_symmetry(self, x, y):
        return np.dot(P, c_[x, y]) + t


#==============================================================================
# Utillity functions
def scale_sim(data, sim_list, scale_func = None):
    '''Scale the data according to a miminimazation of
    sum (data-I_list)**2
    '''
    numerator = sum([(data[i].y*sim_list[i]).sum() for i in range(len(data))
                 if data[i].use])
    denominator = sum([(sim_list[i]**2).sum() for i in range(len(data))
                 if data[i].use])
    scale = numerator/denominator
    print(scale)
    scaled_sim_list = [sim*scale for sim in sim_list]
    if not scale_func == None:
        scale_func(scale)
    return scaled_sim_list

def scale_sqrt_sim(data, sim_list, scale_func = None):
    '''Scale the data according to a miminimazation of
    sum (sqrt(data)-sqrt(I_list))**2
    '''
    numerator = sum([(np.sqrt(data[i].y*sim_list[i])).sum()
                     for i in range(len(data))
                 if data[i].use])
    denominator = sum([(sim_list[i]).sum() for i in range(len(data))
                 if data[i].use])
    scale = numerator/denominator
    scaled_sim_list = [sim*scale**2 for sim in sim_list]
    if not scale_func == None:
        scale_func(scale)
    return scaled_sim_list

## def scale_log_sim(data, sim_list):
##     '''Scale the data according to a miminimazation of
##     sum (log(data)-log(I_list))**2
##     '''
##     numerator = sum([(np.log10(data[i].y)*np.log10(sim_list[i])).sum()
##                      for i in range(len(data)) if data[i].use])
##     denominator = sum([(np.log10(sim_list[i])**2).sum()
##                       for i in range(len(data)) if data[i].use])
##     scale = numerator/denominator
##     print scale
##     scaled_sim_list = [sim*(10**-scale) for sim in sim_list]
##     return scaled_sim_list

def _get_f(inst, el, dinv):
    '''from the elements extract an array with atomic structure factors
    '''
    fdict = {}
    f = np.transpose(np.array([_fatom_eval(inst, fdict, elem, dinv/2.0)
                             for elem in el], dtype = np.complex128))

    return f

def _get_rho(inst, el):
    '''Returns the rho functions for all atoms in el
    '''
    rhos = [getattr(inst.rholib, elem) for elem in el]
    return rhos

def _fatom_eval(inst, f, element, s):
    '''Smart (fast) evaluation of f_atom. Only evaluates f if not
    evaluated before.

    element - element string
    f - dictonary for lookup
    s - sintheta_over_lambda array
    '''
    try:
        fret = f[element]
    except KeyError:
        fret = getattr(inst.flib, element)(s)
        f[element] = fret
            #print element, fret[0]
    return fret
#=============================================================================

if __name__ == '__main__':
    import models.sxrd_test5_sym_new_test_new66_2 as model
    from models.utils import UserVars
    import numpy as np
    from operator import mul
    from numpy.linalg import inv

    class domain_creator():
        def __init__(self,ref_domain,id_list,terminated_layer=0,domain_N=1,new_var_module=None,z_shift=0.):
            #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
            #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
            #function of set_new_vars
            self.ref_domain=ref_domain
            self.id_list=id_list
            self.terminated_layer=terminated_layer
            self.domain_N=domain_N
            self.new_var_module=new_var_module
            self.z_shift=z_shift
            self.domain_A,self.domain_B=self.create_equivalent_domains()

        def create_equivalent_domains(self):
            new_domain_A=self.ref_domain.copy()
            new_domain_B=self.ref_domain.copy()
            for id in self.id_list[:self.terminated_layer]:
                if id!=[]:
                    new_domain_A.del_atom(id)
            #number 5 here is crystal specific, here is the case for hematite
            for id in self.id_list[:self.terminated_layer+5]:
                new_domain_B.del_atom(id)

            return new_domain_A,new_domain_B

        def add_sorbates(self,domain,attach_atm_id=[['id1','id2']],el=['Pb'],id=[1],O_id=['_A'],r1=0.1,r2=None,alpha1=1.7,alpha2=None):
            #this function can add multiple sorbates
            #domain is a slab under consideration
            #attach_atm_id is a list of ids to be attached by absorbates,2 by n
            #el is list of element symbol for the first absorbates
            #id is the list of index number to be attached to elment symbol as the id symbol
            #O_id is list, each member will be attached at the end of id of the other absorbates
            #r1 alpha1 associated to the first absorbates, and r2 alpha2 associated to the other absorbates
            for i in range(len(el)):
                point1_x=domain.x[np.where(domain.id==attach_atm_id[i][0])[0][0]]
                point1_y=domain.y[np.where(domain.id==attach_atm_id[i][0])[0][0]]
                point1_z=domain.z[np.where(domain.id==attach_atm_id[i][0])[0][0]]
                point2_x=domain.x[np.where(domain.id==attach_atm_id[i][1])[0][0]]
                point2_y=domain.y[np.where(domain.id==attach_atm_id[i][1])[0][0]]
                point2_z=domain.z[np.where(domain.id==attach_atm_id[i][1])[0][0]]
                point1=[point1_x,point1_y,point1_z]
                point2=[point2_x,point2_y,point2_z]
                point_sorbate=self._cal_xyz_single(point1,point2,r1,alpha1)
                domain.add_atom(id=el[i]+str(id[i]),element=el[i],x=point_sorbate[0],y=point_sorbate[1],z=point_sorbate[2],u=1.)
                if r2!=None:
                    point_sorbate_1,point_sorbate_2=self._cal_xyz_double(point_sorbate,r2,alpha2)
                    domain.add_atom(id='Oi_1'+str(O_id[i]),element='O',x=point_sorbate_1[0],y=point_sorbate_1[1],z=point_sorbate_1[2],u=1.)
                    domain.add_atom(id='Oi_2'+str(O_id[i]),element='O',x=point_sorbate_2[0],y=point_sorbate_2[1],z=point_sorbate_2[2],u=1.)
            #return domain

        def add_oxygen_pair(self,domain,O_id,ref_point,r,alpha):
            #add single oxygen pair to a ref_point,which does not stand for an atom, the xyz for this point will be set as
            #three fitting parameters.O_id will be attached at the end of each id for the oxygen
            x_shift=r*np.cos(alpha)
            y_shift=r*np.sin(alpha)
            point1=ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]
            point2=ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]
            domain.add_atom(id='Os_1'+str(O_id),element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
            domain.add_atom(id='Os_2'+str(O_id),element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)

        def updata_oxygen_pair(self,domain,ids,ref_point,r,alpha):
            #updata the position information of oxygen pair, to be dropped inside sim func
            print('sensor',np.where(domain.id==ids[0]),np.where(domain.id==ids[0])[0])
            index_1=np.where(domain.id==ids[0])[0][0]
            index_2=np.where(domain.id==ids[1])[0][0]
            x_shift=r*np.cos(alpha)
            y_shift=r*np.sin(alpha)
            domain.x[index_1]=ref_point[0]+x_shift
            domain.y[index_1]=ref_point[1]+y_shift
            domain.z[index_1]=ref_point[2]
            domain.x[index_2]=ref_point[0]-x_shift
            domain.y[index_2]=ref_point[1]-y_shift
            domain.z[index_2]=ref_point[2]

        def group_sorbates_2(self,domain,attach_atm_id,ids_to_be_attached,r,alpha,beta,gamma):
            #updating the sorbate position, to be dropped inside sim function
            #the same as the group_sorbates except more freedome for the attached sorbates
            #r is the distance between Pb and one of O in this case, alpha is half of the open angle between the sorbates
            #beta is the angle between the normal line and the plane formed by three sorbates
            #gamma is then angle between the x axis and the first edge in the two dimentional space
            #alpha from 0-pi/2, beta from 0-pi/2, gamma from 0-2pi
            index_ref=np.where(domain.id==attach_atm_id)[0][0]
            index_1=np.where(domain.id==ids_to_be_attached[0])[0][0]
            index_2=np.where(domain.id==ids_to_be_attached[1])[0][0]
            ref_x=domain.x[index_ref]+domain.dx1[index_ref]+domain.dx2[index_ref]+domain.dx3[index_ref]
            ref_y=domain.y[index_ref]+domain.dy1[index_ref]+domain.dy2[index_ref]+domain.dy3[index_ref]
            ref_z=domain.z[index_ref]+domain.dz1[index_ref]+domain.dz2[index_ref]+domain.dz3[index_ref]
            z_shift=r*np.cos(alpha)*np.cos(beta)
            #r1 is the edge length of triangle inside the circle, alpha1 is the half open angle of that triangle
            r1=(r**2-z_shift**2)**0.5
            alpha1=np.arcsin(r*np.sin(alpha)/r1)
            point1_x_shift=r1*np.cos(gamma)
            point1_y_shift=r1*np.sin(gamma)
            point2_x_shift=r1*np.cos(gamma+2.*alpha1)
            point2_y_shift=r1*np.sin(gamma+2.*alpha1)
            domain.x[index_1]=ref_x+point1_x_shift
            domain.y[index_1]=ref_y+point1_y_shift
            domain.z[index_1]=ref_z+z_shift
            domain.x[index_2]=ref_x+point2_x_shift
            domain.y[index_2]=ref_y+point2_y_shift
            domain.z[index_2]=ref_z+z_shift

        def group_sorbates(self,domain,attach_atm_id,sorbate_ids,r1,alpha1,z_shift):
            #group the oxygen pair to the absorbate specified,attach_atm_id='Pb1',sorbate_ids=[]
            index_ref=np.where(domain.id==attach_atm_id)[0][0]
            index_1=np.where(domain.id==sorbate_ids[0])[0][0]
            index_2=np.where(domain.id==sorbate_ids[1])[0][0]
            ref_x=domain.x[index_ref]+domain.dx1[index_ref]+domain.dx2[index_ref]+domain.dx3[index_ref]
            ref_y=domain.y[index_ref]+domain.dy1[index_ref]+domain.dy2[index_ref]+domain.dy3[index_ref]
            ref_z=domain.z[index_ref]+domain.dz1[index_ref]+domain.dz2[index_ref]+domain.dz3[index_ref]
            O1_point,O2_point=self._cal_xyz_double(ref_point=[ref_x,ref_y,ref_z],r=r1,alpha=alpha1,z_shift=z_shift)
            domain.x[index_1],domain.y[index_1],domain.z[index_1]=O1_point[0],O1_point[1],O1_point[2]
            domain.x[index_2],domain.y[index_2],domain.z[index_2]=O2_point[0],O2_point[1],O2_point[2]

        def updata_sorbates(self,domain,id1,r1,alpha1,z_shift,attach_atm_id=['id1','id2'],id2=[],r2=None,alpha2=None):
            #old version of updating,less freedome for Pb sorbates
            #group all sorbates to the first layer oxygen pair
            #domain is a slab under consideration
            #id1 is the id for the first absorbate(Pb), r1 is positive value, alpha1 is angle lower than pi
            #attach_atm_id is a list of ids of first atoms(oxy)
            #id2 is a list of two pair absorbates, r2 is positive value, alpha2 is anlge less than pi
            index_1=np.where(domain.id==attach_atm_id[0])[0][0]
            index_2=np.where(domain.id==attach_atm_id[1])[0][0]
            point1_x=domain.x[index_1]+domain.dx1[index_1]+domain.dx2[index_1]+domain.dx3[index_1]
            point1_y=domain.y[index_1]+domain.dy1[index_1]+domain.dy2[index_1]+domain.dy3[index_1]
            point1_z=domain.z[index_1]+domain.dz1[index_1]+domain.dz2[index_1]+domain.dz3[index_1]
            point2_x=domain.x[index_2]+domain.dx1[index_2]+domain.dx2[index_2]+domain.dx3[index_2]
            point2_y=domain.y[index_2]+domain.dy1[index_2]+domain.dy2[index_2]+domain.dy3[index_2]
            point2_z=domain.z[index_2]+domain.dz1[index_2]+domain.dz2[index_2]+domain.dz3[index_2]

            point1=[point1_x,point1_y,point1_z]
            point2=[point2_x,point2_y,point2_z]
            point_sorbate=self._cal_xyz_single(point1,point2,r1,alpha1)
            domain.x[np.where(domain.id==id1)[0][0]]=point_sorbate[0]
            domain.y[np.where(domain.id==id1)[0][0]]=point_sorbate[1]
            domain.z[np.where(domain.id==id1)[0][0]]=point_sorbate[2]

            if r2!=None:
                point_sorbate_1,point_sorbate_2=self._cal_xyz_double(point_sorbate,r2,alpha2,z_shift)

                domain.x[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[0]
                domain.y[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[1]
                domain.z[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[2]

                domain.x[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[0]
                domain.y[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[1]
                domain.z[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[2]
            #return domain

        def _cal_xyz_single(self,point1,point2,r,alpha):
            #point1=[x1,y1,z1],point2=[x2,y2,z2],r is a value, alpha is angle less than pi
            slope_pt1_pt2=(point1[1]-point2[1])/(point1[0]-point2[0])
            slope_new1=-1./slope_pt1_pt2
            cent_point=[(point1[0]+point2[0])/2.,(point1[1]+point2[1])/2.]
            dist_pt12=((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5
            tan_theta=r*np.cos(alpha)/(dist_pt12/2.)
            slope_new2=(slope_pt1_pt2+tan_theta)/(1.-slope_pt1_pt2*tan_theta)
            #slope_new1 and cent_point form a line equation
            #slope_new2 and point2 form another line equation
            A=np.array([[-slope_new1,1.],[-slope_new2,1.]])
            C=np.array([cent_point[1]-slope_new1*cent_point[0],point2[1]-slope_new2*point2[0]])
            xy=np.dot(inv(A),C)
            return [xy[0],xy[1],point1[2]+r*np.sin(alpha)]

        def _cal_xyz_double(self,ref_point,r,alpha,z_shift=0.1):
        #ref_point=[x1,y1,z1],r is a positive value, alpha an angle less than pi, z_shift is positive value represent shift at z direction
            x_shift=r*np.cos(alpha)
            y_shift=r*np.sin(alpha)
            new_point1=[ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]+z_shift]
            new_point2=[2.*ref_point[0]-new_point1[0],2.*ref_point[1]-new_point1[1],ref_point[2]+z_shift]
            return new_point1,new_point2

        def grouping_sequence_layer(self, domain=[], first_atom_id=[],sym_file={},id_match_in_sym={},layers_N=1,use_sym=False):
            #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
            #so 4 atoms will group together if consider two chemical equivalent domain
            #domain is list of two chemical equivalent domains
            #first_atom_id is list of first id in id array of two domains
            #sym_file is a library of symmetry file names, the keys are element symbols
            #id_match_in_sym is a library of ids, the order of which match the symmetry operation in the associated sym file
            #layers_N is the number of layer you consider for grouping operation
            #use_sym is a flag to choose the shifting rule (symmetry basis or not)
            atm_gp_list=[]
            for i in range(layers_N):
                index_1=np.where(domain[0].id==first_atom_id[0])[0][0]+i*2
                temp_atm_gp=model.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=sym_file[str(domain[0].el[index_1])])
                temp_atm_gp.add_atom(domain[0],str(domain[0].id[index_1+1]))
                index_2=np.where(domain[1].id==first_atom_id[1])[0][0]+i*2
                temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2]))
                temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2+1]))
                atm_gp_list.append(temp_atm_gp)

            return atm_gp_list

        def grouping_discrete_layer(self,domain=[],atom_ids=[],sym_file=None,id_match_in_sym=[],use_sym=False):
            atm_gp=model.AtomGroup(id_in_sym_file=id_match_in_sym,filename=sym_file,use_sym=use_sym)
            for i in range(len(domain)):
                atm_gp.add_atom(domain[i],atom_ids[i])
            return atm_gp

        def scale_opt(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
            #scale the parameter from first layer atom to deeper layer atom
            #dx,dy,dz,u will decrease inward, oc decrease outward usually
            #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
            #atm_gp_list is a list of atom group to consider the scaling operation
            #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
            #flag is the parameter symbol
            #ref_v is the reference value to start off
            if sign_values==None:
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
            else:
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))

        def set_new_vars(self,head_list=['u_Fe_'],N_list=[2]):
        #set new vars
        #head_list is a list of heading test for a new variable,N_list is the associated number of each set of new variable to be created
            for head,N in zip(head_list,N_list):
                for i in range(N):
                    getattr(self.new_var_module,'new_var')(head+str(i+1),1.)

    ####################################################################
    unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
    inst = model.Instrument(wavel = .833, alpha = 2.0)
    bulk = model.Slab(T_factor='B')
    domain0 =  model.Slab(c = 1.0,T_factor='B')

    bulk.add_atom( "Fe2", "Fe", 0.00000e+00 ,     8.30000e-01 ,     8.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe3", "Fe", 5.00000e-01 ,     3.30000e-01 ,     8.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe4", "Fe", 5.00000e-01 ,     8.80000e-01 ,     6.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe6", "Fe", 0.00000e+00 ,     3.79000e-01 ,     6.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe8", "Fe", 0.00000e+00 ,     7.61000e-01 ,     3.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe9", "Fe", 5.00000e-01 ,     2.60000e-01 ,     3.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe10", "Fe", 5.00000e-01 ,     8.10000e-01 ,     1.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "Fe12", "Fe", 0.00000e+00 ,     3.10000e-01 ,     1.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O1", "O",  6.53000e-01 ,     9.73000e-01 ,     9.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O2", "O",  8.47000e-01 ,     4.73000e-01 ,     9.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O3", "O",  3.06000e-01 ,     6.05000e-01 ,     7.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O4", "O",  1.94000e-01 ,     1.04000e-01 ,     7.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O5", "O",  8.47000e-01 ,     7.37000e-01 ,     5.97000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O6", "O",  6.53000e-01 ,     2.36000e-01 ,     5.97000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O7", "O",  3.47000e-01 ,     9.04000e-01 ,     4.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O8", "O",  1.53000e-01 ,     4.03000e-01 ,     4.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O9", "O",  6.94000e-01 ,     5.35000e-01 ,     2.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O10", "O",  8.06000e-01 ,     3.50000e-02 ,     2.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O11", "O",  1.53000e-01 ,     6.67000e-01 ,     9.70000e-02 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    bulk.add_atom( "O12", "O",  3.47000e-01 ,     1.67000e-01 ,     9.70000e-02 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    #domain0 here is a reference domain, the atoms are ordered according to hight (z values)
    #it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted
    domain0.add_atom( "O1_1_0", "O",  6.53000e-01 ,     1.11210e+00 ,     1.90300e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_2_0", "O",  8.47000e-01 ,     6.12100e-01 ,     1.90300e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_2_0", "Fe", 0.00000e+00 ,     9.69100e-01 ,     1.85500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1. )
    domain0.add_atom( "Fe1_3_0", "Fe", 5.00000e-01 ,     4.69100e-01 ,     1.85500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1. )
    domain0.add_atom( "O1_3_0", "O",  3.06000e-01 ,     7.44100e-01 ,     1.75000e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_4_0", "O",  1.94000e-01 ,     2.43100e-01 ,     1.75000e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_4_0", "Fe", 5.00000e-01 ,     1.01910e+00 ,     1.64500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_6_0", "Fe", 0.00000e+00 ,     5.18100e-01 ,     1.64500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_5_0", "O",  8.47000e-01 ,     8.76100e-01 ,     1.59700e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_6_0", "O",  6.53000e-01 ,     3.75100e-01 ,     1.59700e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_7_0", "O",  3.47000e-01 ,     1.04310e+00 ,     1.40300e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_8_0", "O",  1.53000e-01 ,     5.42100e-01 ,     1.40300e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_8_0", "Fe", 0.00000e+00 ,     9.00100e-01 ,     1.35500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_9_0", "Fe", 5.00000e-01 ,     3.99100e-01 ,     1.35500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_9_0", "O",  6.94000e-01 ,     6.74100e-01 ,     1.25000e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_10_0", "O",  8.06000e-01 ,     1.74100e-01 ,     1.25000e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_10_0", "Fe", 5.00000e-01 ,     9.49100e-01 ,     1.14500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe1_12_0", "Fe", 0.00000e+00 ,     4.49100e-01 ,     1.14500e+00 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_11_0", "O",  1.53000e-01 ,     8.06100e-01 ,     1.09700e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O1_12_0", "O",  3.47000e-01 ,     3.06100e-01 ,     1.09700e+00 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )

    domain0.add_atom( "O1_0", "O",  6.53000e-01 ,     9.73000e-01 ,     9.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O2_0", "O",  8.47000e-01 ,     4.73000e-01 ,     9.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe2_0", "Fe", 0.00000e+00 ,     8.30000e-01 ,     8.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1. )
    domain0.add_atom( "Fe3_0", "Fe", 5.00000e-01 ,     3.30000e-01 ,     8.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1. )
    domain0.add_atom( "O3_0", "O",  3.06000e-01 ,     6.05000e-01 ,     7.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O4_0", "O",  1.94000e-01 ,     1.04000e-01 ,     7.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe4_0", "Fe", 5.00000e-01 ,     8.80000e-01 ,     6.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe6_0", "Fe", 0.00000e+00 ,     3.79000e-01 ,     6.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O5_0", "O",  8.47000e-01 ,     7.37000e-01 ,     5.97000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O6_0", "O",  6.53000e-01 ,     2.36000e-01 ,     5.97000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O7_0", "O",  3.47000e-01 ,     9.04000e-01 ,     4.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O8_0", "O",  1.53000e-01 ,     4.03000e-01 ,     4.03000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe8_0", "Fe", 0.00000e+00 ,     7.61000e-01 ,     3.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe9_0", "Fe", 5.00000e-01 ,     2.60000e-01 ,     3.55000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O9_0", "O",  6.94000e-01 ,     5.35000e-01 ,     2.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O10_0", "O",  8.06000e-01 ,     3.50000e-02 ,     2.50000e-01 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe10_0", "Fe", 5.00000e-01 ,     8.10000e-01 ,     1.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "Fe12_0", "Fe", 0.00000e+00 ,     3.10000e-01 ,     1.45000e-01 ,     3.20000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O11_0", "O",  1.53000e-01 ,     6.67000e-01 ,     9.70000e-02 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )
    domain0.add_atom( "O12_0", "O",  3.47000e-01 ,     1.67000e-01 ,     9.70000e-02 ,     3.30000e-01 ,     1.00000e+00 ,     1.00000e+00 )

    #id list according to the order in the reference domain
    ref_id_list=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
    "O_1_0","O_2_0","Fe_2_0","Fe_3_0","O_3_0","O_4_0","Fe_4_0","Fe_6_0","O_5_0","O_6_0","O_7_0","O_8_0","Fe_8_0","Fe_9_0","O_9_0","O_10_0","Fe_10_0","Fe_12_0","O_11_0","O_12_0"]
    #the matching row Id information in the symfile
    sym_file_Fe=np.array(['Fe1_0','Fe2_0','Fe3_0','Fe4_0','Fe5_0','Fe6_0','Fe7_0','Fe8_0','Fe9_0','Fe10_0','Fe11_0','Fe12_0',\
        'Fe1_1_0','Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_5_0','Fe1_6_0','Fe1_7_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_11_0','Fe1_12_0'])
    sym_file_O=np.array(['O1_0','O2_0','O3_0','O4_0','O5_0','O6_0','O7_0','O8_0','O9_0','O10_0','O11_0','O12_0',\
        'O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0'])
    #create a domain class and initiate the chemical equivalent domains
    rgh_domain1=UserVars()
    domain_class_1=domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,domain_N=1,new_var_module=rgh_domain1)
    domain1A=domain_class_1.domain_A
    domain1B=domain_class_1.domain_B

    #add sorbates for two domains
    domain_class_1.add_sorbates(domain=domain1A,attach_atm_id=[['O1_1_0','O1_2_0'],['O1_3_0','O1_4_0']],el=['Pb','Pb'],id=[1,11],O_id=['_A','_AA'],r1=0.1,r2=0.1,alpha1=np.pi/2.,alpha2=0.)
    domain_class_1.add_sorbates(domain=domain1B,attach_atm_id=[['O1_7_0','O1_8_0'],['O1_9_0','O1_10_0']],el=['Pb','Pb'],id=[2,22],O_id=['_B','_BB'],r1=0.1,r2=0.1,alpha1=np.pi/2.,alpha2=0.)
    #add lone oxygen pair on top
    domain_class_1.add_oxygen_pair(domain1A,O_id='_A1',ref_point=[0.5,0.5,2.203],r=0.1,alpha=0.)
    domain_class_1.add_oxygen_pair(domain1B,O_id='_B1',ref_point=[0.5,0.5,1.703],r=0.1,alpha=0.)
    #set new variables
    domain_class_1.set_new_vars(head_list=['u_o_n','u_Fe_n','dx_n','dy_n','dz_n','oc_n','dx_sign_n','dy_sign_n','dz_sign_n'],N_list=[4,3,7,7,7,7,7,7,7])
    #some other parameters to be used
    rgh_domain1.new_var('r_Pb_O', 0.1)
    rgh_domain1.new_var('r_Pb_O2', 0.1)
    rgh_domain1.new_var('r_O_pair1', 0.1)
    rgh_domain1.new_var('alpha_O_pair1', np.pi/4.)
    rgh_domain1.new_var('alpha_Pb_O', np.pi/4.)
    rgh_domain1.new_var('beta_Pb_O', np.pi/4.)
    rgh_domain1.new_var('gamma_Pb_O', np.pi/4.)
    rgh_domain1.new_var('alpha_Pb_O2', np.pi/4.)
    rgh_domain1.new_var('beta_Pb_O2', np.pi/4.)
    rgh_domain1.new_var('gamma_Pb_O2', np.pi/4.)
    rgh_domain1.new_var('ref_x_O_pair1', 0.5)
    rgh_domain1.new_var('ref_y_O_pair1', 0.5)
    rgh_domain1.new_var('ref_z_O_pair1', 2.203)

    rgh_domain1.new_var('domain_wt', 0.)
    rgh_domain1.new_var('beta', 0.)
    #do grouping for top seven layers
    atm_gp_list_domain1=domain_class_1.grouping_sequence_layer(domain=[domain1A,domain1B], first_atom_id=['O1_1_0','O1_7_0'],\
        sym_file={'Fe':'Fe0 output file for Genx reading.txt','O':'O0 output file for Genx reading.txt'},\
        id_match_in_sym={'Fe':sym_file_Fe,'O':sym_file_O},layers_N=7,use_sym=True)
    #the first atom group will be the reference group for scaling operation of dx dy dz
    ref_atm_gp_domain1=atm_gp_list_domain1[0]
    #group the sorbate of Pb
    atm_gp_Pb_domain1=domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1B],atom_ids=['Pb1','Pb2'])
    atm_gp_Pb2_domain1=domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1B],atom_ids=['Pb11','Pb22'])

    #Group sorbates of Oxygen pair
    atm_gp_O_domain1=domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1A,domain1B,domain1B],atom_ids=['Oi_1_A','Oi_2_A','Oi_1_B','Oi_2_B'])
    atm_gp_O2_domain1=domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1A,domain1B,domain1B],atom_ids=['Oi_1_AA','Oi_2_AA','Oi_1_BB','Oi_2_BB'])
    atm_gp_Os1_domain1=domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1A,domain1B,domain1B],atom_ids=['Os_1_A1','Os_2_A1','Os_1_B1','Os_2_B1'])

    #make a domain libratry wrapping two chemical equivalent domains
    domain={'domain1A':{'slab':domain1A,'wt':1.},'domain1B':{'slab':domain1B,'wt':0.}}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})

    def extract_list(ref_list,extract_index):
        output_list=[]
        for i in extract_index:
            output_list.append(ref_list[i])
        return output_list

    def norm_sign(value,scale=1.):
        if value<=0.5:
            return -scale
        elif value>0.5:
            return scale

    def Sim(data):
    #scale the thermal factor (1-2), note the scaling will be done from deepest layer, so here the list extraction is done from inside [6,3,1]
        scale_values_Fe_u=[rgh_domain1.u_Fe_n1,rgh_domain1.u_Fe_n2,rgh_domain1.u_Fe_n3]
        scale_values_O_u=[rgh_domain1.u_o_n1,rgh_domain1.u_o_n2,rgh_domain1.u_o_n3,rgh_domain1.u_o_n4]
        domain_class_1.scale_opt(extract_list(atm_gp_list_domain1,[6,3,1]),scale_factor=scale_values_Fe_u,sign_values=None,flag='u',ref_v=0.32)
        domain_class_1.scale_opt(extract_list(atm_gp_list_domain1,[5,4,2,0]),scale_factor=scale_values_O_u,sign_values=None,flag='u',ref_v=0.4)

    #scale the occupancy (0.5-1), scaling was done outward, so reverse the atom group list here
        scale_values_all_oc=[rgh_domain1.oc_n1,rgh_domain1.oc_n2,rgh_domain1.oc_n3,rgh_domain1.oc_n4,rgh_domain1.oc_n5,rgh_domain1.oc_n6,rgh_domain1.oc_n7]
        domain_class_1.scale_opt(atm_gp_list_domain1[::-1],scale_factor=scale_values_all_oc,sign_values=None,flag='oc',ref_v=1.)

    #extract reference dxdydz from reference atom group
        ref_dx_domain1=getattr(ref_atm_gp_domain1,'getdx')()
        ref_dy_domain1=getattr(ref_atm_gp_domain1,'getdy')()
        ref_dz_domain1=getattr(ref_atm_gp_domain1,'getdz')()
    #scale dx value(0.1-1), the extra value in norm_sign is a second scaling factor for dxdy compared to dz
    #which is believed to be more likely to relax than dxdy
    #fit the shift amount for the first layer oxygen, and scale the shift for the other deeper layers from n2 to n7
        scale_values_all_dx=[rgh_domain1.dx_n2,rgh_domain1.dx_n3,rgh_domain1.dx_n4,rgh_domain1.dx_n5,rgh_domain1.dx_n6,rgh_domain1.dx_n7]
        sign_values_all_dx=[norm_sign(rgh_domain1.dx_sign_n2,0.1),norm_sign(rgh_domain1.dx_sign_n3,0.05),\
                            norm_sign(rgh_domain1.dx_sign_n4,0.01),norm_sign(rgh_domain1.dx_sign_n5,0.001),norm_sign(rgh_domain1.dx_sign_n6,0.0001),norm_sign(rgh_domain1.dx_sign_n7,0.00001)]
        domain_class_1.scale_opt(atm_gp_list_domain1[1:],scale_factor=scale_values_all_dx,sign_values=sign_values_all_dx,flag='dx',ref_v=ref_dx_domain1)

    #scale dy value(0.1-1)
        scale_values_all_dy=[rgh_domain1.dy_n2,rgh_domain1.dy_n3,rgh_domain1.dy_n4,rgh_domain1.dy_n5,rgh_domain1.dy_n6,rgh_domain1.dy_n7]
        sign_values_all_dy=[norm_sign(rgh_domain1.dy_sign_n2,0.1),norm_sign(rgh_domain1.dy_sign_n3,0.05),\
                            norm_sign(rgh_domain1.dy_sign_n4,0.01),norm_sign(rgh_domain1.dy_sign_n5,0.001),norm_sign(rgh_domain1.dy_sign_n6,0.0001),norm_sign(rgh_domain1.dy_sign_n7,0.00001)]
        domain_class_1.scale_opt(atm_gp_list_domain1[1:],scale_factor=scale_values_all_dy,sign_values=sign_values_all_dy,flag='dy',ref_v=ref_dy_domain1)

    #scale dz value(0.1-1)
        scale_values_all_dz=[rgh_domain1.dz_n2,rgh_domain1.dz_n3,rgh_domain1.dz_n4,rgh_domain1.dz_n5,rgh_domain1.dz_n6,rgh_domain1.dz_n7]
        sign_values_all_dz=[norm_sign(rgh_domain1.dz_sign_n1),norm_sign(rgh_domain1.dz_sign_n2),norm_sign(rgh_domain1.dz_sign_n3),\
                            norm_sign(rgh_domain1.dz_sign_n4),norm_sign(rgh_domain1.dz_sign_n5),norm_sign(rgh_domain1.dz_sign_n6),norm_sign(rgh_domain1.dz_sign_n7)]
        domain_class_1.scale_opt(atm_gp_list_domain1[1:],scale_factor=scale_values_all_dz,sign_values=sign_values_all_dz,flag='dz',ref_v=ref_dz_domain1)

    #updata sorbate xyz (bidentate configuration here)
        domain_class_1.group_sorbates_2(domain=domain1A,attach_atm_id='Pb1',ids_to_be_attached=['Oi_1_A','Oi_2_A'],r=rgh_domain1.r_Pb_O,alpha=rgh_domain1.alpha_Pb_O,beta=rgh_domain1.beta_Pb_O,gamma=rgh_domain1.gamma_Pb_O)
        domain_class_1.group_sorbates_2(domain=domain1B,attach_atm_id='Pb2',ids_to_be_attached=['Oi_1_B','Oi_2_B'],r=rgh_domain1.r_Pb_O,alpha=rgh_domain1.alpha_Pb_O,beta=rgh_domain1.beta_Pb_O,gamma=rgh_domain1.gamma_Pb_O)
        domain_class_1.group_sorbates_2(domain=domain1A,attach_atm_id='Pb11',ids_to_be_attached=['Oi_1_AA','Oi_2_AA'],r=rgh_domain1.r_Pb_O2,alpha=rgh_domain1.alpha_Pb_O2,beta=rgh_domain1.beta_Pb_O2,gamma=rgh_domain1.gamma_Pb_O2)
        domain_class_1.group_sorbates_2(domain=domain1B,attach_atm_id='Pb22',ids_to_be_attached=['Oi_1_BB','Oi_2_BB'],r=rgh_domain1.r_Pb_O2,alpha=rgh_domain1.alpha_Pb_O2,beta=rgh_domain1.beta_Pb_O2,gamma=rgh_domain1.gamma_Pb_O2)
        domain_class_1.updata_oxygen_pair(domain=domain1A,ids=['Os_1_A1','Os_2_A1'],ref_point=[rgh_domain1.ref_x_O_pair1,rgh_domain1.ref_y_O_pair1,rgh_domain1.ref_z_O_pair1],r=rgh_domain1.r_O_pair1,alpha=rgh_domain1.alpha_O_pair1)
        domain_class_1.updata_oxygen_pair(domain=domain1B,ids=['Os_1_B1','Os_2_B1'],ref_point=[rgh_domain1.ref_x_O_pair1,rgh_domain1.ref_y_O_pair1,rgh_domain1.ref_z_O_pair1-0.5],r=rgh_domain1.r_O_pair1,alpha=rgh_domain1.alpha_O_pair1)

        #roughness par
        beta=rgh_domain1.beta
        F = []
        domain['domain1A']['wt']=1.-rgh_domain1.domain_wt
        domain['domain1B']['wt']=rgh_domain1.domain_wt
        #9.a loop through the data sets
        for data_set in data:
            # 9.b create all the h,k,l values for the rod (data_set)
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            # 9.c. calculate roughness using beta model
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l - LB)/dL)**2)**0.5
            # 9.d. Calculate the structure factor
            f = rough*sample.calc_f(h, k, l)
            # 9.e Calculate |F|
            i = abs(f)
            # 9.f Append the calculated intensity to the list I
            F.append(i)

        return F
