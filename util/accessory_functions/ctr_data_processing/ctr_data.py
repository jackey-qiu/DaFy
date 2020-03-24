"""
Original ctr_data module developed by T. Trainor and F. Heberling
There is no longer a CTR Data object
The contents of this module are now only responsible for handling correction factors
All plotting and CTR data handling are now done in the wxIntegrator developed by C. Biwer
J. Stubbs April 2015

The key references for how the geometric
corrections are applied include:

* E. Vlieg, J. Appl. Cryst. (1997). 30, 532-543
* C. Schlepuetz et al, Acta Cryst. (2005). A61, 418-425
"""
##############################################################################

import types, copy
import numpy as num
import time

from mathutil import cosd, sind

from active_area import active_area
import gonio_psic 

DEBUG = False

##############################################################################


   


##############################################################################
def image_point_F(scan,point,I='I',Inorm='io',Ierr='Ierr',Ibgr='Ibgr', transm='transm',
                  corr_params={}, preparsed=False):
    """
    compute F for a single scan point in an image scan
    """
    d = {'I':0.0,'Inorm':0.0,'Ierr':0.0,'Ibgr':0.0, 'transm':0.0, 'F':0.0,'Ferr':0.0,
         'ctot':1.0,'alpha':0.0,'beta':0.0}
    d['I']     = scan[I][point]
    d['Inorm'] = scan[Inorm][point]
    d['Ierr']  = scan[Ierr][point]
    d['Ibgr']  = scan[Ibgr][point]
    d['transm'] = scan[transm][point]
    
    if corr_params == None:
        d['ctot'] = 1.0
        scale = 1.0
    else:
        # compute correction factors
        scale  = corr_params.get('scale')
        if scale == None: scale = 1.
        scale  = float(scale)
        corr = _get_corr(scan,point,corr_params, preparsed)
        if corr == None:
            d['ctot'] = 1.0
        else:
            d['ctot']  = corr.ctot_stationary()
            d['alpha'] = corr.gonio.pangles['alpha']
            d['beta']  = corr.gonio.pangles['beta']

    # compute F
    if d['I'] <= 0.0 or d['Inorm'] <= 0.:
        d['F']    = 0.0
        d['Ferr'] = 0.0
    else:
        scale = scale/d['transm'] * d['ctot']/d['Inorm']
        #scale = scale * d['ctot']/d['Inorm']
        d['F']    = num.sqrt(scale*d['I'])
        d['Ferr'] = 0.5 * scale**0.5 * d['Ierr']/d['I']**0.5
    return d

##############################################################################
def _get_corr(scan,point,corr_params,preparsed=False):
    """
    get CtrCorrection instance
    """
    geom   = corr_params.get('geom','psic')
    beam   = corr_params.get('beam_slits',{})
    det    = corr_params.get('det_slits')
    sample = corr_params.get('sample')
    # get gonio instance for corrections
    if geom == 'psic':
        gonio = gonio_psic.psic_from_spec(scan['G'],preparsed=preparsed)
        _update_psic_angles(gonio,scan,point)
        corr  = CtrCorrectionPsic(gonio=gonio,beam_slits=beam,
                                  det_slits=det,sample=sample)
    else:
        print "Geometry %s not implemented" % geom
        corr = None
    return corr

##############################################################################
def _update_psic_angles(gonio,scan,point,verbose=True):
    """
    given a psic gonio instance, a scandata object
    and a scan point, update the gonio angles...
    """
    try:
        npts = int(scan.dims[0])
    except:
        npts = scan.get('dims', (1,0))[0]
    try: 
        scan_name = scan.name
    except: 
        scan_name = ''
    #
    try:
      if type(scan['phi']) == types.FloatType:
          phi=scan['phi']
      elif len(scan['phi']) == npts:
          phi=scan['phi'][point]
    except:
        phi=None
    if phi == None and verbose==True:
        print "Warning no phi angle:", scan_name
    #
    try:
        if type(scan['chi']) == types.FloatType:
            chi=scan['chi']
        elif len(scan['chi']) == npts:
            chi=scan['chi'][point]
    except:
        chi = None
    if chi == None and verbose==True:
        print "Warning no chi angle", scan_name
    #
    try:
        if type(scan['eta']) == types.FloatType:
            eta=scan['eta']
        elif len(scan['eta']) == npts:
            eta=scan['eta'][point]
    except:
        eta = None
    if eta == None and verbose==True:
        print "Warning no eta angle", scan_name
    #
    try:
        if type(scan['mu']) == types.FloatType:
            mu=scan['mu']
        elif len(scan['mu']) == npts:
            mu=scan['mu'][point]
    except:
        mu = None
    if mu == None and verbose==True:
        print "Warning no mu angle", scan_name
    #
    try:
        if type(scan['nu']) == types.FloatType:
            nu=scan['nu']
        elif len(scan['nu']) == npts:
            nu=scan['nu'][point]
    except:
        nu = None
    if nu == None and verbose==True:
        print "Warning no nu angle", scan_name
    #
    try:
        if type(scan['del']) == types.FloatType:
            delta=scan['del']
        elif len(scan['del']) == npts:
            delta=scan['del'][point]
    except:
        delta = None
    if delta == None and verbose==True:
        print "Warning no del angle", scan_name
    #
    gonio.set_angles(phi=phi,chi=chi,eta=eta,
                     mu=mu,nu=nu,delta=delta)

##############################################################################
class CtrCorrectionPsic:
    """
    Data point operations / corrections for Psic geometry

    Notes:
    ------
    All correction factors are defined such that the
    measured data is corrected by multiplying times
    the correction: 
      Ic  = Im*ct
    where
      Im = Idet/Io = uncorrected (measured) intensity

    In other words we use the following formalism:
      Im = (|F|**2)* prod_i(Xi)
    where Xi are various (geometric) factors that 
    influence the measured intensity.  To get the
    structure factor:
      |F| = sqrt(Im/prod_i(Xi)) = sqrt(Im* ct)
    and
      ct = prod_i(1/Xi) = prod_i(ci)
      ci = 1/Xi
      
    If there is an error or problem in the routine for a specific
    correction factor, (e.g. divide by zero), the routine should
    return a zero.  This way the corrected data is zero'd....

    * The correction factors depend on the goniometer geometry
      gonio = gonio_psic.Psic instance

    * The slits settings are needed.  Note if using a large area detector
      you may pass det_slits = None and just spill off will be computed
        beam_slits = {'horz':.6,'vert':.8}
        det_slits = {'horz':20.0,'vert':10.5}
      these are defined wrt psic phi-frame:
        horz = beam/detector horz width (total slit width in lab-z,
               or the horizontal scattering plane)
        vert = detector vert hieght (total slit width in lab-x,
               or the vertical scattering plane)

    * A sample description is needed.
      sample = {}
        sample['dia'] = is taken as the diameter of a round sample
                        mounted on center. if dia<=0 then we either use
                        the polygon description or ignore the sample.
        sample['polygon'] = [[1.,1.], [.5,1.5], [-1.,1.],
                             [-1.,-1.],[0.,.5],[1.,-1.]]
        sample['angles']  = {'phi':108.0007,'chi':0.4831}

        polygon = [[x,y,z],[x,y,z],[x,y,z],....]
                  is a list of vectors that describe the shape of
                  the sample.  They should be given in general lab
                  frame coordinates.

        angles = {'phi':0.,'chi':0.,'eta':0.,'mu':0.}
                 are the instrument angles at which the sample
                 vectors were determined.

    Note: the lab frame coordinate systems is defined such that:
    x is vertical (perpendicular, pointing to the ceiling of the hutch)
    y is directed along the incident beam path
    z make the system right handed and lies in the horizontal scattering plane
    (i.e. z is parallel to the phi axis)

    The center (0,0,0) of the lab frame is the rotation center of the instrument.

    If the sample vectors are given at the flat phi and chi values and with
    the correct sample hieght (sample Z set so the sample surface is on the
    rotation center), then the z values of the sample vectors will be zero.
    If 2D vectors are passed we therefore assume these are [x,y,0].  If this
    is the case then make sure:
    angles = {'phi':flatphi,'chi':flatchi,'eta':0.,'mu':0.}

    The easiest way to determine the sample coordinate vectors is to take a picture
    of the sample with a camera mounted such that is looks directly down the omega
    axis and the gonio angles set at the sample flat phi and chi values and
    eta = mu = 0. Then find the sample rotation center and measure the position
    of each corner (in mm) with up being the +x direction, and downstream
    being the +y direction.  

    Note this routine does not correct for attenuation factors.  
    
    """
    def __init__(self,gonio=None,beam_slits={},det_slits=None,sample={}):
        """
        Initialize

        Parameters:
        -----------
        * gonio is a goniometer instance used for computing reciprocal
          lattice indicies and psuedo angles from motor angles
        * beam_slits are dictionary defining the incident beam aperature
        * det_slits are a dictionary defining the detector aperature
        * sample is a dictionary describing the sample geometry
        (see the instance documentation for more details)
        """
        self.gonio      = gonio
        if self.gonio.calc_psuedo == False:
            self.gonio.calc_psuedo = True
            self.gonio._update_psuedo()
        self.beam_slits = beam_slits
        self.det_slits  = det_slits
        self.sample     = sample
        # fraction horz polarization
        self.fh         = 1.0

    ##########################################################################
    def ctot_stationary(self,plot=False,fig=None):
        """
        correction factors for stationary measurements (e.g. images)
        """
        cp = self.polarization()
        cl = self.lorentz_stationary()
        ca = self.active_area(plot=plot,fig=fig)
        ct = (cp)*(cl)*(ca)
        if plot == True:
            print "Correction factors (mult by I)" 
            print "   Polarization=%f" % cp
            print "   Lorentz=%f" % cl
            print "   Area=%f" % ca
            print "   Total=%f" % ct
        return ct

    ##########################################################################
    def lorentz_stationary(self):
        """
        Compute the Lorentz factor for a stationary (image)
        measurement.  See Vlieg 1997

        Measured data is corrected for Lorentz factor as: 
          Ic  = Im * cl
        """
        beta  = self.gonio.pangles['beta']
        cl = sind(beta)
        return cl

    ##########################################################################
    def polarization(self,):
        """
        Compute polarization correction factor.
        
        For a horizontally polarized beam (polarization vector
        parrallel to the lab-frame z direction) the polarization
        factor is normally defined as:
           p = 1-(cos(del)*sin(nu))^2
        For a beam with mixed horizontal and vertical polarization:
           p = fh( 1-(cos(del)*sin(nu))^2 ) + (1-fh)(1-sin(del)^2)
        where fh is the fraction of horizontal polarization.

        Measured data is corrected for polarization as: 
          Ic  = Im * cp = Im/p
        """
        fh    = self.fh
        delta = self.gonio.angles['delta']
        nu    = self.gonio.angles['nu']
        p = 1. - ( cosd(delta) * sind(nu) )**2.
        if fh != 1.0:
            p = fh * c_p + (1.-fh)*(1.0 - (sind(delta))**2.)
        if p == 0.:
            cp = 0.
        else:
            cp = 1./p

        return cp

    ##########################################################################
    def active_area(self,plot=False,fig=None):
        """
        Compute active area correction (c_a = A_beam/A_int**2)
        
        Use to correct scattering data for area effects,
        including spilloff, A_int/A_beam and normailization 
        to unit surface area (1/A_beam), i.e.
            Ic = Im * ca = Im/A_ratio 
            A_ratio = A_int/(A_beam**2) 
        where
            A_int = intersection area (area of beam on sample
                    viewed by detector)
            A_beam = total beam area
        """
        if self.beam_slits == {} or self.beam_slits == None:
            print "Warning beam slits not specified"
            return 1.0
        alpha = self.gonio.pangles['alpha']
        beta  = self.gonio.pangles['beta']
        if plot == True:
            print 'Alpha = ', alpha, ', Beta = ', beta
        if alpha < 0.0:
            print 'alpha is less than 0.0'
            return 0.0
        elif beta < 0.0:
            print 'beta is less than 0.0'
            return 0.0

        # get beam vectors
        bh = self.beam_slits['horz']
        bv = self.beam_slits['vert']
        beam = gonio_psic.beam_vectors(h=bh,v=bv)

        # get det vectors
        if self.det_slits == None:
            det = None
        else:
            dh = self.det_slits['horz']
            dv = self.det_slits['vert']
            det  = gonio_psic.det_vectors(h=dh,v=dv,
                                          nu=self.gonio.angles['nu'],
                                          delta=self.gonio.angles['delta'])
        # get sample poly
        if type(self.sample) == types.DictType:
            sample_dia    = self.sample.get('dia',0.)
            sample_vecs   = self.sample.get('polygon',None)
            sample_angles = self.sample.get('angles',{})
            #
            if sample_vecs != None and sample_dia <= 0.:
                sample = gonio_psic.sample_vectors(sample_vecs,
                                                   angles=sample_angles,
                                                   gonio=self.gonio)
            elif sample_dia > 0.:
                sample = sample_dia
            else:
                sample = None
        else:
            sample = self.sample

        # compute active_area
        (A_beam,A_int) = active_area(self.gonio.nm,ki=self.gonio.ki,
                                     kr=self.gonio.kr,beam=beam,det=det,
                                     sample=sample,plot=plot,fig=fig)
        if A_int == 0.:
            ca = 0.
        else:
            ca = A_beam/(A_int**2)
            
        return ca

##############################################################################



