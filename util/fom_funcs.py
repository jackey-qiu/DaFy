'''<h1> Figure of Merit (FOM)</h1>
The Figure of Merit (FOM) is the function that compares how well the simulation matches the measured data. Strictly speaking, for Gaussian errors, a chi squared (&chi;<sup>2</sup>) FOM is the most appropriate. However, the world is not perfect and many times the data can be fitted more easily and more robustly if another FOM is chosen. Each FOM function has its merits and drawbacks, and fitting can rely critically on choosing the right FOM function for the particular data to be analyzed. The following gives a brief summary and explanation of the FOMs included in the standard GenX distribution so far.<br>
It is also possible to create custom FOM functions to be used by GenX. For more information on this refer to the Section "Customization" below.<br>


<h2>Available FOM functions</h2>
In the following, the merged data set consisting of all data sets
that are marked for use is denoted as <var>Y</var> and the corresponding
simulation is denoted as <var>S</var>. A single element of these arrays
is indicated by a subscript <var>i</var>. In the same manner, the
independent variable (denoted as <var>x</var> in the data strucure) is called
<var>X</var>. The error array is denoted <var>E</var>. Finally the total number
of data points is given by <var>N</var> and <var>p</p> is the number of free parameters
in the fit.<br>


<h3>Unweighted FOM functions</h3>


<h4>diff</h4>
Average of the absolute difference between simulation and data.<br>
<br><huge>
    FOM<sub>diff</sub> =  1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124;
</huge><br>


<h4>log</h4>
Average of the absolute difference between the logarithms (base 10) of the data and the simulation.<br>
<br><huge>
    FOM<sub>log</sub> = 1/(N-p) &times;&#8721;<sub><var>i</var></sub>
    &#124;log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)&#124;
</huge><br>


<h4>sqrt</h4>
Average of the absolute difference between the square roots of the data and the simulation:<br>
<br><huge>
    FOM<sub>sqrt</sub> =  1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
    &#124;
</huge><br>


<h4>R1</h4>
Crystallographic R-factor (often denoted as R1, sometimes called residual factor or reliability factor or the R-value or R<sub>work</sub>).<br>
Gives the percentage of the summed structure factor residuals (absolute difference between data and simulation) over the entire data set with respect to the total sum of measured structure factors. For data sets spanning several orders of magnitude in intensity, R1 is dominated by the residuals at high intensities, while large residuals at low intensities have very little impact on R1.
This implementation here assumes that the loaded data are intensities (squares of the structure factors), hence the square roots of the loaded data are taken for the calculation of R1.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976)]<br>
<br><huge>
  FOM<sub>R1</sub> =
  &#8721;<sub><var>i</var></sub> [
  &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
  &#124; ] / &#8721;<sub><var>i</var></sub> [ sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>logR1</h4>
The logarithmic R1 factor is a modification of the crystallographic R-factor, calculated using the logarithm (base 10) of the structure factor and simulation. This scaling results in a more equal weighting of high-intensity and low-intensity data points which can be very helpful when fitting data which is spanning several orders of magnitude on the y-axis. Essentially it gives all data points equal weight when displayed in a log-plot.<br>
<br><huge>
    FOM<sub>logR1</sub> =
    &#8721;<sub><var>i</var></sub> [ &#124;
    log<sub>10</sub>(sqrt(<var>Y<sub>i</sub></var>)) -
    log<sub>10</sub>(sqrt(<var>S<sub>i</sub></var>))
    &#124; ] /
    &#8721;<sub><var>i</var></sub> [
    log<sub>10</sub>(sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>R2</h4>
Crystallographic R2 factor. In contrast to R1, this gives the ratio of the total sum of squared deviations to the total sum of squared structure factors. (Note that sometimes R2 is also defined as the square root of the value defined here.)
Like in the case for R1, this implementation assumes that the loaded data are intensities (squares of the structure factors).<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976)]<br>
<br><huge>
    FOM<sub>R2</sub> =
    &#8721;<sub><var>i</var></sub> [
    (<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>)<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [ <var>Y<sub>i</sub><sup>2</sup></var> ]
</huge><br>


<h4>logR2</h4>
The logarithmic R2 factor is a modification of the crystallographic R2 factor, calculated using the logarithm (base 10) of the structure factor and simulation. This scaling results in a more similar weighting of high-intensity and low-intensity data points which can be very helpful when fitting data which is spanning several orders of magnitude on the y-axis. Essentially it gives all data points equal weight when displayed in a log-plot.<br>
<br><huge>
    FOM<sub>logR2</sub> =
    &#8721;<sub><var>i</var></sub> [
    (log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)
    )<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [
    log<sub>10</sub>(<var>Y<sub>i</sub>)<sup>2</sup></var> ]
</huge><br>


<h4>sintth4</h4>
Gives the average of the absolute differences scaled with a sin(2&theta;)<sup>4</sup> term (2&theta; = tth). For reflectivity data, this will divide away the Fresnel reflectivity. <br>
<br><huge>
    FOM<sub>sintth4</sub> = 1/(N-p) &times;
    &#8721;<sub><var>i</var></sub>
    &#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124; &times;
    sin(<var>tth</var>)<sup>4</sup>
</huge><br>


<h3>Weighted FOM functions</h3>

<h4>chi2bars</h4>
Chi squared (&chi;<sup>2</sup>) FOM including error bars<br>
<br><huge>
    FOM<sub>chi2bars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    ((<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>) /
    <var>E<sub>i</sub></var>)<sup>2</sup>
</huge><br>


<h4>chibars</h4>
Chi squared but without the squaring! Includes error bars:<br>
<br><huge>
    FOM<sub>chibars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;(<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>) /
    <var>E<sub>i</sub></var>&#124;
</huge><br>


<h4>logbars</h4>
Absolute logarithmic (base 10) difference, taking errors into account:<br>
<br><huge>
    FOM<sub>logbars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)&#124; /
    <var>E<sub>i</sub></var>*ln(10)*<var>Y<sub>i</sub></var>
</huge><br>


<h4>R1bars</h4>
Similar to the crystallographic R-factor R1, but with weighting of the data points by the experimental error values. The error values in E are assumed to be proportional to the standard deviation of the measured intensities.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976), W.C. Hamilton, Acta Crystallogr. 18(3), 502 (1965)]<br>
<br><huge>
    FOM<sub>R1bars</sub> =
    &#8721;<sub><var>i</var></sub><var> [ sqrt(1/E<sub>i</sub></var>) &times;
    &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
    &#124; ] /
    &#8721;<sub><var>i</var></sub> [ sqrt(1/E<sub>i</sub></var>) &times;
    sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>R2bars</h4>
Weighted R2 factor. The error values in E are assumed to be proportional to the standard deviation of the measured intensities.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976), W.C. Hamilton, Acta Crystallogr. 18(3), 502 (1965)]<br>
<br><huge>
    FOM<sub>R2bars</sub> =
    &#8721;<sub><var>i</var></sub> [ (1/E<sub>i</sub></var>) &times;
    (<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>)<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [ (1/E<sub>i</sub></var>) &times;
    <var>Y<sub>i</sub><sup>2</sup></var> ]
</huge><br>


<h2>Customization</h2>
Users can add their own cumstom-built FOM functions to be used in GenX. For detailed instructions on how to write the code for a custom FOM function and how to include it in the list of FOM functions available to GenX, see the manual at
<a href = "http://apps.sourceforge.net/trac/genx/wiki/DocPages/WriteFom">
http://apps.sourceforge.net/trac/genx/wiki/DocPages/WriteFom </a>
'''
#==============================================================================

import numpy as np
import functools

# import also the custom FOM functions defined in fom_funcs_custom.py
# (do nothing if file does not exist)
try:
    from fom_funcs_custom import *
    #print "Imported custom-defined FOM functions from fom_funcs_custom.py"
except:
    pass
    #print "Could not find additional custom-defined FOM functions."
    #print "Nothing imported. All standard FOM functions are available."

bg_peaks={'00':[0,2,4,6],'02':[-8.2782,-6.2782,-4.2782,-2.2782,-0.2782,1.7218,3.7218,5.7218,7.7218],\
          '10':[-7,-5.0,-3.0,3.0,5.0,7],'11':[-6.1391,-4.1391,-2.1391,-0.1391,1.8609,3.8609,5.8609],\
          '20':[-8,-6,-4,-2,0,2,4,6,8],'22':[-8.2782,-6.2782,-4.2782,-2.2782,-0.2782,1.7218,3.7218,5.7218,7.7218],\
          '30':[-9,-7,-5,-1,1,5,7,9],'2-1':[-8.8609,-6.8609,-4.8609,-0.8609,3.1391,5.1391,7.1391],\
          '21':[-7.1391,-5.1391,-3.1391,0.8609,4.8609,6.8609]}
#==============================================================================
# BEGIN FOM function defintions

#decorator to weight FOM (will be imported in the model, and the decoration will be done at site)
def weight_fom_based_on_HKL(weight_factor,weight_map):
    #eg weight_factor = 10
    #eg weight_map = {(1,1):[[1.6,2.4],[3.6,4.4]]}, which define the l segments to be weighted
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            # print(kwargs.keys(),len(args))
            if len(kwargs) == 0:
                datasets = args[1]
            else:
                datasets = kwargs['data']
            weight_array_list = []
            for dataset in datasets:
                hk_tag = (int(round(dataset.extra_data['h'][0],0)), int(round(dataset.extra_data['k'][0],0)))
                # if hk_tag in weight_pars.weight_map:
                if hk_tag in weight_map:
                    # L_segments = weight_pars.weight_map[hk_tag]
                    L_segments = weight_map[hk_tag]
                    l = dataset.x
                    conditions_str_list = []
                    for each in L_segments:
                        conditions_str_list.append('((l>={}) & (l<={}))'.format(*each))
                    condition = eval('|'.join(conditions_str_list))
                    coniditon_false = (condition==False)*1
                    # condition_true = condition*1*weight_pars.weight_factor
                    condition_true = condition*1*weight_factor
                    weight_array_list.append(condition_true + coniditon_false)
                else:
                    weight_array_list.append(np.ones(len(dataset.x)))
            results = func(*args,**kwargs)
            for i, weight_array in enumerate(weight_array_list):
                results[i] = results[i]*weight_array
            return results
        return wrapper
    return decorator

#=========================
# unweighted FOM functions
#@weight_fom_based_on_HKL
def diff(simulations, data):
    ''' Average absolute difference
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    #return 1.0/(N-1)*np.sum([np.sum(np.abs(dataset.y - sim))\
    #    for (dataset, sim) in zip(data,simulations) if dataset.use])
    return [(dataset.y - sim)
        for (dataset, sim) in zip(data,simulations)]
diff.__div_dof__ = True

#decorator func for Fom weighting purpose
#@weight_fom_based_on_HKL
def log(simulations, data):
    ''' Average absolute logartihmic difference
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(np.log10(dataset.y)-np.log10(sim))
        for (dataset, sim) in zip(data,simulations)]
log.__div_dof__ = True

#@weight_fom_based_on_HKL
def log_debug(simulations, data):
    ''' Average absolute logartihmic difference
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    print('start')
    for dataset, sim in zip(data,simulations):
        print(np.log10(dataset.y[0:4]),np.log10(sim[0:4]))
    print('end')
    return [(np.log10(dataset.y)-np.log10(sim))
        for (dataset, sim) in zip(data,simulations)]
log_debug.__div_dof__ = True

#@weight_fom_based_on_HKL
def sqrt(simulations, data):
    ''' Average absolute difference of the square root
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(np.sqrt(dataset.y) - np.sqrt(sim))
        for (dataset, sim) in zip(data,simulations)]
sqrt.__div_dof__ = True

#@weight_fom_based_on_HKL
def R1(simulations, data):
    ''' Crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data\
        if dataset.use])
    return [1.0/denom*(np.sqrt(np.abs(dataset.y)) - np.sqrt(np.abs(sim)))\
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def R1_weighted(simulations, data):
    ''' Crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data\
        if dataset.use])
    return [1.0/denom*abs(np.sqrt(np.abs(dataset.y)) - np.sqrt(np.abs(sim)))/np.sqrt(np.abs(dataset.y))\
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def R1_weighted_2(simulations, data):
    ''' Crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data\
        if dataset.use])
    #denom=1
    return_list=[]
    for (dataset, sim) in zip(data,simulations):
        if dataset.x[0]>100:
            scaler=np.average(dataset.y[[6,19,-6]]/sim[[6,19,-6]])
            return_list.append(1.0/denom*abs(np.sqrt(np.abs(dataset.y)) - np.sqrt(np.abs(sim*scaler)))/np.sqrt(np.abs(dataset.y)))
        else:
            return_list.append(1.0/denom*abs(np.sqrt(np.abs(dataset.y)) - np.sqrt(np.abs(sim)))/np.sqrt(np.abs(dataset.y)))
    return return_list


#@weight_fom_based_on_HKL
def chi2bars_2(simulations, data):
    ''' Weighted chi squared
    '''
    return_list=[]
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    for (dataset,sim) in zip(data,simulations):
        if dataset.x[0]>100:
            scaler=np.average(dataset.y[6:-6]/sim[6:-6])
            return_list.append((dataset.y - sim*scaler)**2/dataset.error**2)
        else:
            return_list.append((dataset.y - sim)**2/dataset.error**2)
    return return_list
chi2bars_2.__div_dof__ = True

#@weight_fom_based_on_HKL
def R1_weighted_2b(simulations, data):
        ''' Crystallographic R-factor (R1)
        '''
        denom = np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data\
            if dataset.use])
        #denom=1
        return_list=[]
        for (dataset, sim) in zip(data,simulations):
            if dataset.x[0]>100:
                scaler=np.average(dataset.y[[6,19,32]]/sim[[6,19,32]])
                return_list.append(1.0/denom*abs(np.abs(dataset.y[6:-6]) - np.abs(sim[6:-6]*scaler))/np.abs(dataset.y[6:-6]))
            else:
                return_list.append(1.0/denom*abs(np.sqrt(np.abs(dataset.y)) - np.sqrt(np.abs(sim)))/np.sqrt(np.abs(dataset.y)))
        return return_list

#@weight_fom_based_on_HKL
def R1_weighted_3(simulations, data):
    ''' Crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data\
        if dataset.use])
    #denom=1
    return_list=[]
    for (dataset, sim) in zip(data,simulations):
        if dataset.x[0]>100:
            scaler=np.average(dataset.y[[6,19,32]]/sim[[6,19,32]])
            return_list.append(1.0/denom*abs(np.log10(np.sqrt(np.abs(dataset.y[6:-6]))) - np.log10(np.sqrt(np.abs(sim[6:-6]*scaler)))))
        else:
            return_list.append(1.0/denom*abs(np.log10(np.sqrt(np.abs(dataset.y))) - np.log10(np.sqrt(np.abs(sim)))))
    return return_list

#@weight_fom_based_on_HKL
def logR1(simulations, data):
    ''' logarithmic crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.log10(np.sqrt(dataset.y))) for dataset in data\
        if dataset.use])
    return [1.0/denom*(np.log10(np.sqrt(dataset.y)) - \
                                        np.log10(np.sqrt(sim)))\
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def R2(simulations, data):
    ''' Crystallographic R2 factor
    '''
    denom = np.sum([np.sum(dataset.y**2) for dataset in data\
        if dataset.use])
    return [1.0/denom*np.sign(dataset.y - sim)*(dataset.y - sim)**2\
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def R2_weighted(simulations, data):
    ''' Crystallographic R2 factor
    '''
    denom = np.sum([np.sum(dataset.y**2) for dataset in data\
        if dataset.use])
    return [1.0/denom*np.sign(dataset.y - sim)*(dataset.y - sim)**2/dataset.error**2\
        for (dataset, sim) in zip(data,simulations)]


#@weight_fom_based_on_HKL
def logR2(simulations, data):
    ''' logarithmic crystallographic R2 factor
    '''
    denom = np.sum([np.sum(np.log10(dataset.y)**2) for dataset in data\
        if dataset.use])
    return [1.0/denom*np.sign(np.log10(dataset.y) - np.log10(sim))*(np.log10(dataset.y) - np.log10(sim))**2\
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def sintth4(simulations, data):
    ''' Sin(tth)^4 scaling of the average absolute difference for reflectivity.
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [np.sin(dataset.x*np.pi/360.0)**4*
        (dataset.y - sim)
        for (dataset, sim) in zip(data,simulations)]
sintth4.__div_dof__ = True

#@weight_fom_based_on_HKL
def Norm(simulations, data):
    '''  dataset normalized 1/3 scaling of the error
    '''
    return [1.0/np.sum(np.abs(dataset.y))*(np.sign(dataset.y)*np.abs(dataset.y) - np.sign(sim)*np.abs(sim))\
        for (dataset, sim) in zip(data,simulations)]
Norm.__div_dof__ = True

#=======================
# weighted FOM functions

#@weight_fom_based_on_HKL
def chi2bars(simulations, data):
    ''' Weighted chi squared
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(dataset.y - sim)**2/dataset.error**2 for (dataset, sim) in zip(data,simulations)]
chi2bars.__div_dof__ = True

#@weight_fom_based_on_HKL
def chi2bars_one_time(simulations, data):
    ''' Weighted chi squared
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(dataset.y - sim)**2/(dataset.error+4.55+dataset.y*0.03)**2 for (dataset, sim) in zip(data,simulations)]
chi2bars_one_time.__div_dof__ = True

#@weight_fom_based_on_HKL
def chi2bars_w_trainor(simulations, data):
    ''' Weighted chi squared
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(dataset.y - sim)**2/(dataset.y*0.2)**2 for (dataset, sim) in zip(data,simulations)]
chi2bars_w_trainor.__div_dof__ = True

#fom's are weighted with dip zones having higher wt number and bragg peak zone having lower wt number
#@weight_fom_based_on_HKL
def chi2bars_weighted(simulations, data):
    ''' Weighted chi squared
    '''
    def _weight_fom(h,k,l_list=[]):
        wt_array=[]
        hk=str(int(h))+str(int(k))
        for l in l_list:
            temp_sign=np.array(bg_peaks[hk])-l
            left,right=0,0
            for sign in temp_sign:
                if sign>=0:
                    right=list(temp_sign).index(sign)
                    left=right-1
                    break
            l_mid=(bg_peaks[hk][left]+bg_peaks[hk][right])/2
            l_half_span=(bg_peaks[hk][right]-bg_peaks[hk][left])/2
            l_span=abs(l-l_mid)
            wt_array.append(50/(1+l_span/l_half_span*50))
        #print wt_array
        return np.array(wt_array)
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [np.sign(dataset.y - sim)*(dataset.y - sim)**2/dataset.error**2*_weight_fom(dataset.extra_data['h'][0],dataset.extra_data['k'][0],dataset.x)
        for (dataset, sim) in zip(data,simulations)]
chi2bars_weighted.__div_dof__ = True


#@weight_fom_based_on_HKL
def chibars(simulations, data):
    ''' Weighted chi squared but without the squaring
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [((dataset.y - sim)/dataset.error)
        for (dataset, sim) in zip(data,simulations)]
chibars.__div_dof__ = True

#@weight_fom_based_on_HKL
def logbars(simulations, data):
    ''' Weighted average absolute difference of the logarithm of the data
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [((np.log10(dataset.y) - np.log10(sim))
        /dataset.error*np.log(10)*dataset.y)
        for (dataset, sim) in zip(data,simulations)]
logbars.__div_dof__ = True

#@weight_fom_based_on_HKL
def R1bars(simulations, data):
    ''' Weighted crystallographic R-factor (R1)
    '''
    denom = np.sum([np.sum(np.sqrt(1/dataset.error)*np.sqrt(dataset.y))
                    for dataset in data if dataset.use])
    return [1.0/denom*np.sqrt(1/dataset.error)*
           (np.sqrt(dataset.y) - np.sqrt(sim))
        for (dataset, sim) in zip(data,simulations)]

#@weight_fom_based_on_HKL
def R2bars(simulations, data):
    ''' Weighted crystallographic R2 factor
    '''
    denom = np.sum([(1/dataset.error)*np.sum(dataset.y**2)
                    for dataset in data if dataset.use])
    return [1.0/denom*(1/dataset.error) * np.sign(dataset.y - sim)*(dataset.y - sim)**2
        for (dataset, sim) in zip(data,simulations)]


# END FOM function definition
#==============================================================================


# create introspection variables so that everything updates automatically
# Find all objects in this namespace
# (this includes the custom-defined FOM functions from fom_funcs_custom.py)
obj_list = dir()[:]

# find all functions
all_func_names = [s for s in obj_list if type(eval(s)).__name__ == 'function']
func_names = [s for s in all_func_names if all_func_names[0] != '_']

# End of file
#==============================================================================
