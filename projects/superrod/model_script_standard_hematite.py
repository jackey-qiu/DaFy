import models.sxrd_new1_hematite as model
from models.utils import UserVars
from datetime import datetime
import numpy as np
import sys,pickle,os
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
import accessory_functions.make_par_table.make_parameter_table_GenX_hematite_rcut as make_grid
import accessory_functions.data_formating.formate_xyz_to_vtk as xyz
from copy import deepcopy
import models.setup_domain_hematite_rcut as setup_domain_hematite_rcut
from accessory_functions.data_formating.data_formating import format_hkl
from UtilityFunctions import config_file_parser_bv, update_O_NUMBER

##matching index##
"""
*********************************************************************************************************************************
* Note the pickup index supposed to be used in a ternary complex structure have no concept of domain type, so you can           *
* freely mix using both. For example, the pickup_index item of [15,26,26] has no difference from [15,11,11] and also [15,11,26].*
*********************************************************************************************************************************

HL-->0          1           2           3             4              5(Face-sharing)     6           7
CS(O1O2)        CS(O2O3)    ES(O1O3)    ES(O1O4)      TD(O1O2O3)     TD(O1O3O4)          OS          Clean

TERNARY HL-->     8         9           10            11             12                  13          14
                  M(HO1)    M(HO2)      M(HO3)        B(HO1_HO2)     B(HO1_HO3)          B(HO2_HO3)  T(HO1_HO2_HO3)

FL-->15         16          17          18            19             20(Face-sharing)    21          22
CS(O5O6)        ES(O5O7)    ES(O5O8)    CS(O6O7)      TD(O5O6O7)     TD(O5O7O8)          OS          Clean

TERNARY FL-->   23        24          25            26             27                  28          29
                M(HO1)    M(HO2)      M(HO3)        B(HO1_HO2)     B(HO1_HO3)          B(HO2_HO3)  T(HO1_HO2_HO3)
"""
##############################################main setup zone###############################################
#depository path for output files(structure model files(.xyz,.cif), optimized values (CTR,RAXR,E_Density) for plotting
batch_path_head=batch_path.module_path_locator()
output_file_path=output_path.module_path_locator()

#/pick_index/begin#
pickup_index=[[7], [19], [19]]#using matching index table above
half_layer=[3]#2 for short slab and 3 for long slab
full_layer=[0, 1]#0 for short slab and 1 for long slab
N_FL=len(full_layer)#number of full layer domains
N_HL=len(half_layer)#number of half layer domains
half_layer_pick=half_layer+[None]*N_FL
full_layer_pick=[None]*N_HL+full_layer
COHERENCE=[{True:range(len(pickup_index))}] #want to add up in coherence? items inside list corresponding to each domain
FULL_LAYER_PICK_INDEX=domain_creator.make_pick_index(full_layer_pick=full_layer_pick,
                                                     pick=pickup_index,
                                                     half_layer_cases=15,
                                                     full_layer_cases=15)
HALF_LAYER_PICK_INDEX=domain_creator.make_pick_index_half_layer(half_layer_pick=half_layer_pick,
                                                     pick=pickup_index,
                                                     half_layer_cases=15)
#/pick_index/end#

#domain_type
DOMAIN=domain_creator.pick([1]*15+[2]*15,pickup_index)
DOMAIN_NUMBER=len(DOMAIN)

#metal_id/begin#
SORBATE=[['Pb'], ['Pb'], ['Pb']]#same shape as pickup_index
SORBATE_EL_LIST = list(set(sum(SORBATE,[])))
#metal_id/end#

#setup_sorbate/begin#
#-------------------------------------------------------#
#update the sorbate in sim function based on the frame of geometry?
UPDATE_SORBATE_IN_SIM=True
sym_site_index=[[[0,1]]* len(each) for each in pickup_index]
#fit top angle if true otherwise fit the Pb-O bond length (used in bidentate case)
USE_TOP_ANGLE=True
MIRROR=[[False for each_item in each] for each in pickup_index]
BASAL_EL=[[None]+each_domain[:-1] for each_domain in SORBATE]
setting_attributs_lib = setup_domain_hematite_rcut.setup_standard(HALF_LAYER_PICK_INDEX,\
                                                                  FULL_LAYER_PICK_INDEX,\
                                                                  N_HL,N_FL,SORBATE_EL_LIST,\
                                                                  sym_site_index,pickup_index)
locals().update(setting_attributs_lib)
#setup bond valence attributes
locals().update(config_file_parser_bv(os.path.join(batch_path_head,'bv_data_base','config_bond_valence_db.ini')))
#if only you want to specify the coords of sorbates
USE_COORS=[[0,0,0,0]*10]*len(pickup_index)
COORS={(0,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]},\
       (2,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]}}

#/setup_distal_oxygen_number/begin#
O_NUMBER_HL=[[4,4],[0,0],[1,1],[0,0],[3,3],[0,0],[3,3],[0,0]]
O_NUMBER_HL_EXTRA=[[0,0],[0,0],[0,0],[1,1],[1,1],[0,0],[0,0]]
O_NUMBER_FL=[[2,2],[0,0],[0,0],[0,0],[0,0],[0,0],[4,4],[0,0]]
O_NUMBER_FL_EXTRA=[[0,0],[0,0],[0,0],[2,2],[0,0],[0,0],[4,4]]
#update distal oxygen number according to CN and binding mode
O_NUMBER_lib = update_O_NUMBER(SORBATE,pickup_index,METAL_VALENCE)
for key in O_NUMBER_lib:
    for i in range(len(O_NUMBER_lib[key])):
        index_, value_ = O_NUMBER_lib[key][i]
        locals()[key][index_] = [value_]*2
#/setup_distal_oxygen_number/end#

#sorbate number(2 by default) within each unitcell#
SORBATE_NUMBER_HL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_HL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]
SORBATE_NUMBER_FL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_FL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]

SORBATE_NUMBER=domain_creator.pick_act(SORBATE_NUMBER_HL+SORBATE_NUMBER_HL_EXTRA+SORBATE_NUMBER_FL+SORBATE_NUMBER_FL_EXTRA,pickup_index)
O_NUMBER=domain_creator.pick_act(O_NUMBER_HL+O_NUMBER_HL_EXTRA+O_NUMBER_FL+O_NUMBER_FL_EXTRA,pickup_index)
SORBATE_LIST=domain_creator.create_sorbate_el_list2(SORBATE,SORBATE_NUMBER)

#Outersphere ref setup#
OS_X_REF, OS_Y_REF, OS_Z_REF = domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])
#setup_sorbate/end#

#/setup_layered_water/sorbate/begin#
WATER_LAYER_NUM=[4, 4, 4]
ref_height_adsorb_water_map={0:[['O1_5_0','O1_6_0']],1:[['O1_11_t','O1_12_t']],2:[['O1_7_0','O1_8_0']],3:[['O1_1_0','O1_2_0']]}
WATER_LAYER_REF=list(map(lambda key,n:ref_height_adsorb_water_map[key]*int(n/2),half_layer+full_layer,WATER_LAYER_NUM))
water_pars={'use_default':False,'number':WATER_LAYER_NUM,'ref_point':WATER_LAYER_REF}
ref_height_water_map={0:'O1_5_0',1:'O1_11_t',2:'O1_7_0',3:'O1_1_0'}
layered_water_pars={'yes_OR_no':[1]*len(pickup_index),\
                    'ref_layer_height':list(map(lambda key:ref_height_water_map[key],half_layer+full_layer))}
WATER_PAIR=True#add water pair each time if True, otherwise only add single water each time (only needed par is V_SHIFT)
WATER_NUMBER,REF_POINTS=setup_domain_hematite_rcut.setup_water_pars(water_pars,N_HL,N_FL,
                                                                    pickup_index,FULL_LAYER_PICK_INDEX,
                                                                    HALF_LAYER_PICK_INDEX)
layered_sorbate_pars={'yes_OR_no':[0]*len(pickup_index),'ref_layer_height':['O1_1_0']*len(pickup_index),'el':'Pb'}
#----------------------------------------------#
#/setup_layered_water/sorbate/end#

#/setup_constants/begin#
wal=0.8625#wavelength of x ray
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = wal, alpha = 2.0)
re = 2.818e-5#electron radius
SURFACE_PARMS={'delta1':0.,'delta2':0.1391}#correction factor in surface unit cell
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)
#/setup_constants/end#

#setting namespace
names,vars_container=setup_domain_hematite_rcut.setup_atm_ids(DOMAIN_NUMBER,\
                                                              DOMAIN,SORBATE,SORBATE_NUMBER,\
                                                              SORBATE_LIST,WATER_NUMBER,\
                                                              O_NUMBER,half_layer_pick,full_layer_pick)
vars().update(dict(zip(names, vars_container)))

#/bond_valence_constraint/begin#
#--------------------------------------------------#
USE_BV=0#do you want to use bv contraints
debug_bv=False
DOMAINS_BV=range(len(pickup_index))
COUNT_DISTAL_OXYGEN=False
ADD_DISTAL_LIGAND_WILD=[[False]*10]*10
BV_OFFSET_SORBATE=[[0.2]*8]*len(pickup_index)
if not CONSIDER_WATER_IN_BV:
    BOND_VALENCE_WAIVER=BOND_VALENCE_WAIVER+['Os'+str(ii+1) for ii in range(10)]

##pars for sorbates##
LOCAL_STRUCTURE=deepcopy(SORBATE)
METAL_BV_EACH=deepcopy(SORBATE)
BOND_LENGTH_EACH=deepcopy(SORBATE)
for i in range(len(LOCAL_STRUCTURE)):
    for j in range(len(LOCAL_STRUCTURE[i])):
        #valence for each bond
        METAL_BV_EACH[i][j]=METAL_VALENCE[SORBATE[i][j]][0]/METAL_VALENCE[SORBATE[i][j]][1]
        #ideal bond length using bond valence equation
        BOND_LENGTH_EACH[i][j]=R0_BV[(SORBATE[i][j],'O')]-np.log(METAL_BV_EACH[i][j])*0.37
        for key in LOCAL_STRUCTURE_MATCH_LIB.keys():
            if LOCAL_STRUCTURE[i][j] in LOCAL_STRUCTURE_MATCH_LIB[key]:
                LOCAL_STRUCTURE[i][j]=key
                break
            else:
                pass

#specify the searching range and penalty factor for surface atoms and sorbates
#The value for each item [searching radius(A),scaling factor]
SEARCHING_PARS={'surface':[2.5,50],'sorbate':[[np.array(each)+SEARCH_RANGE_OFFSET for each in BOND_LENGTH_EACH],50]}
N_BOND,METAL_BV=setup_domain_hematite_rcut.setup_bv_condition(O_NUMBER,SORBATE_ATTACH_ATOM,METAL_BV_EACH,BV_OFFSET_SORBATE)
#Protonation of distal oxygens, any number in [0,1,2], where 1 means singly protonated, two means doubly protonated
PROTONATION_DISTAL_OXYGEN=[[0,0]]*len(pickup_index)
#--------------------------------------------------#
#/bond_valence_constraint/end#

#/setup_raxr_fitting/begin#
RAXR_EL='Pb'
RAXR_FIT_MODE='MI'#model dependent (MD) or Model independent (MI)
NUMBER_SPECTRA=0
RESONANT_EL_LIST=[1]+[0]*(len(pickup_index)-1)#use average A+P for the whole domain
E0=11873
F1F2_FILE='As_K_edge_March28_2018.f1f2'
##set up Fourier pars if there are RAXR datasets
#Fourier component looks like A_Dn0_n1, where n0, n1 are used to specify the index for domain, and spectra, respectively
#Each spectra will have its own set of A and P list, and each domain has its own set of P and A list
rgh_raxr,F1F2=setup_domain_hematite_rcut.setup_raxr_pars(NUMBER_SPECTRA,batch_path_head,F1F2_FILE,RESONANT_EL_LIST)
#/setup_raxr_fitting/end#

#/setup_group_scheme/begin#
GROUPING_SCHEMES=[[2, 1]]#domain tag of first domain is 1 (Domain2=Domain1)
GROUPING_DEPTH=[[0, 10]]#means I will group top 10 (range(0,10)) layers of domain2 to those of domain1
commands_surface=domain_creator.generate_commands_for_surface_atom_grouping_new(np.array(GROUPING_SCHEMES),\
                                                                   domain_creator.translate_domain_type(GROUPING_SCHEMES,half_layer+full_layer),\
                                                                   GROUPING_DEPTH)
#/setup_group_scheme/end#

#/setup_slabs/begin#
bulk = model.Slab(T_factor='B')
domain_creator.add_atom_in_slab(bulk,os.path.join(batch_path_head,'hematite_rcut','bulk.str'))
for i in range(DOMAIN_NUMBER):
    domain_tag_ = int(int(DOMAIN[i])==1)
    file_prefix = ['full_layer','half_layer'][domain_tag_]
    pick_case = [full_layer_pick,half_layer_pick][domain_tag_]
    tag_slab_ = [2,3][int(pick_case[i]==[0,2][domain_tag_])]
    vars()['domain_class_'+str(int(i+1))]=\
        domain_creator.domain_creator.add_atom_in_slab(slab = model.Slab(c = 1.0,T_factor='B'),\
                                                       terminated_layer=0, N_layers = 5, \
                                                       domain_tag='_D'+str(int(i+1)),\
                                                       new_var=vars()['rgh_domain'+str(int(i+1))],\
                                                       filename = os.path.join(batch_path_head,'hematite_rcut','{}{}.str'.format(file_prefix,tag_slab_)))

    #Adding sorbates to domainA and domainB
    sorbate_methods = {0:"setup_sorbate_OS_new", 1:"setup_sorbate_MD_new", 2:"setup_sorbate_BD_new", 3:"setup_sorbate_TD_new"}
    for j in range(sum(SORBATE_NUMBER[i])):
        method_ = getattr(setup_domain_hematite_rcut,sorbate_methods[len(SORBATE_ATTACH_ATOM[i][j])])
        return_list = method_(vars(),i,j)
        vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))],vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=return_list

    #setup water layers
    water_gp_names,water_groups=setup_domain_hematite_rcut.setup_layer_structure(vars(),i)

    for iii in range(len(water_groups)):vars()[water_gp_names[iii]]=water_groups[iii]
#/setup_slabs/end#

#/setup_rgh_global/begin#
rgh=UserVars()
rgh.new_var('beta', 0.0)#roughness factor
rgh.new_var('mu',1)#liquid film thickness
scales=['scale_CTR']
for scale in scales:
    rgh.new_var(scale,1.)
#/setup_rgh_global/end#

#/build_parameter/begin#
O_N,binding_mode=setup_domain_hematite_rcut.setup_fit_table(O_NUMBER,DOMAIN_NUMBER,SORBATE_ATTACH_ATOM)
make_grid.make_structure(list(map(sum,SORBATE_NUMBER)),O_N,WATER_NUMBER,DOMAIN,
                        Metal=SORBATE,binding_mode=binding_mode,\
                        long_slab=full_layer_pick,long_slab_HL=half_layer_pick,\
                        local_structure=LOCAL_STRUCTURE,add_distal_wild=ADD_DISTAL_LIGAND_WILD,\
                        use_domains=TABLE_DOMAINS,N_raxr=NUMBER_SPECTRA,domain_raxr_el=RESONANT_EL_LIST,\
                        layered_water=layered_water_pars['yes_OR_no'],layered_sorbate=layered_sorbate_pars['yes_OR_no'],\
                        tab_path=os.path.join(output_file_path,'table.tab'))
#/build_parameter/end#

######################################do grouping###############################################
#####################surface atom grouping+sorbate grouping#####################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together (4 atms grouped together in sequence grouping)
    #you may group in symmetry, then atoms of same layer are not independent. Know here the symmetry (equal opposite) is impressively defined in the function
    vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new3(layers_N=10)
    vars().update(dict(zip(vars()['sequence_gp_names_domain'+str(int(i+1))],vars()['atm_gp_list_domain'+str(int(i+1))])))

    #you may also only want to group each chemically equivalent atom from two domains (the use_sym is set to true here)
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        ids_ = [vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]]
        sym_ = [[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]]
        gp_ = vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer4(atom_ids= ids_,sym_array=sym_)
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(gp_)
    vars().update(dict(zip(vars()['discrete_gp_names_domain'+str(int(i+1))],vars()['atm_gp_discrete_list_domain'+str(int(i+1))])))

    #group sorbates in pair or not
    for j in range(0,sum(SORBATE_NUMBER[i]),[1,2][int(SORBATE_NUMBER[i]==2)]):
        vars()['gp_'+SORBATE_LIST[i][j]+'_set'+str(j+1)+'_D'+str(i+1)]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_metal_layer()
        locals().update(vars()['domain_class_'+str(int(i+1))].grouping_discrete_HO_layer(which_set = j, which_domain = i))

#####################################do bond valence matching###################################
######################find the coordinating atoms for each constituent atom#####################
if USE_BV:
    for i in range(DOMAIN_NUMBER):
        lib_sorbate={}
        if SORBATE_NUMBER[i]!=0:
            lib_sorbate=domain_creator.create_sorbate_match_lib4_test(metal=SORBATE_LIST[i],\
                                                                      HO_list=vars()['HO_list_domain'+str(int(i+1))+'a'],\
                                                                      anchors=SORBATE_ATTACH_ATOM[i],\
                                                                      anchor_offsets=SORBATE_ATTACH_ATOM_OFFSET[i],\
                                                                      domain_tag=i+1)
        if DOMAIN[i]==1:
            rem_atom_ids = {3:['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A'],
                            2:['Fe1_8_0_D'+str(int(i+1))+'A','Fe1_9_0_D'+str(int(i+1))+'A']}[half_layer_pick[i]]

            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting\
                           (\
                             search_range=2.3,\
                             atm_list=vars()['atm_list_'+str(int(i+1))+'A'],\
                             domain_class=vars()['domain_class_'+str(int(i+1))],\
                             domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],\
                                                                                           rem_atom_ids=rem_atom_ids)\
                           )
        elif DOMAIN[i]==2:
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting\
                            (\
                             search_range = 2.3,\
                             atm_list=vars()['atm_list_'+str(int(i+1))+'A'],\
                             domain_class=vars()['domain_class_'+str(int(i+1))],\
                             domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],\
                                                                                           rem_atom_ids=None)\
                            )
        vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)

#set up multiple domains
#note for each domain there are two sub domains which symmetrically related to each other, so have equivalent wt
domain = {}
for i in range(DOMAIN_NUMBER):
    domain['domain{}'.format(i+1)] = vars()['domain_class_'+str(int(i+1))]
sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE, surface_parms=SURFACE_PARMS)

#commands to be executed in SIM function#
#you can add more commads in this list here(like 'command1','command2')#
commands_other= []
commands=commands_other+commands_surface

VARS_out=vars()#pass local variables to sim function

def Sim(data):
    VARS = VARS_out
    for command in commands:eval(command)
    F =[]
    bv=0
    bv_container={}
    fom_scaler=[]
    beta=rgh.beta
    SCALES=[getattr(rgh,scale) for scale in scales]
    total_wt= sum([VARS['domain_class_'+str(int(i+1))].new_var_module.wt for i in range(DOMAIN_NUMBER)])

    for i in range(DOMAIN_NUMBER):
        #grap wt for each domain and cal the total wt
        VARS['domain_class_'+str(int(i+1))].wt = VARS['domain_class_'+str(int(i+1))].new_var_module.wt/total_wt/2

        #update sorbates
        if UPDATE_SORBATE_IN_SIM:
            binding_mode_ = VARS['domain_class_'+str(int(i+1))].binding_type
            update_sorbate_method = getattr(setup_domain_hematite_rcut, "update_sorbate_in_SIM_{}_new".format(binding_mode_))
            for j in range(sum(VARS['SORBATE_NUMBER'][i])):
                if not USE_COORS[i][j]:
                    update_sorbate_method(VARS['domain_class_'+str(int(i+1))], i, j)

        #updata adsorbed water structure
        setup_domain_hematite_rcut.update_water_structure(VARS['domain_class_'+str(int(i+1))],i)
        
        #calculate bv panelty factor
        if USE_BV and i in DOMAINS_BV:
            bv_temp,bv_container_temp=setup_domain_hematite_rcut.calculate_BV_sum_in_SIM(i,VARS)
            bv=bv+bv_temp
            if debug_bv:
                for key in bv_container_temp.keys():bv_container[key]=bv_container_temp[key]
    if debug_bv:
        print("Print out the species, which are not under bond valence saturation")
        for i in bv_container.keys():
            if bv_container[i]!=0:
                print(i,"BV after considering penalty",bv_container[i])

    i=0
    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']

        if data_set.use:
            if data_set.x[0]>100:#doing RAXR calculation(x is energy column typically in magnitude of 10000 ev)
                h_,k_,y_=format_hkl(h,k,y,h_list = [1,2,3],k_list=[0])
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                f=sample.cal_structure_factor_hematite_RAXR(i,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h_, k_, y_, x, E0, F1F2,SCALES,rough)
                F.append(abs(f))
                fom_scaler.append(1)
                i+=1
            else:#doing CTR calculation (x is perpendicular momentum transfer L typically smaller than 15)
                h_,k_,x_=format_hkl(h,k,x, h_list= [1,2,3], k_list = [0])
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(x-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                    q=np.pi*2*unitcell.abs_hkl(h_,k_,x_)
                    pre_factor=(np.exp(-exp_const*rgh.mu/q))*(4*np.pi*re/auc)*3e6
                    pre_factor = 1
                    f = pre_factor*SCALES[0]*rough*sample.calc_f4_specular(h_, k_, x_, RAXR_EL)
                else:
                    f = rough*sample.calc_f4(h_, k_, x_)
                F.append(abs(f))
                fom_scaler.append(1)
        else:
            if x[0]>100:
                i+=1
            f=np.zeros(len(y))
            F.append(f)
            fom_scaler.append(1)

    #model files output (structure and bestfit results)
    if PRINT_MODEL_FILES:
        for i in range(DOMAIN_NUMBER):
            #make sure you have the test.tab file in the specified folder to output file for publication
            setup_domain_hematite_rcut.output_model_files(i,COVALENT_HYDROGEN_NUMBER,PROTONATION_DISTAL_OXYGEN,SORBATE_NUMBER,O_NUMBER,WATER_NUMBER,VARS,output_file_path,xyz,water_pars,half_layer,full_layer,DOMAIN,half_layer_pick,full_layer_pick)

    #make dummy raxr dataset you will need to double check the LB,dL and the hkl#
    DUMMY_RAXR_BUILT,COMBINE_DATA_SETS=False,False
    if DUMMY_RAXR_BUILT:
        setup_domain_hematite_rcut.create_dummy_raxr_data_hematite(sample,data,VARS,RESONANT_EL_LIST,RAXR_EL,beta,E0,F1F2,SCALES,output_file_path)

    #The A and P list returned is calculated based on the model dependent structure#
    Print_AP=False
    if Print_AP:
        AP=sample.find_A_P(np.arange(0,10.38,0.35),'Pb',True)

    #export the model results for plotting if PLOT set to true#
    if PLOT:
        z_min,z_max=0,40
        setup_domain_hematite_rcut.plot_ctr_raxr_e_profiles(model,inst, bulk, domain, unitcell,COHERENCE,data,beta,VARS,E0,F1F2,RAXR_FIT_MODE,RESONANT_EL_LIST,SCALES,output_file_path,RAXR_EL,exp_const,rgh,re,auc,z_min,z_max,SURFACE_PARMS)

    #you may play with the weighting rule by setting eg 2**bv, 5**bv for the wt factor, that way you are pushing the GenX to find a fit btween a good fit (low wt factor) and a reasonable fit (high wt factor)
    return F,1+WT_BV*bv,fom_scaler
