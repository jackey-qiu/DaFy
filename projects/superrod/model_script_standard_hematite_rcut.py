import models.sxrd_new1 as model
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
#************************************program begins from here **********************************************
###############################################global vars##################################################
COUNT_TIME=False
if COUNT_TIME:t_0=datetime.now()

running_mode=1#if 0 will print model plotting files
USE_BV=0#do you want to use bv contraints
debug_bv=False
pickup_index=[[7], [19], [19]]#using matching index table above
SORBATE=[['Pb'], ['Pb'], ['Pb']]#same shape as pickup_index
half_layer=[3]#2 for short slab and 3 for long slab
full_layer=[0, 1]#0 for short slab and 1 for long slab
MIRROR=[[False for each_item in each] for each in pickup_index]
#change the O number according to coordination structure and binding mode
O_NUMBER_HL=[[4,4],[0,0],[1,1],[0,0],[3,3],[0,0],[3,3],[0,0]]
O_NUMBER_HL_EXTRA=[[0,0],[0,0],[0,0],[1,1],[1,1],[0,0],[0,0]]
O_NUMBER_FL=[[2,2],[0,0],[0,0],[0,0],[0,0],[0,0],[4,4],[0,0]]
O_NUMBER_FL_EXTRA=[[0,0],[0,0],[0,0],[2,2],[0,0],[0,0],[4,4]]

#water layer and layer sorbate setup here#
WATER_LAYER_NUM=[4, 4, 4]
ref_height_adsorb_water_map={0:[['O1_5_0','O1_6_0']],1:[['O1_11_t','O1_12_t']],2:[['O1_7_0','O1_8_0']],3:[['O1_1_0','O1_2_0']]}
WATER_LAYER_REF=list(map(lambda key,n:ref_height_adsorb_water_map[key]*int(n/2),half_layer+full_layer,WATER_LAYER_NUM))
water_pars={'use_default':False,'number':WATER_LAYER_NUM,'ref_point':WATER_LAYER_REF}
ref_height_water_map={0:'O1_5_0',1:'O1_11_t',2:'O1_7_0',3:'O1_1_0'}
layered_water_pars={'yes_OR_no':[1]*len(pickup_index),'ref_layer_height':list(map(lambda key:ref_height_water_map[key],half_layer+full_layer))}#For physical reason, there should be layer water for each domain
WATER_PAIR=True#add water pair each time if True, otherwise only add single water each time (only needed par is V_SHIFT)
layered_sorbate_pars={'yes_OR_no':[0]*len(pickup_index),'ref_layer_height':['O1_1_0']*len(pickup_index),'el':'Pb'}

#raxr setup here#
RAXR_EL='Pb'
RAXR_FIT_MODE='MI'#model dependent (MD) or Model independent (MI)
NUMBER_SPECTRA=0
RESONANT_EL_LIST=[1]+[0]*(len(pickup_index)-1)#use average A+P for the whole domain
E0=11873
F1F2_FILE='As_K_edge_March28_2018.f1f2'
F1F2=None

#setup grouping scheme
GROUPING_SCHEMES=[[2, 1]]#domain tag of first domain is 1 (Domain2=Domain1)
GROUPING_DEPTH=[[0, 10]]#means I will group top 10 (range(0,10)) layers of domain2 to those of domain1

#setting slabs##
wal=0.8625#wavelength of x ray
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
SURFACE_PARMS={'delta1':0.,'delta2':0.1391}#correction factor in surface unit cell
inst = model.Instrument(wavel = wal, alpha = 2.0)
bulk = model.Slab(T_factor='B')
ref_S_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_L_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_S_domain2 =  model.Slab(c = 1.0,T_factor='B')
ref_L_domain2 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)#roughness factor
rgh.new_var('mu',1)#liquid film thickness
scales=['scale_CTR']
for scale in scales:
    rgh.new_var(scale,1.)

#set up experimental constant#
re = 2.818e-5#electron radius
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)

#bond valence setup#
COVALENT_HYDROGEN_RANDOM=False
COUNT_DISTAL_OXYGEN=False
ADD_DISTAL_LIGAND_WILD=[[False]*10]*10
BOND_VALENCE_WAIVER=[]
CONSIDER_WATER_IN_BV=False
if not CONSIDER_WATER_IN_BV:BOND_VALENCE_WAIVER=BOND_VALENCE_WAIVER+['Os'+str(ii+1) for ii in range(10)]

BASAL_EL=[[None]+each_domain[:-1] for each_domain in SORBATE]
sym_site_index=[[[0,1]]* len(each) for each in pickup_index]
half_layer_pick=half_layer+[None]*len(full_layer)
full_layer_pick=[None]*len(half_layer)+full_layer

#Outersphere ref setup#
OS_X_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[0]
OS_Y_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[1]
OS_Z_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[2]
DOMAINS_BV=range(len(pickup_index))
TABLE_DOMAINS=[1]*len(pickup_index)

#if only you want to specify the coords of sorbates
USE_COORS=[[0,0,0,0]*10]*len(pickup_index)
COORS={(0,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]},\
       (2,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]}}

#sorbate number(2 by default) within each unitcell#
SORBATE_NUMBER_HL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_HL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]
SORBATE_NUMBER_FL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_FL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]

#grouping commands to be executed in SIM function#
commands_surface=domain_creator.generate_commands_for_surface_atom_grouping_new(np.array(GROUPING_SCHEMES),domain_creator.translate_domain_type(GROUPING_SCHEMES,half_layer+full_layer),GROUPING_DEPTH)
#you can add more commads in this list here(like 'command1','command2')#
commands_other=['gp_HO_set2_D2.setu(gp_HO_set1_D2.getu())', 'gp_sorbates_set2_D2.setoc(gp_sorbates_set1_D2.getoc())', 'gp_HO_set2_D3.setu(gp_HO_set1_D3.getu())', 'gp_sorbates_set2_D3.setoc(gp_sorbates_set1_D3.getoc())']
commands=commands_other+commands_surface
commands=commands_surface
##############################################end of main setup zone############################################
#                                                                                                              #
#                                                                                                              #
#                           You seldomly need to touch script lines hereafter!!!                               #
#                                                                                                              #
#                                                                                                              #
#                                                                                                              #
################################################################################################################
#depository path for output files(structure model files(.xyz,.cif), optimized values (CTR,RAXR,E_Density) for plotting
output_file_path=output_path.module_path_locator()

WT_BV=1#weighting for bond valence constrain (1 recommended)
BV_TOLERANCE=[-0.2,0.2]#ideal bv value + or - this value is acceptable, negative side is over-saturation and positive site is under-saturated
USE_TOP_ANGLE=True#fit top angle if true otherwise fit the Pb-O bond length (used in bidentate case)

#sorbate_el_list is a unique list of sorbate elements being considered in the model system
SORBATE_EL_LIST=[]
[SORBATE_EL_LIST.append(each_el) for each_el in sum(SORBATE,[]) if each_el not in SORBATE_EL_LIST]

FULL_LAYER_PICK_INDEX=domain_creator.make_pick_index(full_layer_pick=full_layer_pick,pick=pickup_index,half_layer_cases=15,full_layer_cases=15)
HALF_LAYER_PICK_INDEX=domain_creator.make_pick_index_half_layer(half_layer_pick=half_layer_pick,pick=pickup_index,half_layer_cases=15)
N_FL=len([i for i in full_layer_pick if i!=None])
N_HL=len(pickup_index)-N_FL
COHERENCE=[{True:range(len(pickup_index))}] #want to add up in coherence? items inside list corresponding to each domain

#setup interfacial water#
WATER_NUMBER,REF_POINTS=setup_domain_hematite_rcut.setup_water_pars(water_pars,N_HL,N_FL,pickup_index,FULL_LAYER_PICK_INDEX,HALF_LAYER_PICK_INDEX)

UPDATE_SORBATE_IN_SIM=True#you may not want to update the sorbate in sim function based on the frame of geometry, then turn this off
SORBATE_ATTACH_ATOM,SORBATE_ATTACH_ATOM_OFFSET,ANCHOR_REFERENCE,ANCHOR_REFERENCE_OFFSET,POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_NUMBER,POTENTIAL_HYDROGEN_ACCEPTOR=setup_domain_hematite_rcut.setup_standard(HALF_LAYER_PICK_INDEX,FULL_LAYER_PICK_INDEX,N_HL,N_FL,SORBATE_EL_LIST,sym_site_index,pickup_index)

##chemically different domain type##
DOMAIN=domain_creator.pick([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],pickup_index)
DOMAIN_NUMBER=len(DOMAIN)

SORBATE_NUMBER=domain_creator.pick_act(SORBATE_NUMBER_HL+SORBATE_NUMBER_HL_EXTRA+SORBATE_NUMBER_FL+SORBATE_NUMBER_FL_EXTRA,pickup_index)
O_NUMBER=[[0, 0], [0, 0], [0, 0]]
SORBATE_LIST=domain_creator.create_sorbate_el_list2(SORBATE,SORBATE_NUMBER)
#give a unique id to each domain atom#
names,vars_container=setup_domain_hematite_rcut.setup_atm_ids(DOMAIN_NUMBER,DOMAIN,SORBATE,SORBATE_NUMBER,SORBATE_LIST,WATER_NUMBER,O_NUMBER,half_layer_pick,full_layer_pick)
for i in range(len(names)):vars()[names[i]]=vars_container[i]

##bond valence calculation handles##
BV_OFFSET_SORBATE=[[0.2]*8]*len(pickup_index)
SEARCH_RANGE_OFFSET=0.3
SEARCH_MODE_FOR_SURFACE_ATOMS=True#If true then cal bond valence of surface atoms based on searching within a spherical region
METAL_VALENCE={'Pb':(2.,3.),'Sb':(5.,6.),'As':(5.,4.),'P':(5.,4.),'Cr':(6.,4.),'Cd':(2.,6.),'Cu':(2.,6.),'Zn':(2.,6.)}#for each value (valence charge,coordination number)
R0_BV={('As','O'):1.767,('Cr','O'):1.794,('Cd','O'):1.904,('Cu','O'):1.679,('Zn','O'):1.704,('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973,('P','O'):1.617}#r0 for different couples
IDEAL_BOND_LENGTH={('As','O'):1.68,('Cr','O'):1.64,('Cd','O'):2.31,('Cu','O'):2.09,('Zn','O'):2.11,('Fe','O'):2.02,('Pb','O'):2.19,('Sb','O'):2.04,('P','O'):1.534}#ideal bond length for each case
LOCAL_STRUCTURE_MATCH_LIB={'trigonal_pyramid':['Pb'],'octahedral':['Sb','Fe','Cd','Cu','Zn'],'tetrahedral':['As','P','Cr']}

##pars for sorbates##
LOCAL_STRUCTURE=deepcopy(SORBATE)
METAL_BV_EACH=deepcopy(SORBATE)
BOND_LENGTH_EACH=deepcopy(SORBATE)
for i in range(len(LOCAL_STRUCTURE)):
    for j in range(len(LOCAL_STRUCTURE[i])):
        METAL_BV_EACH[i][j]=METAL_VALENCE[SORBATE[i][j]][0]/METAL_VALENCE[SORBATE[i][j]][1]#valence for each bond
        BOND_LENGTH_EACH[i][j]=R0_BV[(SORBATE[i][j],'O')]-np.log(METAL_BV_EACH[i][j])*0.37#ideal bond length using bond valence equation
        for key in LOCAL_STRUCTURE_MATCH_LIB.keys():
            if LOCAL_STRUCTURE[i][j] in LOCAL_STRUCTURE_MATCH_LIB[key]:
                LOCAL_STRUCTURE[i][j]=key
                break
            else:pass
#specify the searching range and penalty factor for surface atoms and sorbates
SEARCHING_PARS={'surface':[2.5,50],'sorbate':[[np.array(each)+SEARCH_RANGE_OFFSET for each in BOND_LENGTH_EACH],50]}#The value for each item [searching radius(A),scaling factor]
N_BOND,METAL_BV=setup_domain_hematite_rcut.setup_bv_condition(O_NUMBER,SORBATE_ATTACH_ATOM,METAL_BV_EACH,BV_OFFSET_SORBATE)
#Protonation of distal oxygens, any number in [0,1,2], where 1 means singly protonated, two means doubly protonated
PROTONATION_DISTAL_OXYGEN=[[0,0]]*len(pickup_index)

######################################print commands############################################
##want to make parameter table?##
TABLE=not running_mode
if TABLE:
    O_N,binding_mode=setup_domain_hematite_rcut.setup_fit_table(O_NUMBER,DOMAIN_NUMBER,SORBATE_ATTACH_ATOM)
    make_grid.make_structure(list(map(sum,SORBATE_NUMBER)),O_N,WATER_NUMBER,DOMAIN,Metal=SORBATE,binding_mode=binding_mode,long_slab=full_layer_pick,long_slab_HL=half_layer_pick,local_structure=LOCAL_STRUCTURE,add_distal_wild=ADD_DISTAL_LIGAND_WILD,use_domains=TABLE_DOMAINS,N_raxr=NUMBER_SPECTRA,domain_raxr_el=RESONANT_EL_LIST,layered_water=layered_water_pars['yes_OR_no'],layered_sorbate=layered_sorbate_pars['yes_OR_no'],tab_path=os.path.join(output_file_path,'table.tab'))
    print('Parameter table is saved!')
##want to output the data for plotting?##
PLOT=not running_mode
##want to print out the protonation status?##
PRINT_PROTONATION=not running_mode
##want to print bond valence?##
PRINT_BV=not running_mode
##want to print the xyz files to build a 3D structure?##
PRINT_MODEL_FILES=not running_mode

################################################build up ref domains############################################
#add atoms for bulk and two ref domains (ref_domain1<half layer> and ref_domain2<full layer>)                  #
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values      #
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted      #
################################################################################################################
batch_path_head=batch_path.module_path_locator()
domain_creator.add_atom_in_slab(bulk,os.path.join(batch_path_head,'hematite_rcut','bulk.str'))
domain_creator.add_atom_in_slab(ref_L_domain1,os.path.join(batch_path_head,'hematite_rcut','half_layer2.str'))
domain_creator.add_atom_in_slab(ref_S_domain1,os.path.join(batch_path_head,'hematite_rcut','half_layer3.str'))
domain_creator.add_atom_in_slab(ref_L_domain2,os.path.join(batch_path_head,'hematite_rcut','full_layer2.str'))
domain_creator.add_atom_in_slab(ref_S_domain2,os.path.join(batch_path_head,'hematite_rcut','full_layer3.str'))

##set up Fourier pars if there are RAXR datasets
#Fourier component looks like A_Dn0_n1, where n0, n1 are used to specify the index for domain, and spectra, respectively
#Each spectra will have its own set of A and P list, and each domain has its own set of P and A list
rgh_raxr,F1F2=setup_domain_hematite_rcut.setup_raxr_pars(NUMBER_SPECTRA,batch_path_head,F1F2_FILE,RESONANT_EL_LIST)

###################create domain classes and initiate the chemical equivalent domains####################
##id list according to the order in the reference domain (used to set up ref domain)
ref_id_list_L_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_S_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_S_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
##setup domains(initialization+add sorbates+add water layers)
for i in range(DOMAIN_NUMBER):
    vars()['HB_MATCH_'+str(i+1)]={}
    HB_MATCH=vars()['HB_MATCH_'+str(i+1)]
    if int(DOMAIN[i])==1:
        if half_layer_pick[i]==2:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_S_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_S_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
        elif half_layer_pick[i]==3:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_L_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_L_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    elif int(DOMAIN[i])==2:
        if full_layer_pick[i]==0:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_S_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_S_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
        elif full_layer_pick[i]==1:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_L_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_L_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']

    #Adding sorbates to domainA and domainB
    for j in range(sum(SORBATE_NUMBER[i])):
        SORBATE_coors_a=[]
        O_coors_a=[]
        if len(SORBATE_ATTACH_ATOM[i][j])==1:#monodentate case
            return_list=setup_domain_hematite_rcut.setup_sorbate_MD(vars(),i,j)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))],vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=return_list

        elif len(SORBATE_ATTACH_ATOM[i][j])==2:#bidentate case
            return_list=setup_domain_hematite_rcut.setup_sorbate_BD(vars(),i,j)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))],vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=return_list

        elif len(SORBATE_ATTACH_ATOM[i][j])==3:#tridentate case (no oxygen sorbate here considering it is a trigonal pyramid structure)
            return_list=setup_domain_hematite_rcut.setup_sorbate_TD(vars(),i,j)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))],vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=return_list

        else:#add an outer-sphere case here
            return_list=setup_domain_hematite_rcut.setup_sorbate_OS(vars(),i,j)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))],vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=return_list

    #setup water layers
    water_gp_names,water_groups=setup_domain_hematite_rcut.setup_water_layer(vars()['rgh_domain'+str(int(i+1))],vars()['Os_list_domain'+str(int(i+1))+'a'],vars()['Os_list_domain'+str(int(i+1))+'b'],vars()['domain_class_'+str(int(i+1))],vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B'],i,layered_water_pars,WATER_NUMBER,WATER_PAIR,REF_POINTS,layered_sorbate_pars)
    for iii in range(len(water_groups)):vars()[water_gp_names[iii]]=water_groups[iii]

######################################do grouping###############################################
#####################surface atom grouping+sorbate grouping#####################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together (4 atms grouped together in sequence grouping)
    #you may group in symmetry, then atoms of same layer are not independent. Know here the symmetry (equal opposite) is impressively defined in the function
    if DOMAIN[i]==1:
        if half_layer_pick[i]==3:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B']],layers_N=10)
        elif half_layer_pick[i]==2:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_7_0_D'+str(int(i+1))+'A','O1_1_1_D'+str(int(i+1))+'B']],layers_N=10)
    elif DOMAIN[i]==2:
        if full_layer_pick[i]==1:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_11_t_D'+str(int(i+1))+'A','O1_5_0_D'+str(int(i+1))+'B']],layers_N=10)
        elif full_layer_pick[i]==0:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_5_0_D'+str(int(i+1))+'A','O1_11_0_D'+str(int(i+1))+'B']],layers_N=10)

    #assign name to each group
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]

    #you may also only want to group each chemically equivalent atom from two domains (the use_sym is set to true here)
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]]))
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

    try:#group sorbates in pairs
        for N in range(0,sum(SORBATE_NUMBER[i]),2):
            vars()['gp_'+SORBATE_LIST[i][N]+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]*2,atom_ids=[SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'B',SORBATE_LIST[i][N]+str(N+2)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+2)+'_D'+str(i+1)+'B'],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.],[1.,0.,0.,0.,1.,0.,0.,0.,1.]])
            if vars()['HO_list_domain'+str(i+1)+'a']!=[]:
                HO_A1=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_B1=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_A2=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+2)+'_') in each]
                HO_B2=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+2)+'_') in each]
                for NN in range(len(HO_A1)):
                    vars()['gp_HO'+str(NN+1)+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B'],vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B']],\
                          atom_ids=[HO_A1[NN],HO_B1[NN],HO_A2[NN],HO_B2[NN]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[1,0,0,0,1,0,0,0,1]])
    except:#consider single site for each domain otherwise
        for N in range(0,sum(SORBATE_NUMBER[i]),1):
            vars()['gp_'+SORBATE_LIST[i][N]+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],atom_ids=[SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'B'],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]])
            if vars()['HO_list_domain'+str(i+1)+'a']!=[]:
                HO_A1=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_B1=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                for NN in range(len(HO_A1)):
                    vars()['gp_HO'+str(NN+1)+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B']],\
                          atom_ids=[HO_A1[NN],HO_B1[NN]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]])

#####################################do bond valence matching###################################
######################find the coordinating atoms for each constituent atom#####################
if USE_BV:
    for i in range(DOMAIN_NUMBER):
        lib_sorbate={}
        if SORBATE_NUMBER[i]!=0:
            lib_sorbate=domain_creator.create_sorbate_match_lib4_test(metal=SORBATE_LIST[i],HO_list=vars()['HO_list_domain'+str(int(i+1))+'a'],anchors=SORBATE_ATTACH_ATOM[i],anchor_offsets=SORBATE_ATTACH_ATOM_OFFSET[i],domain_tag=i+1)
        if DOMAIN[i]==1:
            if half_layer_pick[i]==3:
                vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            elif half_layer_pick[i]==2:
                vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_8_0_D'+str(int(i+1))+'A','Fe1_9_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)
        elif DOMAIN[i]==2:
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=None),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)

###################################fitting function part##########################################
VARS=vars()#pass local variables to sim function
if COUNT_TIME:t_1=datetime.now()

def Sim(data,VARS=VARS):
    for command in commands:eval(command)
    VARS=VARS
    F =[]
    bv=0
    bv_container={}
    fom_scaler=[]
    beta=rgh.beta
    SCALES=[getattr(rgh,scale) for scale in scales]
    total_wt=0
    domain={}

    for i in range(DOMAIN_NUMBER):
        #grap wt for each domain and cal the total wt
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]

        #update sorbates
        if UPDATE_SORBATE_IN_SIM:
            for j in range(sum(VARS['SORBATE_NUMBER'][i])):
                if len(VARS['SORBATE_ATTACH_ATOM'][i][j])==1 and not USE_COORS[i][j]:#monodentate case
                    setup_domain_hematite_rcut.update_sorbate_in_SIM_MD(VARS,domain_class_1,i,j,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST)
                elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==2 and not USE_COORS[i][j]:#bidentate case
                    setup_domain_hematite_rcut.update_sorbate_in_SIM_BD(VARS,domain_class_1,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,ANCHOR_REFERENCE,ANCHOR_REFERENCE_OFFSET,USE_TOP_ANGLE,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST)
                elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==3 and not USE_COORS[i][j]:#tridentate case
                    setup_domain_hematite_rcut.update_sorbate_in_SIM_TD(VARS,domain_class_1,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST,MIRROR)
                elif not USE_COORS[i][j]:#outer-sphere case
                    setup_domain_hematite_rcut.update_sorbate_in_SIM_OS(VARS,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,OS_X_REF,OS_Y_REF,OS_Z_REF,LOCAL_STRUCTURE,SORBATE_LIST)

        #updata water structure
        if WATER_NUMBER[i]!=0:#add water molecules if any
            if WATER_PAIR:
                for jj in range(int(WATER_NUMBER[i]/2)):#note will add water pair (two oxygens) each time, and you can't add single water
                    O_ids_a=VARS['Os_list_domain'+str(int(i+1))+'a'][jj*2:jj*2+2]
                    O_ids_b=VARS['Os_list_domain'+str(int(i+1))+'b'][jj*2:jj*2+2]
                    alpha=getattr(VARS['rgh_domain'+str(int(i+1))],'alpha_W_'+str(jj+1))
                    r=0.5*5.434/2./np.sin(alpha/180.*np.pi)#here r is constrained by the condition of y1-y2=0.5
                    v_shift=getattr(VARS['rgh_domain'+str(int(i+1))],'v_shift_W_'+str(jj+1))
                    #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                    H2O_coors_a=VARS['domain_class_'+str(int(i+1))].add_oxygen_pair2B(domain=VARS['domain'+str(int(i+1))+'A'],O_ids=O_ids_a,ref_id=list(map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj])),v_shift=v_shift,r=r,alpha=alpha)
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
            else:
                for jj in range(WATER_NUMBER[i]):#note will add single water each time
                    O_ids_a=[VARS['Os_list_domain'+str(int(i+1))+'a'][jj]]
                    O_ids_b=[VARS['Os_list_domain'+str(int(i+1))+'b'][jj]]
                    v_shift=getattr(VARS['rgh_domain'+str(int(i+1))],'v_shift_W_'+str(jj+1))
                    #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                    H2O_coors_a=VARS['domain_class_'+str(int(i+1))].add_single_oxygen(domain=VARS['domain'+str(int(i+1))+'A'],O_id=O_ids_a,ref_id=list(map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj])),v_shift=v_shift)
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O'])

        #calculate bv panelty factor
        if USE_BV and i in DOMAINS_BV:
            bv_temp,bv_container_temp=setup_domain_hematite_rcut.calculate_BV_sum_in_SIM(i,VARS,domain_class_1,BV_TOLERANCE,DOMAIN,SORBATE_NUMBER,SORBATE_LIST,\
                                                                               WATER_NUMBER,O_NUMBER,BOND_VALENCE_WAIVER,CONSIDER_WATER_IN_BV,\
                                                                               debug_bv,SORBATE_EL_LIST,SEARCH_MODE_FOR_SURFACE_ATOMS,SEARCH_RANGE_OFFSET,\
                                                                               IDEAL_BOND_LENGTH,SEARCH_RANGE_OFFSET,running_mode,R0_BV,\
                                                                               COUNT_DISTAL_OXYGEN,SEARCHING_PARS,METAL_BV,PRINT_BV,\
                                                                               COVALENT_HYDROGEN_RANDOM,COVALENT_HYDROGEN_ACCEPTOR,POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR,PRINT_PROTONATION,\
                                                                               POTENTIAL_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_NUMBER)
            bv=bv+bv_temp
            if debug_bv:
                for key in bv_container_temp.keys():bv_container[key]=bv_container_temp[key]
    if debug_bv:
        print("Print out the species, which are not under bond valence saturation")
        for i in bv_container.keys():
            if bv_container[i]!=0:
                print(i,"BV after considering penalty",bv_container[i])

    #set up multiple domains
    #note for each domain there are two sub domains which symmetrically related to each other, so have equivalent wt
    for i in range(DOMAIN_NUMBER):
        #extract domain weight
        wt_DA=getattr(VARS['rgh_domain'+str(int(i+1))],'wt_domainA')

        #extract layered water info
        layered_water_A,layered_water_B=[],[]
        if layered_water_pars['yes_OR_no'][i]:
            layered_water_A,layered_water_B=setup_domain_hematite_rcut.extract_layer_water_info(VARS,i,layered_water_pars)

        #extract layered sorbate info
        layered_sorbate_A,layered_sorbate_B=[],[]
        if layered_sorbate_pars['yes_OR_no'][i]:
            layered_sorbate_A,layered_sorbate_B=setup_domain_hematite_rcut.extract_layer_sorbate_info(VARS,i,layered_sorbate_pars,F1F2)

        #combine domain info together
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':wt_DA*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_A,'layered_sorbate':layered_sorbate_A}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':(1-wt_DA)*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_B,'layered_sorbate':layered_sorbate_B}

    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms=SURFACE_PARMS)

    if COUNT_TIME:t_2=datetime.now()

    #cal structure factor for each dataset in this for loop
    #fun to deal with the symmetrical shape of 10,30 and 20L rod at positive and negative sides
    def formate_hkl(h_,k_,x_):
        new_h,new_k,new_x=[],[],[]
        if np.around(h_[0],0) in [1,2,3] and np.around(k_[0],0)==0:
            for iii in range(len(x_)):
                if x_[iii]>0:
                    new_h.append(-h_[iii])
                    new_k.append(-k_[iii])
                    new_x.append(-x_[iii])
                else:
                    new_h.append(h_[iii])
                    new_k.append(k_[iii])
                    new_x.append(x_[iii])
            return  np.array(new_h),np.array(new_k),np.array(new_x)
        else:
            return np.array(h_),np.array(k_),np.array(x_)

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
                h_,k_,y_=formate_hkl(h,k,y)
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                f=sample.cal_structure_factor_hematite_RAXR(i,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h_, k_, y_, x, E0, F1F2,SCALES,rough)
                F.append(abs(f))
                fom_scaler.append(1)
                i+=1
            else:#doing CTR calculation (x is perpendicular momentum transfer L typically smaller than 15)
                h_,k_,x_=formate_hkl(h,k,x)
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
        #dummy_raxr_dataset_Pb_case.dat will be dumped to the output_file_path folder
    if COMBINE_DATA_SETS:
        domain_creator.combine_all_datasets(file=os.path.join(output_file_path,'temp_full_dataset.dat'),data=data)

    #The A and P list returned is calculated based on the model dependent structure#
    Print_AP=False
    if Print_AP:
        AP=sample.find_A_P(np.arange(0,10.38,0.35),'Pb',True)

    #export the model results for plotting if PLOT set to true#
    if PLOT:
        z_min,z_max=0,40
        setup_domain_hematite_rcut.plot_ctr_raxr_e_profiles(model,inst, bulk, domain, unitcell,COHERENCE,data,beta,VARS,E0,F1F2,RAXR_FIT_MODE,RESONANT_EL_LIST,SCALES,output_file_path,RAXR_EL,exp_const,rgh,re,auc,z_min,z_max,SURFACE_PARMS)

    #some ducumentation about using this script#
    print_help_info=False
    if print_help_info:
        setup_domain_hematite_rcut.print_help_doc()
    #do this in shell 'model.script_module.setup_domain_hematite_rcut.print_help_doc()' to get help info

    #output how fast the code is running#
    if COUNT_TIME:t_3=datetime.now()
    if COUNT_TIME:
        print("It took "+str(t_1-t_0)+" seconds to setup")
        print("It took "+str(t_2-t_1)+" seconds to calculate bv weighting")
        print("It took "+str(t_3-t_2)+" seconds to calculate structure factor")

    #you may play with the weighting rule by setting eg 2**bv, 5**bv for the wt factor, that way you are pushing the GenX to find a fit btween a good fit (low wt factor) and a reasonable fit (high wt factor)
    return F,1+WT_BV*bv,fom_scaler
