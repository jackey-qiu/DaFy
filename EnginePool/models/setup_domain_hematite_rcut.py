import models.domain_creator as domain_creator
from models.utils import UserVars
import os,pickle
import numpy as np

def setup_standard(HALF_LAYER_PICK_INDEX,FULL_LAYER_PICK_INDEX,N_HL,N_FL,SORBATE_EL_LIST,sym_site_index,pickup_index):
    SORBATE_ATTACHE_ATOM_EXTRA=[[['HO1_'],['HO1_']],[['HO2_'],['HO2_']],[['HO3_'],['HO3_']],[['HO1_','HO2_'],['HO1_','HO2_']],[['HO1_','HO3_'],['HO1_','HO3_']],[['HO2_','HO3_'],['HO2_','HO3_']],[['HO1_','HO2_','HO3_'],['HO1_','HO2_','HO3_']]]
    SORBATE_ATTACH_ATOM_HL_L=[[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']],[['O1_1_0','O1_4_0'],['O1_3_0','O1_2_0']],[['O1_1_0','O1_3_0'],['O1_4_0','O1_2_0']],[['O1_1_0','O1_4_0'],['O1_3_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0'],['O1_1_0','O1_2_0','O1_4_0']],[['O1_1_0','O1_3_0','O1_4_0'],['O1_2_0','O1_3_0','O1_4_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
    SORBATE_ATTACH_ATOM_HL_S=[[['O1_7_0','O1_8_0'],['O1_7_0','O1_8_0']],[['O1_7_0','O1_10_0'],['O1_9_0','O1_8_0']],[['O1_7_0','O1_9_0'],['O1_10_0','O1_8_0']],[['O1_7_0','O1_10_0'],['O1_9_0','O1_8_0']],[['O1_8_0','O1_7_0','O1_9_0'],['O1_8_0','O1_7_0','O1_10_0']],[['O1_7_0','O1_9_0','O1_10_0'],['O1_8_0','O1_9_0','O1_10_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
    SORBATE_ATTACH_ATOM_FL_L=[[['O1_11_t','O1_12_t'],['O1_11_t','O1_12_t']],[['O1_11_t','O1_1_0'],['O1_2_0','O1_12_t']],[['O1_11_t','O1_2_0'],['O1_1_0','O1_12_t']],[['O1_11_t','O1_1_0'],['O1_2_0','O1_12_t']],[['O1_11_t','O1_12_t','O1_2_0'],['O1_11_t','O1_12_t','O1_1_0']],[['O1_11_t','O1_2_0','O1_1_0'],['O1_12_t','O1_2_0','O1_1_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
    SORBATE_ATTACH_ATOM_FL_S=[[['O1_5_0','O1_6_0'],['O1_5_0','O1_6_0']],[['O1_5_0','O1_7_0'],['O1_8_0','O1_6_0']],[['O1_5_0','O1_8_0'],['O1_7_0','O1_6_0']],[['O1_7_0','O1_5_0'],['O1_6_0','O1_8_0']],[['O1_6_0','O1_5_0','O1_8_0'],['O1_6_0','O1_5_0','O1_7_0']],[['O1_5_0','O1_7_0','O1_8_0'],['O1_6_0','O1_7_0','O1_8_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
    SORBATE_ATTACH_ATOM_FL=domain_creator.pick_full_layer(LFL=SORBATE_ATTACH_ATOM_FL_L,SFL=SORBATE_ATTACH_ATOM_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    SORBATE_ATTACH_ATOM_HL=domain_creator.pick_half_layer(LHL=SORBATE_ATTACH_ATOM_HL_L,SHL=SORBATE_ATTACH_ATOM_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    SORBATE_ATTACH_ATOM_SEPERATED=[domain_creator.deep_pick(SORBATE_ATTACH_ATOM_HL_L+each_FL,sym_site_index,pickup_index) for each_FL in SORBATE_ATTACH_ATOM_FL]
    SORBATE_ATTACH_ATOM_SEPERATED_HL=[domain_creator.deep_pick(each_HL+SORBATE_ATTACH_ATOM_FL_L,sym_site_index,pickup_index) for each_HL in SORBATE_ATTACH_ATOM_HL]
    SORBATE_ATTACH_ATOM=[SORBATE_ATTACH_ATOM_SEPERATED_HL[i][i] for i in range(N_HL)]+[SORBATE_ATTACH_ATOM_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    SORBATE_ATTACH_ATOM_OFFSET_EXTRA=[[[None],[None]],[[None],[None]],[[None],[None]],[[None,None],[None,None]],[[None,None],[None,None]],[[None,None],[None,None]],[[None,None,None],[None,None,None]]]
    SORBATE_ATTACH_ATOM_OFFSET_HL_L=[[[None,None],[None,'+y']],[['-y','+x'],[None,None]],[[None,None],['+x',None]],[[None,'+y'],['+x',None]],[[None,None,None],['-y',None,'+x']],[[None,None,'+y'],['-x',None,None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
    SORBATE_ATTACH_ATOM_OFFSET_HL_S=[[[None,None],[None,'+y']],[['-y','-x'],[None,None]],[[None,None],['-x',None]],[[None,'+y'],[None,'+x']],[[None,None,None],[None,'-y','-x']],[[None,None,'+y'],['+x',None,None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
    SORBATE_ATTACH_ATOM_OFFSET_FL_L=[[[None,None],[None,'+y']],[[None,'-x'],[None,None]],[[None,'-x'],['-y',None]],[[None,None],['-x',None]],[[None,None,'-x'],[None,'+y',None]],[['+x',None,None],[None,None,'-y']],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
    SORBATE_ATTACH_ATOM_OFFSET_FL_S=[[[None,None],[None,'+y']],[[None,'+x'],[None,None]],[[None,'+x'],['-y',None]],[[None,None],[None,'+x']],[[None,None,'+x'],['+y',None,None]],[['-x',None,None],[None,'-y',None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
    SORBATE_ATTACH_ATOM_OFFSET_FL=domain_creator.pick_full_layer(LFL=SORBATE_ATTACH_ATOM_OFFSET_FL_L,SFL=SORBATE_ATTACH_ATOM_OFFSET_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    SORBATE_ATTACH_ATOM_OFFSET_HL=domain_creator.pick_half_layer(LHL=SORBATE_ATTACH_ATOM_OFFSET_HL_L,SHL=SORBATE_ATTACH_ATOM_OFFSET_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    SORBATE_ATTACH_ATOM_OFFSET_SEPERATED=[domain_creator.deep_pick(SORBATE_ATTACH_ATOM_OFFSET_HL_L+each_FL,sym_site_index,pickup_index) for each_FL in SORBATE_ATTACH_ATOM_OFFSET_FL]
    SORBATE_ATTACH_ATOM_OFFSET_SEPERATED_HL=[domain_creator.deep_pick(each_HL+SORBATE_ATTACH_ATOM_OFFSET_FL_L,sym_site_index,pickup_index) for each_HL in SORBATE_ATTACH_ATOM_OFFSET_HL]
    SORBATE_ATTACH_ATOM_OFFSET=[SORBATE_ATTACH_ATOM_OFFSET_SEPERATED_HL[i][i] for i in range(N_HL)]+[SORBATE_ATTACH_ATOM_OFFSET_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    ANCHOR_REFERENCE_EXTRA=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]
    ANCHOR_REFERENCE_HL_L=[[None,None],['Fe1_4_0','Fe1_6_0'],['Fe1_4_0','Fe1_6_0'],['Fe1_4_0','Fe1_6_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
    ANCHOR_REFERENCE_HL_S=[[None,None],['Fe1_10_0','Fe1_12_0'],['Fe1_10_0','Fe1_12_0'],['Fe1_10_0','Fe1_12_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA
    ANCHOR_REFERENCE_FL_L=[[None,None],['Fe1_2_0','Fe1_3_0'],['Fe1_2_0','Fe1_3_0'],['Fe1_2_0','Fe1_3_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
    ANCHOR_REFERENCE_FL_S=[[None,None],['Fe1_8_0','Fe1_9_0'],['Fe1_8_0','Fe1_9_0'],['Fe1_8_0','Fe1_9_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
    ANCHOR_REFERENCE_FL=domain_creator.pick_full_layer(LFL=ANCHOR_REFERENCE_FL_L,SFL=ANCHOR_REFERENCE_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    ANCHOR_REFERENCE_HL=domain_creator.pick_half_layer(LHL=ANCHOR_REFERENCE_HL_L,SHL=ANCHOR_REFERENCE_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    ANCHOR_REFERENCE_SEPERATED=[domain_creator.deep_pick(ANCHOR_REFERENCE_HL_L+each_FL,sym_site_index,pickup_index) for each_FL in ANCHOR_REFERENCE_FL]
    ANCHOR_REFERENCE_SEPERATED_HL=[domain_creator.deep_pick(each_HL+ANCHOR_REFERENCE_FL_L,sym_site_index,pickup_index) for each_HL in ANCHOR_REFERENCE_HL]
    ANCHOR_REFERENCE=[ANCHOR_REFERENCE_SEPERATED_HL[i][i] for i in range(N_HL)]+[ANCHOR_REFERENCE_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    ANCHOR_REFERENCE_OFFSET_EXTRA=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]
    ANCHOR_REFERENCE_OFFSET_HL_L=[[None,None],['-y','+x'],[None,'+x'],[None,'+x'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
    ANCHOR_REFERENCE_OFFSET_HL_S=[[None,None],['-y',None],[None,None],[None,'+x'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
    ANCHOR_REFERENCE_OFFSET_FL_L=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
    ANCHOR_REFERENCE_OFFSET_FL_S=[[None,None],['+x',None],['+x',None],['+x',None],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
    ANCHOR_REFERENCE_OFFSET_FL=domain_creator.pick_full_layer(LFL=ANCHOR_REFERENCE_OFFSET_FL_L,SFL=ANCHOR_REFERENCE_OFFSET_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    ANCHOR_REFERENCE_OFFSET_HL=domain_creator.pick_half_layer(LHL=ANCHOR_REFERENCE_OFFSET_HL_L,SHL=ANCHOR_REFERENCE_OFFSET_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    ANCHOR_REFERENCE_OFFSET_SEPERATED=[domain_creator.deep_pick(ANCHOR_REFERENCE_OFFSET_HL_L+each_FL,sym_site_index,pickup_index) for each_FL in ANCHOR_REFERENCE_OFFSET_FL]
    ANCHOR_REFERENCE_OFFSET_SEPERATED_HL=[domain_creator.deep_pick(each_HL+ANCHOR_REFERENCE_OFFSET_FL_L,sym_site_index,pickup_index) for each_HL in ANCHOR_REFERENCE_OFFSET_HL]
    ANCHOR_REFERENCE_OFFSET=[ANCHOR_REFERENCE_OFFSET_SEPERATED_HL[i][i] for i in range(N_HL)]+[ANCHOR_REFERENCE_OFFSET_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    #if consider hydrogen bonds#
    #Arbitrary number of distal oxygens(6 here) will be helpful and handy if you want to consider the distal oxygen for bond valence constrain in a random mode, sine you wont need extra edition for that.
    #It wont hurt even if the distal oxygen in the list doesn't actually exist for your model. Same for the potential hydrogen acceptor below
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA=[['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*7
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL=domain_creator.pick_full_layer(LFL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L,SFL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL=domain_creator.pick_half_layer(LHL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L,SHL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED=[domain_creator.pick(POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L+each_FL,pickup_index) for each_FL in POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL]
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL=[domain_creator.pick(each_HL+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L,pickup_index) for each_HL in POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL]
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR=[POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    COVALENT_HYDROGEN_ACCEPTOR_EXTRA=[[None]]*7
    COVALENT_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
    COVALENT_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA
    COVALENT_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
    COVALENT_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
    COVALENT_HYDROGEN_ACCEPTOR_FL=domain_creator.pick_full_layer(LFL=COVALENT_HYDROGEN_ACCEPTOR_FL_L,SFL=COVALENT_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    COVALENT_HYDROGEN_ACCEPTOR_HL=domain_creator.pick_half_layer(LHL=COVALENT_HYDROGEN_ACCEPTOR_HL_L,SHL=COVALENT_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    COVALENT_HYDROGEN_ACCEPTOR_SEPERATED=[domain_creator.pick(COVALENT_HYDROGEN_ACCEPTOR_HL_L+each_FL,pickup_index) for each_FL in COVALENT_HYDROGEN_ACCEPTOR_FL]
    COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL=[domain_creator.pick(each_HL+COVALENT_HYDROGEN_ACCEPTOR_FL_L,pickup_index) for each_HL in COVALENT_HYDROGEN_ACCEPTOR_HL]
    COVALENT_HYDROGEN_ACCEPTOR=[COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[COVALENT_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]

    COVALENT_HYDROGEN_NUMBER_EXTRA=[[None]]*7
    COVALENT_HYDROGEN_NUMBER_HL=[[1,1,1,1],[2,1,0,1],[2,1,1,0],[2,1,0,1],[1,1,1,0],[2,1,0,0],[2,2,1,1],[2,2,1,1]]+COVALENT_HYDROGEN_NUMBER_EXTRA
    COVALENT_HYDROGEN_NUMBER_FL=[[1,1,1,1],[2,1,1,0],[2,1,0,1],[2,1,1,0],[1,1,0,1],[2,1,0,0],[2,2,1,1],[2,2,1,1]]+COVALENT_HYDROGEN_NUMBER_EXTRA
    COVALENT_HYDROGEN_NUMBER=domain_creator.pick(COVALENT_HYDROGEN_NUMBER_HL+COVALENT_HYDROGEN_NUMBER_FL,pickup_index)

    POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA=[['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*7
    POTENTIAL_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
    POTENTIAL_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA
    POTENTIAL_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
    POTENTIAL_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
    POTENTIAL_HYDROGEN_ACCEPTOR_FL=domain_creator.pick_full_layer(LFL=POTENTIAL_HYDROGEN_ACCEPTOR_FL_L,SFL=POTENTIAL_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    POTENTIAL_HYDROGEN_ACCEPTOR_HL=domain_creator.pick_half_layer(LHL=POTENTIAL_HYDROGEN_ACCEPTOR_HL_L,SHL=POTENTIAL_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED=[domain_creator.pick(POTENTIAL_HYDROGEN_ACCEPTOR_HL_L+each_FL,pickup_index) for each_FL in POTENTIAL_HYDROGEN_ACCEPTOR_FL]
    POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED_HL=[domain_creator.pick(each_HL+POTENTIAL_HYDROGEN_ACCEPTOR_FL_L,pickup_index) for each_HL in POTENTIAL_HYDROGEN_ACCEPTOR_HL]
    POTENTIAL_HYDROGEN_ACCEPTOR=[POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]
    return SORBATE_ATTACH_ATOM,SORBATE_ATTACH_ATOM_OFFSET,ANCHOR_REFERENCE,ANCHOR_REFERENCE_OFFSET,POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_NUMBER,POTENTIAL_HYDROGEN_ACCEPTOR


def setup_water_pars(water_pars,N_HL,N_FL,pickup_index,FULL_LAYER_PICK_INDEX,HALF_LAYER_PICK_INDEX):
    ##pars for interfacial waters##
    WATER_NUMBER=None
    REF_POINTS=None

    if not water_pars['use_default']:
        WATER_NUMBER=water_pars['number']
        REF_POINTS=water_pars['ref_point']
    else:
        WATER_NUMBER=domain_creator.pick([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],pickup_index)
        REF_POINTS_HL_L=[[['O1_1_0','O1_2_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
        REF_POINTS_HL_S=[[['O1_7_0','O1_8_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
        REF_POINTS_FL_L=[[['O1_11_t','O1_12_t']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
        REF_POINTS_FL_S=[[['O1_5_0','O1_6_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
        REF_POINTS_FL=domain_creator.pick_full_layer(LFL=REF_POINTS_FL_L,SFL=REF_POINTS_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
        REF_POINTS_HL=domain_creator.pick_half_layer(LHL=REF_POINTS_HL_L,SHL=REF_POINTS_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
        REF_POINTS_SEPERATED=[domain_creator.pick(REF_POINTS_HL_L+each_FL,pickup_index) for each_FL in REF_POINTS_FL]
        REF_POINTS_SEPERATED_HL=[domain_creator.pick(each_HL+REF_POINTS_FL_L,pickup_index) for each_HL in REF_POINTS_HL]
        REF_POINTS=[REF_POINTS_SEPERATED_HL[i][i] for i in range(N_HL)]+[REF_POINTS_SEPERATED[i][N_HL+i] for i in range(N_FL)]+[[None]]*7#each item inside is a list of one or couple items, and each water set has its own ref point
    return WATER_NUMBER,REF_POINTS

def setup_bv_condition(O_NUMBER,SORBATE_ATTACH_ATOM,METAL_BV_EACH,BV_OFFSET_SORBATE):
    #specify the METAL_BV based on the metal valence charge and the coordinated local structure
    N_BOND=[]
    METAL_BV=[]
    for i in range(len(O_NUMBER)):
        temp_box=[]
        for j in range(len(O_NUMBER[i])):
            temp_box.append(O_NUMBER[i][j]+len(SORBATE_ATTACH_ATOM[i][j]))
        N_BOND.append(temp_box)
    for i in range(len(N_BOND)):
        temp_box=[]
        for j in range(len(N_BOND[i])):
            if len(METAL_BV_EACH[i])!=len(N_BOND[i]):
                temp_box.append([METAL_BV_EACH[i][int(j/2)]*N_BOND[i][j]-BV_OFFSET_SORBATE[i][j],METAL_BV_EACH[i][int(j/2)]*N_BOND[i][j]])
            else:
                temp_box.append([METAL_BV_EACH[i][j]*N_BOND[i][j]-BV_OFFSET_SORBATE[i][j],METAL_BV_EACH[i][j]*N_BOND[i][j]])
        METAL_BV.append(temp_box)
    return N_BOND,METAL_BV

def setup_fit_table(O_NUMBER,DOMAIN_NUMBER,SORBATE_ATTACH_ATOM):
    O_N=[]
    binding_mode=[]
    for i in O_NUMBER:
        temp=[]
        for j in range(0,len(i),2):
            temp.append(i[j])
        O_N.append(temp)
    for i in range(DOMAIN_NUMBER):
        temp_binding_mode=[]
        for j in range(0,len(SORBATE_ATTACH_ATOM[i]),2):
            if SORBATE_ATTACH_ATOM[i][j]==[]:
                temp_binding_mode.append('OS')
            else:
                if len(SORBATE_ATTACH_ATOM[i][j])==1:
                    temp_binding_mode.append('MD')
                elif len(SORBATE_ATTACH_ATOM[i][j])==2:
                    temp_binding_mode.append('BD')
                elif len(SORBATE_ATTACH_ATOM[i][j])==3:
                    temp_binding_mode.append('TD')
        binding_mode.append(temp_binding_mode)
    return O_N,binding_mode

def setup_atm_ids(DOMAIN_NUMBER,DOMAIN,SORBATE,SORBATE_NUMBER,SORBATE_LIST,WATER_NUMBER,O_NUMBER,half_layer_pick,full_layer_pick):
    ##############################################set up atm ids###############################################
    names=[]
    vars_container=[]
    for i in range(DOMAIN_NUMBER):
        ##user defined variables
        vars()['rgh_domain'+str(int(i+1))]=UserVars()
        vars()['rgh_domain'+str(int(i+1))].new_var('wt', 1.)
        vars()['rgh_domain'+str(int(i+1))].new_var('wt_domainA', 0.5)
        names.append('rgh_domain'+str(int(i+1)))
        vars_container.append(vars()['rgh_domain'+str(int(i+1))])

        ##sorbate list (HO is oxygen binded to pb and Os is water molecule)
        vars()['SORBATE_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids2(el=SORBATE[i],N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'A')
        vars()['SORBATE_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids2(el=SORBATE[i],N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'B')
        names.append('SORBATE_list_domain'+str(int(i+1))+'a')
        vars_container.append(vars()['SORBATE_list_domain'+str(int(i+1))+'a'])
        names.append('SORBATE_list_domain'+str(int(i+1))+'b')
        vars_container.append(vars()['SORBATE_list_domain'+str(int(i+1))+'b'])

        vars()['HO_list_domain'+str(int(i+1))+'a']=domain_creator.create_HO_ids3(anchor_els=SORBATE_LIST[i],O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'A')
        vars()['HO_list_domain'+str(int(i+1))+'b']=domain_creator.create_HO_ids3(anchor_els=SORBATE_LIST[i],O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'B')
        names.append('HO_list_domain'+str(int(i+1))+'a')
        vars_container.append(vars()['HO_list_domain'+str(int(i+1))+'a'])
        names.append('HO_list_domain'+str(int(i+1))+'b')
        vars_container.append(vars()['HO_list_domain'+str(int(i+1))+'b'])

        vars()['Os_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'A')
        vars()['Os_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'B')
        names.append('Os_list_domain'+str(int(i+1))+'a')
        vars_container.append(vars()['Os_list_domain'+str(int(i+1))+'a'])
        names.append('Os_list_domain'+str(int(i+1))+'b')
        vars_container.append(vars()['Os_list_domain'+str(int(i+1))+'b'])

        vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['SORBATE_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['Os_list_domain'+str(int(i+1))+'a']
        vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['SORBATE_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+vars()['Os_list_domain'+str(int(i+1))+'b']
        names.append('sorbate_ids_domain'+str(int(i+1))+'a')
        vars_container.append(vars()['sorbate_ids_domain'+str(int(i+1))+'a'])
        names.append('sorbate_ids_domain'+str(int(i+1))+'b')
        vars_container.append(vars()['sorbate_ids_domain'+str(int(i+1))+'b'])

        vars()['sorbate_els_domain'+str(int(i+1))]=SORBATE_LIST[i]+['O']*(sum([np.sum(N_list) for N_list in O_NUMBER[i]])+WATER_NUMBER[i])
        names.append('sorbate_els_domain'+str(int(i+1)))
        vars_container.append(vars()['sorbate_els_domain'+str(int(i+1))])

        ##set up group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
        #atom ids for grouping(containerB must be the associated chemically equivalent atoms)
        equivalent_atm_list_A_L_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0"]
        equivalent_atm_list_A_S_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1"]

        equivalent_atm_list_A_S_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1"]
        equivalent_atm_list_A_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0"]

        equivalent_atm_list_B_L_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1"]
        equivalent_atm_list_B_S_1=["O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1","O1_7_1","O1_8_1","Fe1_8_1","Fe1_9_1","O1_9_1","O1_10_1","Fe1_10_1","Fe1_12_1","O1_11_1","O1_12_1"]
        equivalent_atm_list_B_S_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1","O1_7_1","O1_8_1","Fe1_8_1","Fe1_9_1","O1_9_1","O1_10_1","Fe1_10_1","Fe1_12_1"]
        equivalent_atm_list_B_L_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1"]

        atm_sequence_gp_names_L_1=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
        atm_sequence_gp_names_S_1=['O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12']
        atm_sequence_gp_names_S_2=['O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12']
        atm_sequence_gp_names_L_2=['O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6']

        atm_list_A_L_1=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0']
        atm_list_A_S_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_10_0','Fe1_12_0','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1',]
        atm_list_A_S_2=['O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0','Fe1_4_1','Fe1_6_1']
        atm_list_A_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0",'O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0']

        atm_list_B_L_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_10_0','Fe1_12_0','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1']
        atm_list_B_S_1=['O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1','Fe1_10_1','Fe1_12_1']
        atm_list_B_S_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1",'O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1']
        atm_list_B_L_2=['O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0','Fe1_4_1','Fe1_6_1']

        if int(DOMAIN[i])==1:
            tag=None
            if half_layer_pick[i]==2:
                tag='S'
            elif half_layer_pick[i]==3:
                tag='L'
            vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+list(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))]))
            vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+list(map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('ids_domain'+str(int(i+1))+'A')
            vars_container.append(vars()['ids_domain'+str(int(i+1))+'A'])
            names.append('ids_domain'+str(int(i+1))+'B')
            vars_container.append(vars()['ids_domain'+str(int(i+1))+'B'])

            vars()['discrete_gp_names_domain'+str(int(i+1))]=list(map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a']))+\
                                                         list(map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])))
            vars()['sequence_gp_names_domain'+str(int(i+1))]=list(map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('discrete_gp_names_domain'+str(int(i+1)))
            vars_container.append(vars()['discrete_gp_names_domain'+str(int(i+1))])
            names.append('sequence_gp_names_domain'+str(int(i+1)))
            vars_container.append(vars()['sequence_gp_names_domain'+str(int(i+1))])

            vars()['atm_list_'+str(int(i+1))+'A']=list(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))]))
            vars()['atm_list_'+str(int(i+1))+'B']=list(map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('atm_list_'+str(int(i+1))+'A')
            vars_container.append(vars()['atm_list_'+str(int(i+1))+'A'])
            names.append('atm_list_'+str(int(i+1))+'B')
            vars_container.append(vars()['atm_list_'+str(int(i+1))+'B'])


        elif int(DOMAIN[i])==2:
            tag=None
            if full_layer_pick[i]==0:
                tag='S'
            elif full_layer_pick[i]==1:
                tag='L'
            vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+list(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))]))
            vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+list(map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('ids_domain'+str(int(i+1))+'A')
            vars_container.append(vars()['ids_domain'+str(int(i+1))+'A'])
            names.append('ids_domain'+str(int(i+1))+'B')
            vars_container.append(vars()['ids_domain'+str(int(i+1))+'B'])

            vars()['discrete_gp_names_domain'+str(int(i+1))]=list(map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a']))+\
                                                         list(map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])))
            vars()['sequence_gp_names_domain'+str(int(i+1))]=list(map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('discrete_gp_names_domain'+str(int(i+1)))
            vars_container.append(vars()['discrete_gp_names_domain'+str(int(i+1))])
            names.append('sequence_gp_names_domain'+str(int(i+1)))
            vars_container.append(vars()['sequence_gp_names_domain'+str(int(i+1))])

            vars()['atm_list_'+str(int(i+1))+'A']=list(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))]))
            vars()['atm_list_'+str(int(i+1))+'B']=list(map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
            names.append('atm_list_'+str(int(i+1))+'A')
            vars_container.append(vars()['atm_list_'+str(int(i+1))+'A'])
            names.append('atm_list_'+str(int(i+1))+'B')
            vars_container.append(vars()['atm_list_'+str(int(i+1))+'B'])
    return names,vars_container

def setup_raxr_pars(NUMBER_SPECTRA,batch_path_head,F1F2_FILE,RESONANT_EL_LIST):
    if NUMBER_SPECTRA!=0:
        F1F2=np.loadtxt(os.path.join(batch_path_head,F1F2_FILE))
        rgh_raxr=UserVars()
        for i in range(NUMBER_SPECTRA):
            rgh_raxr.new_var('a'+str(i+1),0.0)
            rgh_raxr.new_var('b'+str(i+1),0.0)
            rgh_raxr.new_var('c'+str(i+1),0.0)
            for j in range(len(RESONANT_EL_LIST)):
                if RESONANT_EL_LIST[j]!=0:
                    rgh_raxr.new_var('A_D'+str(j+1)+'_'+str(i+1),2.0)
                    rgh_raxr.new_var('P_D'+str(j+1)+'_'+str(i+1),0.0)
    else:
        rgh_raxr=None
        F1F2=None
    return rgh_raxr,F1F2

def setup_sorbate_MD(VARS,i,j):

    rgh,\
    LOCAL_STRUCTURE,\
    ADD_DISTAL_LIGAND_WILD,\
    SORBATE_ATTACH_ATOM,\
    SORBATE_ATTACH_ATOM_OFFSET,\
    sorbate_list_a,\
    sorbate_list_b,\
    HO_list_a,\
    HO_list_b,\
    USE_COORS,COORS,\
    domain_class,\
    domainA,domainB,\
    MIRROR,\
    SORBATE_LIST,\
    SORBATE_coors_a,\
    O_coors_a\
    =\
    VARS['rgh_domain'+str(int(i+1))],\
    VARS['LOCAL_STRUCTURE'],VARS['ADD_DISTAL_LIGAND_WILD'],\
    VARS['SORBATE_ATTACH_ATOM'],VARS['SORBATE_ATTACH_ATOM_OFFSET'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'a'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'b'],\
    VARS['HO_list_domain'+str(int(i+1))+'a'],\
    VARS['HO_list_domain'+str(int(i+1))+'b'],\
    VARS['USE_COORS'],VARS['COORS'],\
    VARS['domain_class_'+str(int(i+1))],\
    VARS['domain'+str(int(i+1))+'A'],\
    VARS['domain'+str(int(i+1))+'B'],\
    VARS['MIRROR'],VARS['SORBATE_LIST'],\
    VARS['SORBATE_coors_a'],\
    VARS['O_coors_a']

    if j%2==0:
        rgh.new_var('top_angle_MD_'+str(j), 71.)
        rgh.new_var('phi_MD_'+str(j), 0.)
        rgh.new_var('r_MD_'+str(j), 2.)
    if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            [rgh.new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(2)]
            [rgh.new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(2)]
            [rgh.new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(2)]
    if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            [rgh.new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(3)]
            [rgh.new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(3)]
            [rgh.new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(3)]
    if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='octahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            [rgh.new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(5)]
            [rgh.new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(5)]
            [rgh.new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(5)]
    ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j][0]
    if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
        ids=SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
    else:
        ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']
    SORBATE_id=sorbate_list_a[j]#pb_id is a str NOT list
    O_id=[HO_id for HO_id in HO_list_a if SORBATE_id in HO_id]
    sorbate_coors=[]
    if USE_COORS[i][j]:
        sorbate_coors=COORS[(i,j)]['sorbate'][0]+COORS[i]['oxygen'][0]
    else:
        if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
            sorbate_coors=domain_class.adding_sorbate_pyramid_monodentate(domain=domainA,top_angle=70,phi=0,r=2,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,O_id=O_id,mirror=MIRROR[i],sorbate_el=SORBATE_LIST[i][j])
        elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
            sorbate_coors=domain_class.adding_sorbate_octahedral_monodentate(domain=domainA,phi=0,r=2,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
        elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
            sorbate_coors=domain_class.adding_sorbate_tetrahedral_monodentate(domain=domainA,phi=0.,r=2.25,attach_atm_id=ids,offset=offset,sorbate_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
    SORBATE_coors_a.append(sorbate_coors[0])
    if O_id!=[]:
        [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_id_B=sorbate_list_b[j]
    O_id_B=[HO_id for HO_id in HO_list_b if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    if USE_COORS[i][j]:
        SORBATE_id_A=sorbate_list_a[j]
        O_id_A=[HO_id for HO_id in HO_list_a if SORBATE_id_A in HO_id]
        sorbate_ids_A=[SORBATE_id_A]+O_id_A
        domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
    domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
    #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
    sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
    HO_set_ids=O_id+O_id_B
    N=len(sorbate_set_ids)/2
    M=len(O_id)
    gp_sorbates_set=domain_class.grouping_discrete_layer3(domain=[domainA]*N+[domainB]*N,atom_ids=sorbate_set_ids)
    if M!=0:
        gp_HO_set=domain_class.grouping_discrete_layer3(domain=[domainA]*M+[domainB]*M,atom_ids=HO_set_ids)
    return gp_sorbates_set,gp_HO_set

def update_sorbate_in_SIM_MD(VARS,domain_class_1,i,j,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST):
    SORBATE_coors_a=[]
    O_coors_a=[]
    top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_MD_'+str(j/2*2))
    phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_MD_'+str(j/2*2))
    r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_MD_'+str(j/2*2))
    ids=VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A'
    if 'HO' in ids:
        ids=VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
    offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j][0]
    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
    #O_id=VARS['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]#O_ide is a list of str
    sorbate_coors=[]
    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,r=r,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=VARS['MIRROR'][i][j])
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,r=r,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id)
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_tetrahedral_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,r=r,attach_atm_id=ids,offset=offset,sorbate_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_coors_a.append(sorbate_coors[0])
    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    return None

def setup_sorbate_BD(VARS,i,j):
    rgh,\
    LOCAL_STRUCTURE,\
    ADD_DISTAL_LIGAND_WILD,\
    SORBATE_ATTACH_ATOM,\
    SORBATE_ATTACH_ATOM_OFFSET,\
    sorbate_list_a,\
    sorbate_list_b,\
    HO_list_a,\
    HO_list_b,\
    USE_COORS,\
    COORS,\
    domain_class,\
    domainA,\
    domainB,\
    MIRROR,\
    SORBATE_LIST,\
    SORBATE_coors_a,\
    O_coors_a,\
    BASAL_EL,\
    ANCHOR_REFERENCE,\
    ANCHOR_REFERENCE_OFFSET\
    =\
    VARS['rgh_domain'+str(int(i+1))],\
    VARS['LOCAL_STRUCTURE'],VARS['ADD_DISTAL_LIGAND_WILD'],\
    VARS['SORBATE_ATTACH_ATOM'],VARS['SORBATE_ATTACH_ATOM_OFFSET'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'a'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'b'],\
    VARS['HO_list_domain'+str(int(i+1))+'a'],\
    VARS['HO_list_domain'+str(int(i+1))+'b'],\
    VARS['USE_COORS'],VARS['COORS'],\
    VARS['domain_class_'+str(int(i+1))],\
    VARS['domain'+str(int(i+1))+'A'],\
    VARS['domain'+str(int(i+1))+'B'],\
    VARS['MIRROR'],VARS['SORBATE_LIST'],\
    VARS['SORBATE_coors_a'],VARS['O_coors_a'],\
    VARS['BASAL_EL'],VARS['ANCHOR_REFERENCE'],VARS['ANCHOR_REFERENCE_OFFSET']

    if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            rgh.new_var('offset_BD_'+str(j), 0.)
            rgh.new_var('offset2_BD_'+str(j), 0.)
            rgh.new_var('angle_offset_BD_'+str(j), 0.)
            rgh.new_var('phi_BD_'+str(j), 0.)
            rgh.new_var('top_angle_BD_'+str(j), 70.)
            rgh.new_var('r_BD_'+str(j), 2.27)
            [rgh.new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(1)]
            [rgh.new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(1)]
            [rgh.new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(1)]
        else:
            rgh.new_var('offset_BD_'+str(j), 0.)
            rgh.new_var('offset2_BD_'+str(j), 0.)
            rgh.new_var('angle_offset_BD_'+str(j), 0.)
            rgh.new_var('phi_BD_'+str(j), 0.)
            rgh.new_var('top_angle_BD_'+str(j), 70.)
            rgh.new_var('r_BD_'+str(j), 2.27)
    elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='octahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            rgh.new_var('phi_BD_'+str(j), 0.)
            [rgh.new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(4)]
            [rgh.new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(4)]
            [rgh.new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(4)]
        else:
            rgh.new_var('phi_BD_'+str(j), 0.)
    elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            rgh.new_var('phi_BD_'+str(j), 0.)
            rgh.new_var('anchor_offset_BD_'+str(j), 0.)
            rgh.new_var('top_angle_offset_BD_'+str(j), 0.)
            [rgh.new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(2)]
            [rgh.new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(2)]
            [rgh.new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(2)]
        else:
            rgh.new_var('anchor_offset_BD_'+str(j), 0.)
            rgh.new_var('offset_BD_'+str(j), 0.)
            rgh.new_var('offset2_BD_'+str(j), 0.)
            rgh.new_var('top_angle_offset_BD_'+str(j), 0.)
            rgh.new_var('angle_offset_BD_'+str(j), 0.)
            rgh.new_var('angle_offset2_BD_'+str(j), 0.)
            rgh.new_var('phi_BD_'+str(j), 0.)

    ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
    anchor,anchor_offset=None,None
    if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
        ids=[SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
        anchor=BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
        anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
    else:
        ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']
        if ANCHOR_REFERENCE[i][j]!=None:
            anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
            anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]

    SORBATE_id=sorbate_list_a[j]
    O_id=[HO_id for HO_id in HO_list_a if SORBATE_id in HO_id]
    sorbate_coors=[]
    if USE_COORS[i][j]:
        sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
    else:
        if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
            sorbate_coors=domain_class.adding_sorbate_pyramid_distortion_B(domain=domainA,top_angle=70,phi=0,edge_offset=[0,0],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=MIRROR[i][j/2])
        elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
            sorbate_coors=domain_class.adding_sorbate_bidentate_octahedral(domain=domainA,phi=90,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
        elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
            sorbate_coors=domain_class.adding_sorbate_bidentate_tetrahedral(domain=domainA,phi=0,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
    SORBATE_coors_a.append(sorbate_coors[0])
    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_id_B=sorbate_list_b[j]
    O_id_B=[HO_id for HO_id in HO_list_b if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    if USE_COORS[i][j]:
        SORBATE_id_A=sorbate_list_a[j]
        O_id_A=[HO_id for HO_id in HO_list_a if SORBATE_id_A in HO_id]
        sorbate_ids_A=[SORBATE_id_A]+O_id_A
        domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
    domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
    #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
    sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
    HO_set_ids=O_id+O_id_B
    N=len(sorbate_set_ids)/2
    M=len(O_id)
    gp_sorbates_set=domain_class.grouping_discrete_layer3(domain=[domainA]*N+[domainB]*N,atom_ids=sorbate_set_ids)
    if M!=0:
        gp_HO_set=domain_class.grouping_discrete_layer3(domain=[domainA]*M+[domainB]*M,atom_ids=HO_set_ids)
    return gp_sorbates_set,gp_HO_set

def update_sorbate_in_SIM_BD(VARS,domain_class_1,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,ANCHOR_REFERENCE,ANCHOR_REFERENCE_OFFSET,USE_TOP_ANGLE,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST):
    SORBATE_coors_a=[]
    O_coors_a=[]
    ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
    anchor,anchor_offset=None,None
    if "HO" in VARS['SORBATE_ATTACH_ATOM'][i][j][0]:#a sign for ternary complex structure forming
        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
        anchor=BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
        anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
    else:
        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A']
        if ANCHOR_REFERENCE[i][j]!=None:
            anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
            anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]

    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
    sorbate_coors=[]
    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
        if (i+j)%2==1:edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
        else:edge_offset=-getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
        edge_offset2=getattr(VARS['rgh_domain'+str(int(i+1))],'offset2_BD_'+str(j/2*2))
        angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset_BD_'+str(j/2*2))
        top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_BD_'+str(j/2*2))
        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
        if not USE_TOP_ANGLE:
            r1=getattr(VARS['rgh_domain'+str(int(i+1))],'r_BD_'+str(j/2*2))
            r2=r1+getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
            l=domain_creator.extract_coor_offset(domain=VARS['domain'+str(int(i+1))+'A'],id=ids,offset=offset,basis=[5.038,5.434,7.3707])
            top_angle=np.arccos((r1**2+r2**2-l**2)/2/r1/r2)/np.pi*180
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion_B2(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,edge_offset=[edge_offset,edge_offset2],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=VARS['MIRROR'][i][j/2],angle_offset=angle_offset)
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=[],anchor_ref=anchor,anchor_offset=anchor_offset)
            if (i+j)%2==1:
                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
            else:
                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
        else:
            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
        if (i+j)%2==1:
            ids=ids[::-1]
            offset=offset[::-1]
            phi=-phi
        top_angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_offset_BD_'+str(j/2*2))
        edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'anchor_offset_BD_'+str(j/2*2))

        if ADD_DISTAL_LIGAND_WILD[i][j]:
            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,distal_length_offset=[0,0],distal_angle_offset=[0,0],top_angle_offset=top_angle_offset,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=[],anchor_ref=anchor,anchor_offset=anchor_offset,edge_offset=edge_offset)
            if (i+j)%2==1:
                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
            else:
                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
        else:
            angle_offsets=[getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset_BD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset2_BD_'+str(j/2*2))]
            distal_length_offset=[getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'offset2_BD_'+str(j/2*2))]
            if (i+j)%2==1:
                distal_length_offset=distal_length_offset[::-1]
                angle_offsets=-np.array(angle_offsets[::-1])
            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,distal_length_offset=distal_length_offset,distal_angle_offset=angle_offsets,top_angle_offset=top_angle_offset,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset,edge_offset=edge_offset)
    SORBATE_coors_a.append(sorbate_coors[0])
    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    return None

def setup_sorbate_TD(VARS,i,j):
    rgh,\
    LOCAL_STRUCTURE,\
    ADD_DISTAL_LIGAND_WILD,\
    SORBATE_ATTACH_ATOM,\
    SORBATE_ATTACH_ATOM_OFFSET,\
    sorbate_list_a,\
    sorbate_list_b,\
    HO_list_a,\
    HO_list_b,\
    USE_COORS,\
    COORS,\
    domain_class,\
    domainA,\
    domainB,\
    MIRROR,\
    SORBATE_LIST,\
    SORBATE_coors_a,\
    O_coors_a,\
    BASAL_EL\
    =\
    VARS['rgh_domain'+str(int(i+1))],\
    VARS['LOCAL_STRUCTURE'],VARS['ADD_DISTAL_LIGAND_WILD'],\
    VARS['SORBATE_ATTACH_ATOM'],VARS['SORBATE_ATTACH_ATOM_OFFSET'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'a'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'b'],\
    VARS['HO_list_domain'+str(int(i+1))+'a'],\
    VARS['HO_list_domain'+str(int(i+1))+'b'],\
    VARS['USE_COORS'],VARS['COORS'],\
    VARS['domain_class_'+str(int(i+1))],\
    VARS['domain'+str(int(i+1))+'A'],\
    VARS['domain'+str(int(i+1))+'B'],\
    VARS['MIRROR'],VARS['SORBATE_LIST'],\
    VARS['SORBATE_coors_a'],VARS['O_coors_a'],\
    VARS['BASAL_EL']
    if j%2==0 and LOCAL_STRUCTURE[i][int(j/2)]=='trigonal_pyramid':
        rgh.new_var('top_angle_TD_'+str(j), 70.)
    elif j%2==0 and LOCAL_STRUCTURE[i][int(j/2)]=='octahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            rgh.new_var('dr1_oct_TD_'+str(j), 0.)
            rgh.new_var('dr2_oct_TD_'+str(j), 0.)
            rgh.new_var('dr3_oct_TD_'+str(j), 0.)
            [rgh.new_var('r1_'+str(KK+1)+'_TD_'+str(j), 2.27) for KK in range(3)]
            [rgh.new_var('theta1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(3)]
            [rgh.new_var('phi1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(3)]
        else:
            rgh.new_var('dr1_oct_TD_'+str(j), 0.)
            rgh.new_var('dr2_oct_TD_'+str(j), 0.)
            rgh.new_var('dr3_oct_TD_'+str(j), 0.)
    elif j%2==0 and LOCAL_STRUCTURE[i][int(j/2)]=='tetrahedral':
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            [rgh.new_var('r1_'+str(KK+1)+'_TD_'+str(j), 2.27) for KK in range(1)]
            [rgh.new_var('theta1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(1)]
            [rgh.new_var('phi1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(1)]
            rgh.new_var('dr_tetrahedral_TD_'+str(j), 0.)
        else:
            rgh.new_var('dr_tetrahedral_TD_'+str(j), 0.)
            rgh.new_var('dr_bc_tetrahedral_TD_'+str(j), 0.)

    ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
    if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
        ids=[SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][2]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A']
    else:
        ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][2]+'_D'+str(int(i+1))+'A']
    SORBATE_id=sorbate_list_a[j]
    O_index,O_id,sorbate_coors,O_id_B,HO_set_ids,SORBATE_id_B,sorbate_ids,SORBATE_coors_a=[],[],[],[],[],[],[],[]
    if LOCAL_STRUCTURE[i][int(j/2)]=='octahedral':
        O_id=[HO_id for HO_id in HO_list_a if SORBATE_id in HO_id]
        if USE_COORS[i][j]:
            sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
        else:
            sorbate_coors=domain_class.adding_share_triple_octahedra(domain=domainA,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id)
        SORBATE_coors_a.append(sorbate_coors[0])
        [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
        SORBATE_id_B=sorbate_list_b[j]
        O_id_B=[HO_id for HO_id in HO_list_b if SORBATE_id_B in HO_id]
        sorbate_ids=[SORBATE_id_B]+O_id_B
        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
        if USE_COORS[i][j]:
            SORBATE_id_A=sorbate_list_a[j]
            O_id_A=[HO_id for HO_id in HO_list_a if SORBATE_id_A in HO_id]
            sorbate_ids_A=[SORBATE_id_A]+O_id_A
            domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
        domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='trigonal_pyramid':
        if USE_COORS[i][j]:
            sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
        else:
            sorbate_coors=domain_class.adding_pb_share_triple4(domain=domainA,top_angle=70,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j])
        SORBATE_coors_a.append(sorbate_coors[0])
        SORBATE_id_B=sorbate_list_b[j]
        #now put on sorbate on the symmetrically related domain
        sorbate_ids=[SORBATE_id_B]
        sorbate_els=[SORBATE_LIST[i][j]]
        if USE_COORS[i][j]:
            SORBATE_id_A=sorbate_list_a[j]
            sorbate_ids_A=[SORBATE_id_A]
            domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a),ids=sorbate_ids_A,els=sorbate_els)
        domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='tetrahedral':
        O_id=[HO_id for HO_id in HO_list_a if SORBATE_id in HO_id]
        if USE_COORS[i][j]:
            sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
        else:
            sorbate_coors=domain_class.adding_sorbate_tridentate_tetrahedral(domain=domainA,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id)
        SORBATE_coors_a.append(sorbate_coors[0])
        [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
        SORBATE_id_B=sorbate_list_b[j]
        O_id_B=[HO_id for HO_id in HO_list_b if SORBATE_id_B in HO_id]
        sorbate_ids=[SORBATE_id_B]+O_id_B
        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
        if USE_COORS[i][j]:
            SORBATE_id_A=sorbate_list_a[j]
            O_id_A=[HO_id for HO_id in HO_list_a if SORBATE_id_A in HO_id]
            sorbate_ids_A=[SORBATE_id_A]+O_id_A
            domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
        domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)

    #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
    #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
    #if SORBATE_LIST[i][j]=='Sb':
    sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
    HO_set_ids=O_id+O_id_B
    N=int(len(sorbate_set_ids)/2)
    M=len(O_id)
    gp_sorbates_set=domain_class.grouping_discrete_layer3(domain=[domainA]*N+[domainB]*N,atom_ids=sorbate_set_ids)
    if M!=0:
        gp_HO_set=domain_class.grouping_discrete_layer3(domain=[domainA]*M+[domainB]*M,atom_ids=HO_set_ids)
    else:
        gp_HO_set=[]
    return gp_sorbates_set,gp_HO_set

def update_sorbate_in_SIM_TD(VARS,domain_class_1,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,LOCAL_STRUCTURE,ADD_DISTAL_LIGAND_WILD,SORBATE_LIST,MIRROR):
    SORBATE_coors_a=[]
    O_coors_a=[]
    if LOCAL_STRUCTURE[i][int(j/2)]=='trigonal_pyramid':
        top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_TD_'+str(int(j/2)*2))
        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
        if 'HO' in ids[0]:
            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A']
        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_pb_share_triple4(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j])
        SORBATE_coors_a.append(sorbate_coors[0])
        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
        #now put on sorbate on the symmetrically related domain
        sorbate_ids=[SORBATE_id_B]
        sorbate_els=[SORBATE_LIST[i][j]]
        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='octahedral':
        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
        if 'HO' in ids[0]:
            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A']
        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
        dr=[getattr(VARS['rgh_domain'+str(int(i+1))],'dr1_oct_TD_'+str(int(j/2)*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'dr2_oct_TD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'dr3_oct_TD_'+str(j/2*2))]
        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
        O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_share_triple_octahedra(domain=VARS['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id,dr=dr,mirror=MIRROR[i][int(j/2)])
        SORBATE_coors_a.append(sorbate_coors[0])
        #sorbate_offset=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)-domain_creator.extract_coor2(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
        O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
        #now put on sorbate on the symmetrically related domain
        sorbate_ids=[SORBATE_id_B]+O_id_B
        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='tetrahedral':
        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
        if 'HO' in ids[0]:
            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][int(j/2)]+str(j-1)+'_D'+str(int(i+1))+'A']
        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
        edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'dr_tetrahedral_TD_'+str(int(j/2)*2))
        angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'dr_bc_tetrahedral_TD_'+str(int(j/2)*2))
        O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_tridentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id,edge_offset=edge_offset,top_angle_offset=angle_offset)
        SORBATE_coors_a.append(sorbate_coors[0])
        #sorbate_offset=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)-domain_creator.extract_coor2(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)
        if ADD_DISTAL_LIGAND_WILD[i][j]:
            if (i+j)%2==1:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)))) for ligand_id in range(len(O_id))]
            else:
                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(int(j/2)*2)))) for ligand_id in range(len(O_id))]
        else:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
        O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
        #now put on sorbate on the symmetrically related domain
        sorbate_ids=[SORBATE_id_B]+O_id_B
        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
        return None

def setup_sorbate_OS(VARS,i,j):

    rgh,\
    LOCAL_STRUCTURE,\
    ADD_DISTAL_LIGAND_WILD,\
    SORBATE_ATTACH_ATOM,\
    SORBATE_ATTACH_ATOM_OFFSET,\
    sorbate_list_a,\
    sorbate_list_b, \
    HO_list_a,\
    HO_list_b,\
    USE_COORS,\
    COORS,\
    domain_class,\
    domainA,\
    domainB,\
    MIRROR,\
    SORBATE_LIST,\
    SORBATE_coors_a,\
    O_coors_a\
    =\
    VARS['rgh_domain'+str(int(i+1))],\
    VARS['LOCAL_STRUCTURE'],VARS['ADD_DISTAL_LIGAND_WILD'],\
    VARS['SORBATE_ATTACH_ATOM'],VARS['SORBATE_ATTACH_ATOM_OFFSET'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'a'],\
    VARS['SORBATE_list_domain'+str(int(i+1))+'b'],\
    VARS['HO_list_domain'+str(int(i+1))+'a'],\
    VARS['HO_list_domain'+str(int(i+1))+'b'],\
    VARS['USE_COORS'],VARS['COORS'],\
    VARS['domain_class_'+str(int(i+1))],\
    VARS['domain'+str(int(i+1))+'A'],\
    VARS['domain'+str(int(i+1))+'B'],\
    VARS['MIRROR'],VARS['SORBATE_LIST'],\
    VARS['SORBATE_coors_a'],VARS['O_coors_a']


    if j%2==0:
        rgh.new_var('phi_OS_'+str(j), 0.)
        rgh.new_var('r0_OS_'+str(j), 2.26)
        rgh.new_var('top_angle_OS_'+str(j), 70.)
        rgh.new_var('ct_offset_dx_OS_'+str(j), 0.)
        rgh.new_var('ct_offset_dy_OS_'+str(j), 0.)
        rgh.new_var('ct_offset_dz_OS_'+str(j), 0.)
        rgh.new_var('rot_x_OS_'+str(j), 0.)
        rgh.new_var('rot_y_OS_'+str(j), 0.)
        rgh.new_var('rot_z_OS_'+str(j), 0.)

    SORBATE_id=sorbate_list_a[j]#pb_id is a str NOT list
    O_id=[HO_id for HO_id in HO_list_a if SORBATE_id in HO_id]
    consider_distal=False
    if O_id!=[]:
        consider_distal=True
    sorbate_coors=[]
    if USE_COORS[i][j]:
        sorbate_coors=COORS[(i,j)]['sorbate'][0]+COORS[(i,j)]['oxygen'][0]
    else:
        if LOCAL_STRUCTURE[i][int(j/2)]=='trigonal_pyramid':
            sorbate_coors=domain_class.outer_sphere_complex_2(domain=domainA,cent_point=[0.75,0.+j*0.5,2.1],r_Pb_O=2.28,O_Pb_O_ang=70,phi=j*np.pi-0,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
        elif LOCAL_STRUCTURE[i][int(j/2)]=='octahedral':
            sorbate_coors=domain_class.outer_sphere_complex_oct(domain=domainA,cent_point=[0.75,0.+j*0.5,2.1],r0=1.62,phi=j*np.pi-0,Sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
        elif LOCAL_STRUCTURE[i][int(j/2)]=='tetrahedral':
            sorbate_coors=domain_class.outer_sphere_tetrahedral2(domain=domainA,cent_point=[0.75,0.+j*0.5,2.1],r_sorbate_O=1.65,phi=j*np.pi-0,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
    SORBATE_coors_a.append(sorbate_coors[0])
    if O_id!=[]:
        [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_id_B=sorbate_list_b[j]
    O_id_B=[HO_id for HO_id in HO_list_b if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    if USE_COORS[i][j]:
        SORBATE_id_A=sorbate_list_a[j]
        O_id_A=[HO_id for HO_id in HO_list_a if SORBATE_id_A in HO_id]
        sorbate_ids_A=[SORBATE_id_A]+O_id_A
        domain_creator.add_atom(domain=domainA,ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
    domain_creator.add_atom(domain=domainB,ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
    #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
    sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
    HO_set_ids=O_id+O_id_B
    N=int(len(sorbate_set_ids)/2)
    M=len(O_id)
    gp_sorbates_set=domain_class.grouping_discrete_layer3(domain=[domainA]*N+[domainB]*N,atom_ids=sorbate_set_ids)
    #if O_NUMBER[i][j]!=0:
    if M!=0:
        gp_HO_set=domain_class.grouping_discrete_layer3(domain=[domainA]*M+[domainB]*M,atom_ids=HO_set_ids)
    return gp_sorbates_set,gp_HO_set

def setup_water_layer(rgh,Os_list_a,Os_list_b,domain_class,domainA,domainB,i,layered_water_pars,WATER_NUMBER,WATER_PAIR,REF_POINTS,layered_sorbate_pars):
    gp_names=[]
    gp_container=[]
    if layered_water_pars['yes_OR_no'][i]:
        rgh.new_var('u0',0.4)
        rgh.new_var('ubar',0.4)
        rgh.new_var('first_layer_height',4.0)#relative height in A
        rgh.new_var('d_w',1.9)#inter-layer water seperation in A
        rgh.new_var('density_w',0.033)#number density in unit of # of waters per cubic A

    if layered_sorbate_pars['yes_OR_no'][i]:
        rgh.new_var('u0_s',0.4)
        rgh.new_var('ubar_s',0.4)
        rgh.new_var('first_layer_height_s',4.0)#relative height in A
        rgh.new_var('d_s',1.9)#inter-layer sorbate seperation in A
        rgh.new_var('density_s',0.033)#number density in unit of # of sorbates per cubic A
        rgh.new_var('oc_damping_factor',0.0)#number density in unit of # of sorbates per cubic A

    if WATER_NUMBER[i]!=0:#add water molecules if any
        if WATER_PAIR:
            for jj in range(int(WATER_NUMBER[i]/2)):#note will add water pair (two oxygens) each time, and you can't add single water
                rgh.new_var('alpha_W_'+str(jj+1),90.)
                #vars()['rgh_domain'+str(int(i+1))].new_var('R_W_'+str(jj+1),1)
                rgh.new_var('v_shift_W_'+str(jj+1),1.)

                O_ids_a=Os_list_a[jj*2:jj*2+2]
                O_ids_b=Os_list_b[jj*2:jj*2+2]
                #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                H2O_coors_a=domain_class.add_oxygen_pair2B(domain=domainA,O_ids=O_ids_a,ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=1,r=2.717,alpha=90)
                domain_creator.add_atom(domain=domainB,ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
                #group water molecules at each layer (set equivalent the oc and u during fitting)
                M=len(O_ids_a)
                #group waters on a layer basis(every four, two from each domain)
                vars()['gp_waters_set'+str(jj+1)+'_D'+str(int(i+1))]=domain_class.grouping_discrete_layer3(domain=[domainA]*M+[domainB]*M,atom_ids=O_ids_a+O_ids_b,sym_array=[[1, 0, 0, 0, 1, 0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]])
                gp_names.append('gp_waters_set'+str(jj+1)+'_D'+str(int(i+1)))
                gp_container.append(vars()['gp_waters_set'+str(jj+1)+'_D'+str(int(i+1))])
                #group each two waters on two symmetry domains together (to be used as constrain on inplane movements)
                #group names look like: gp_Os1_D1 which will group Os1_D1A and Os1_D1B together
                for O_id in O_ids_a:
                    index=O_ids_a.index(O_id)
                    gp_name='gp_'+O_id.rsplit('_')[0]+'_D'+str(int(i+1))
                    gp_names.append(gp_name)
                    gp_container.append(domain_class.grouping_discrete_layer3(domain=[domainA]+[domainB],atom_ids=[O_ids_a[index],O_ids_b[index]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]]))
        else:
            for jj in range(WATER_NUMBER[i]):#note will add single water each time
                rgh.new_var('v_shift_W_'+str(jj+1),1)

                O_ids_a=[Os_list_a[jj]]
                O_ids_b=[Os_list_b[jj]]
                #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                H2O_coors_a=domain_class.add_single_oxygen_2(domain=domainA,O_id=O_ids_a[0],ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=1)
                domain_creator.add_atom(domain=domainB,ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O'])
                #group each two waters on two symmetry domains together (to be used as constrain on inplane movements)
                #group names look like: gp_Os1_D1 which will group Os1_D1A and Os1_D1B together
                gp_names.append('gp_'+O_ids_a[0].rsplit('_')[0]+'_D'+str(int(i+1)))
                gp_container.append(domain_class.grouping_discrete_layer3(domain=[domainA]+[domainB],atom_ids=[O_ids_a[0],O_ids_b[0]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]]))
    return gp_names,gp_container

def update_sorbate_in_SIM_OS(VARS,i,j,SORBATE_ATTACH_ATOM_OFFSET,BASAL_EL,OS_X_REF,OS_Y_REF,OS_Z_REF,LOCAL_STRUCTURE,SORBATE_LIST):
    SORBATE_coors_a=[]
    O_coors_a=[]
    phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_OS_'+str(int(j/2)*2))
    r_Pb_O=getattr(VARS['rgh_domain'+str(int(i+1))],'r0_OS_'+str(int(j/2)*2))
    O_Pb_O_ang=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_OS_'+str(int(j/2)*2))
    ct_offset_dx=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dx_OS_'+str(int(j/2)*2))
    ct_offset_dy=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dy_OS_'+str(int(j/2)*2))
    ct_offset_dz=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dz_OS_'+str(int(j/2)*2))
    rot_x,rot_y,rot_z=getattr(VARS['rgh_domain'+str(int(i+1))],'rot_x_OS_'+str(int(j/2)*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'rot_y_OS_'+str(int(j/2)*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'rot_z_OS_'+str(int(j/2)*2))
    ref_x,ref_y,ref_z=OS_X_REF[i][j],OS_Y_REF[i][j],OS_Z_REF[i][j]
    if (j+i)%2==1:
        phi=180-phi#note all angles in degree
        ct_offset_dx=-getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dx_OS_'+str(int(j/2)*2))
        rot_y,rot_z=-rot_y,-rot_z
    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
    consider_distal=False
    if O_id!=[]:
        consider_distal=True
    sorbate_coors=[]
    if LOCAL_STRUCTURE[i][int(j/2)]=='trigonal_pyramid':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_complex_2(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r_Pb_O=r_Pb_O,O_Pb_O_ang=O_Pb_O_ang,phi=phi,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='octahedral':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_complex_oct(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r0=r_Pb_O,phi=phi,Sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
    elif LOCAL_STRUCTURE[i][int(j/2)]=='tetrahedral':
        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_tetrahedral2(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r_sorbate_O=r_Pb_O,phi=phi,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal,rotation_x=rot_x,rotation_y=rot_y,rotation_z=rot_z)

    SORBATE_coors_a.append(sorbate_coors[0])
    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
    #now put on sorbate on the symmetrically related domain
    sorbate_ids=[SORBATE_id_B]+O_id_B
    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    return None

def calculate_BV_sum_in_SIM(i,VARS,domain_class_1,BV_TOLERANCE,DOMAIN,SORBATE_NUMBER,SORBATE_LIST,WATER_NUMBER,O_NUMBER,BOND_VALENCE_WAIVER,CONSIDER_WATER_IN_BV,debug_bv,SORBATE_EL_LIST,SEARCH_MODE_FOR_SURFACE_ATOMS,EARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,SEARCH_RANGE_OFFSET,running_mode,R0_BV,COUNT_DISTAL_OXYGEN,SEARCHING_PARS,METAL_BV,PRINT_BV,COVALENT_HYDROGEN_RANDOM,COVALENT_HYDROGEN_ACCEPTOR,POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR,PRINT_PROTONATION,POTENTIAL_HYDROGEN_ACCEPTOR,COVALENT_HYDROGEN_NUMBER):
    bv_container={}
    bv=0
    #set up dynamic super cells,where water and sorbate is a library and surface is a domain instance
    def _widen_validness(value):#acceptable bond valence offset can be adjusted (here is 0.08)
        if value<BV_TOLERANCE[0]:return 100
        elif value>=BV_TOLERANCE[0] and value<BV_TOLERANCE[1]:return 0
        else:return value
    def _widen_validness_range(value_min,value_max):#consider a range of (ideal_bv-temp_bv)
        if (value_min<BV_TOLERANCE[0] and value_max>BV_TOLERANCE[1]) or (value_min>=BV_TOLERANCE[0] and value_min<=BV_TOLERANCE[1]) or (value_max>=BV_TOLERANCE[0] and value_max<=BV_TOLERANCE[1]):
            return 0
        elif value_min>BV_TOLERANCE[1]:return value_min
        else:return 100
    def _widen_validness_hydrogen_acceptor(value,H_N=0):#here consider possible contribution of hydrogen bond (~0.2)
        if (value-H_N*0.2)<BV_TOLERANCE[0]:return 100
        elif (value-H_N*0.2)>=BV_TOLERANCE[0] and (value-H_N*0.2)<BV_TOLERANCE[1]:return 0
        else:return (value-H_N*0.2)
    def _widen_validness_potential_hydrogen_acceptor(value):#value=2-temp_bv(temp_bv include covalent hydrogen bond possibly)
        if value<0.2 and value>BV_TOLERANCE[0]: return 0
        elif value<BV_TOLERANCE[0]: return 100
        else:return value

    super_cell_sorbate,super_cell_surface,super_cell_water=None,None,None
    if WATER_NUMBER[i]!=0:
        if DOMAIN[i]==1:
            super_cell_water=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+[4,5]+range(-(sum(SORBATE_NUMBER[i])+WATER_NUMBER[i]+sum([np.sum(N_list) for N_list in O_NUMBER[i]])),0))
        elif DOMAIN[i]==2:
            super_cell_water=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1,2,3]+range(-(sum(SORBATE_NUMBER[i])+WATER_NUMBER[i]+sum([np.sum(N_list) for N_list in O_NUMBER[i]])),0))

    def _return_right_value(value):
        if value:return value
        else:return 1
    NN=_return_right_value(sum(SORBATE_NUMBER[i]))#number of sorbate sets(1 or 2)
    N_HB_SURFACE,N_HB_DISTAL=0,0
    O_N_for_this_domain=O_NUMBER[i]
    total_sorbate_number=sum(SORBATE_NUMBER[i])+sum(O_N_for_this_domain)
    #the idea is that we want to have only one set of sorbate and hydrogen within each domain (ie don't count symmetry counterpart twice)
    def _cal_segment2(O_N_list=[],water_N=0):
        cum_list=[-water_N]
        segment2_boundary=[]
        segment2=[]
        for O_N in O_N_list[::-1]:
            cum_list.append(cum_list[-1]-(O_N+1))
        for i in range(0,len(cum_list)-1,2):
            segment2_boundary.append([cum_list[i],cum_list[i+1]])
        for each in segment2_boundary:
            segment2=segment2+range(each[1],each[0])
        return segment2

    segment1=range(-WATER_NUMBER[i],0)
    segment2=_cal_segment2(O_N_for_this_domain,WATER_NUMBER[i])
    segment3=range(-(WATER_NUMBER[i]+total_sorbate_number),-(WATER_NUMBER[i]+total_sorbate_number))

    if DOMAIN[i]==1:
        #note here if there are two symmetry pair, then only consider one of the couple for bv consideration, the other one will be skipped in the try except statement
        super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(4,8)+segment1+segment2+segment3)
        if SEARCH_MODE_FOR_SURFACE_ATOMS:
            super_cell_surface=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(4,30)+segment1+segment2+segment3)
        else:
            super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()
            #delete the first iron layer atoms if considering a half layer
            super_cell_surface.del_atom(super_cell_surface.id[2])
            super_cell_surface.del_atom(super_cell_surface.id[2])
    elif DOMAIN[i]==2:
        super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],range(0,6)+segment1+segment2+segment3)
        if SEARCH_MODE_FOR_SURFACE_ATOMS:
            super_cell_surface=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],range(0,30)+segment1+segment2+segment3)
        else:
            super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()

    #consider hydrogen bond only among interfacial water molecules and top surface Oxygen layer and Oxygen ligand
    if WATER_NUMBER[i]!=0:
        water_ids=VARS['Os_list_domain'+str(int(i+1))+'a']
        for id in water_ids:
            tmp_bv=domain_class_1.cal_hydrogen_bond_valence2B(super_cell_water,id,3.,2.5,BOND_VALENCE_WAIVER)*int(CONSIDER_WATER_IN_BV)
            bv=bv+tmp_bv
            if debug_bv:bv_container[id]=tmp_bv
    #cal bv for surface atoms and sorbates
    #only consdier domainA since domain B is symmetry related to domainA
    waiver_box=[]#the first set of anchored oxygens will be waived for being considered for bond valence constraints
    attach_atom_ids=VARS['SORBATE_ATTACH_ATOM'][i]
    if len(attach_atom_ids)==0:
        pass
    elif len(attach_atom_ids)!=0 and len(attach_atom_ids)%2==0:
        if len(attach_atom_ids[0])<3:#only for monodentate and bidentate binding mode
            waiver_box=map(lambda x:x+'_D'+str(i+1)+'A',attach_atom_ids[0])
        else:
            pass
    for key in [each_key for each_key in VARS['match_lib_'+str(i+1)+'A'].keys() if each_key not in waiver_box]:
        temp_bv=None
        if ([sorbate not in key for sorbate in SORBATE_EL_LIST]==[True]*len(SORBATE_EL_LIST)) and ("HO" not in key) and ("Os" not in key):#surface atoms
            if SEARCH_MODE_FOR_SURFACE_ATOMS:#cal temp_bv based on searching within spherical region
                el=None
                if "Fe" in key: el="Fe"
                elif "O" in key and "HB" not in key: el="O"
                elif "HB" in key: el="H"
                if el=="H":
                    temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_surface,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
                else:
                    temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_surface,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
            else:
                #no searching in this algorithem
                temp_bv=domain_class_1.cal_bond_valence4B(super_cell_surface,key,VARS['match_lib_'+str(i+1)+'A'][key],2.5)
        else:#sorbates including water
            #searching included in this algorithem
            if "HO" in key and COUNT_DISTAL_OXYGEN:#distal oxygen and its associated hydrogen
                el="O"
                if "HB" in key:
                    el="H"
                if el=="O":
                    try:
                        temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                    except:
                        temp_bv=2
                else:
                    try:
                        temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_sorbate,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
                    except:
                        temp_bv=1
            elif "HO" in key and not COUNT_DISTAL_OXYGEN:
                temp_bv=2
            elif "Os" in key:#water and the associated hydrogen
                el="O"
                if "HB" in key:
                    el="H"
                if el=="O":
                    temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                else:
                    temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_sorbate,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
            else:#metals
                try:
                    temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,SORBATE_LIST[i][0],SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],SEARCHING_PARS['sorbate'][1],False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                except:
                    temp_bv=METAL_BV[i][int(key.rsplit('_')[0][-1])-1][0]

        if PRINT_BV:print(key, temp_bv)
        #consider possible hydrogen bond and hydroxyl bond fro oxygen atoms
        if 'O' in key:
            #For O you may consider possible binding to proton (+0.8)
            #And note the maximum coordination number for O is 4
            case_tag=len(VARS['match_lib_'+str(i+1)+'A'][key])#current coordination number
            if COVALENT_HYDROGEN_RANDOM==True:
                if case_tag<4 and key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR[i]):
                    C_H_N=range(4-case_tag+1)#max CN allowed is 4,if case_tag=3, then range(4-case_tag+1)=[0,1]
                    bv_offset=[ _widen_validness_range(2-0.88*N-temp_bv,2-0.68*N-temp_bv) for N in C_H_N]
                    C_H_N=C_H_N[bv_offset.index(min(bv_offset))]
                    case_tag=case_tag+C_H_N#CN after considering the proton
                    if PRINT_PROTONATION:
                        print(key,C_H_N)
                    if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider potential hydrogen bond (you can have or have not H-bonding)
                        if _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==0 or _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==100:
                        #if saturated already or over-saturated, then adding H-bonding wont help decrease the the total bv anyhow
                        #or reach the maximum CN(4), the adding one hydrogen bond is not allowed
                            bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                        else:#you can add one hydrogen bond at most
                        #if undersaturation, then compare the cases of inclusion of H-bonding and exclusion of H-bonding. Whichever give rise to the lower bv will be used.
                            bv=bv+min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                            if debug_bv:bv_container[key]=min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                    else:#consider covalent hydrogen bond only
                        bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                        if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                elif case_tag==4:#coordination saturation achieved, so neighter covalent hydrogen bond nor hydrogen bond
                    bv=bv+_widen_validness(2-temp_bv)
                    if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
            else:
                if key in map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]):
                    #if consider convalent hydrogen bond (bv=0.68 to 0.88) while the hydrogen bond has bv from 0.13 to 0.25
                    C_H_N=COVALENT_HYDROGEN_NUMBER[i][map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]).index(key)]
                    case_tag=case_tag+C_H_N
                    if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider potential hydrogen bond (you can have or have not H-bonding)
                        if _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==0 or _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==100 or case_tag>=4:
                        #if saturated already or over-saturated, then adding H-bonding wont help decrease the the total bv anyhow
                            bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                        else:
                        #if undersaturation, then compare the cases of inclusion of H-bonding and exclusion of H-bonding. Whichever give rise to the lower bv will be used.
                            bv=bv+min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                            if debug_bv:bv_container[key]=min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                    else:
                        bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                        if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                else:
                    if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider hydrogen bond
                        if _widen_validness(2-temp_bv)==0 or _widen_validness(2-temp_bv)==100 or case_tag>=4:
                            bv=bv+_widen_validness(2-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                        else:
                            bv=bv+min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                            if debug_bv:bv_container[key]=min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                    else:
                        bv=bv+_widen_validness(2-temp_bv)
                        if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
        elif 'Fe' in key:
            bv=bv+_widen_validness(3-temp_bv)
            if debug_bv:bv_container[key]=_widen_validness(3-temp_bv)
        else:#do metal sorbates
            metal_bv_range=[]
            metal_bv_range=METAL_BV[i][int(key.rsplit('_')[0][-1])-1]
            bv=bv+_widen_validness_range(metal_bv_range[0]-temp_bv,metal_bv_range[1]-temp_bv)
            if debug_bv:bv_container[key]=_widen_validness_range(metal_bv_range[0]-temp_bv,metal_bv_range[1]-temp_bv)
    return bv, bv_container

def extract_layer_water_info(VARS,i,layered_water_pars):
    u0=getattr(VARS['rgh_domain'+str(int(i+1))],'u0')
    ubar=getattr(VARS['rgh_domain'+str(int(i+1))],'ubar')
    d_w=getattr(VARS['rgh_domain'+str(int(i+1))],'d_w')
    first_layer_height=getattr(VARS['rgh_domain'+str(int(i+1))],'first_layer_height')
    density_w=getattr(VARS['rgh_domain'+str(int(i+1))],'density_w')
    ref_atom=layered_water_pars['ref_layer_height'][i]+'_D'+str(i+1)+'A'
    ref_height=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],ref_atom)[2]*7.3707
    layered_water_A=[u0,ubar,d_w,first_layer_height+ref_height,density_w]#7.3707 is specifically for hematite rcut
    layered_water_B=[u0,ubar,d_w,first_layer_height+ref_height-3.68535,density_w]#symmetry related domain has height offset of 0.5*7.3707
    return layered_water_A,layered_water_B

def extract_layer_sorbate_info(VARS,i,layered_sorbate_pars,F1F2):
    u0_s=getattr(VARS['rgh_domain'+str(int(i+1))],'u0_s')
    ubar_s=getattr(VARS['rgh_domain'+str(int(i+1))],'ubar_s')
    d_s=getattr(VARS['rgh_domain'+str(int(i+1))],'d_s')
    first_layer_height_s=getattr(VARS['rgh_domain'+str(int(i+1))],'first_layer_height_s')
    density_s=getattr(VARS['rgh_domain'+str(int(i+1))],'density_s')
    oc_damping_factor=getattr(VARS['rgh_domain'+str(int(i+1))],'oc_damping_factor')
    ref_atom_s=layered_sorbate_pars['ref_layer_height'][i]+'_D'+str(i+1)+'A'
    ref_height_s=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],ref_atom_s)[2]*7.3707
    layered_sorbate_A=[u0_s,ubar_s,d_s,first_layer_height_s+ref_height_s,density_s,oc_damping_factor,F1F2]#7.3707 is specifically for hematite rcut
    layered_sorbate_B=[u0_s,ubar_s,d_s,first_layer_height_s+ref_height_s-3.68535,density_s,oc_damping_factor,F1F2]#symmetry related domain has height offset of 0.5
    return layered_sorbate_A,layered_sorbate_B

def output_model_files(i,COVALENT_HYDROGEN_NUMBER,PROTONATION_DISTAL_OXYGEN,SORBATE_NUMBER,O_NUMBER,WATER_NUMBER,VARS,output_file_path,xyz,water_pars,half_layer,full_layer,DOMAIN,half_layer_pick,full_layer_pick):
    N_HB_SURFACE=sum(COVALENT_HYDROGEN_NUMBER[i])
    N_HB_DISTAL=sum(PROTONATION_DISTAL_OXYGEN[i])
    total_sorbate_number=sum(SORBATE_NUMBER[i])+sum([np.sum(N_list) for N_list in O_NUMBER[i]])
    N_sorbate,N_distal_old=np.array(SORBATE_NUMBER[i]),np.array([np.sum(N_list) for N_list in O_NUMBER[i]])
    N_distal=[np.sum(N_distal_old[j*2:j*2+2]) for j in range(int(len(N_distal_old)/2))]
    N_sorbate_and_distal=N_sorbate+N_distal
    first_item_index=[0]+[np.sum(N_sorbate_and_distal[0:j+1]) for j in range(len(N_sorbate_and_distal)-1)]
    length_of_each_segment=list(N_sorbate_and_distal/2)
    first_item_index.append(np.sum(N_sorbate_and_distal))
    length_of_each_segment.append(WATER_NUMBER[i])
    water_number=WATER_NUMBER[i]*3
    TOTAL_NUMBER=total_sorbate_number+int(water_number/3)
    domain_creator.print_data2(N_sorbate=TOTAL_NUMBER,domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'_dsv.xyz'))
    domain_creator.print_data2C(domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'.xyz'),sorbate_index_list=first_item_index,each_segment_length=length_of_each_segment)
    domain_creator.make_cif_file(domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'.cif'),sorbate_index_list=first_item_index,each_segment_length=length_of_each_segment)
    test=xyz.formate_vtk(os.path.join(output_file_path,'Model_domain'+str(i+1)+'.xyz'))
    test.all_in_all()
    #output for publication
    domain_creator.print_data_for_publication_B2(N_sorbate=np.sum(SORBATE_NUMBER[i])+np.sum(O_NUMBER[i])+WATER_NUMBER[i],domain=VARS['domain'+str(int(i+1))+'A'],z_shift=1,layer_types=(half_layer+full_layer)[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'A_publication.dat'))
    #make sure you have the test.tab file in the specified folder
    domain_creator.make_publication_table2(model_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'A_publication.dat'),par_file=os.path.join(output_file_path,"test.tab"),el_substrate=[each for each in list(set(VARS['domain'+str(i+1)+'A'].el)) if each not in VARS['SORBATE_EL_LIST']],el_sorbate=VARS['SORBATE_EL_LIST'],abc=[VARS['unitcell'].a,VARS['unitcell'].b,VARS['unitcell'].c])

    return None

def create_dummy_raxr_data(beta,model,inst, bulk, domain, unitcell,COHERENCE,SURFACE_PARMS,batch_path_head,output_file_path):
    #need more work
    LB=2
    dL=2
    h,k,l=np.zeros(28),np.zeros(28),np.arange(0.35,9.9,0.35)
    rough_temp = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l-LB)/dL)**2)**0.5
    f1f2_data_calculated=None
    f1f2_data_calculated=np.loadtxt(os.path.join(batch_path_head,'Lead_CL_output.f1f2'))
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms=SURFACE_PARMS)
    aa=rough_temp*sample.calc_f4_specular_RAXR_for_test_purpose(h,k,l,f1f2_data_calculated[:,(1,2)],res_el='Pb')
    raxr_data=np.zeros((1,8))[0:0]
    for i in range(28):#28 hkl values
        for j in range(len(f1f2_data_calculated)):
            temp_data=[[f1f2_data_calculated[j,0],0,0,l[i],aa[j,i],aa[j,i]*0.005,LB,dL]]
            raxr_data=np.append(raxr_data,temp_data,axis=0)
    np.savetxt(os.path.join(output_file_path,'dummy_raxr_dataset_Pb_case.dat'),raxr_data)
    return None

def create_dummy_raxr_data_hematite(sample,data,VARS,RESONANT_EL_LIST,RAXR_EL,beta,E0,F1F2,SCALES,output_file_path):
    H,K,L,E=[],[],[],[]
    F=[]
    i=0
    for data_set in data:
        if data_set.x[0]>100:
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            x = data_set.x
            y = data_set.extra_data['Y']
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            H.append(h[0])
            K.append(k[0])
            L.append(y[0])
            if i==0:
                E=x
            if data_set.x[0]>100:#doing RAXR calculation(x is energy column typically in magnitude of 10000 ev)
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                f=sample.cal_structure_factor_hematite_RAXR(i,VARS,'MD',RESONANT_EL_LIST,RAXR_EL,h, k, y, x, E0, F1F2,SCALES,rough)
                F.append(abs(f))
                i+=1
        else:#doing CTR calculation (x is perpendicular momentum transfer L typically smaller than 15)
            pass
    #need more work
    LB=2
    dL=2
    h,k,l=H,K,L
    raxr_data=np.zeros((1,8))[0:0]
    for i in range(len(h)):#28 hkl values
        for j in range(len(E)):
            temp_data=[[E[j],h[i],k[i],l[i],F[i][j],F[i][j]*0.005,LB,dL]]
            raxr_data=np.append(raxr_data,temp_data,axis=0)
    np.savetxt(os.path.join(output_file_path,'dummy_raxr_dataset_Pb_case.dat'),raxr_data)
    return None

def plot_ctr_raxr_e_profiles(model,inst, bulk, domain, unitcell,COHERENCE,data,beta,VARS,E0,F1F2,RAXR_FIT_MODE,RESONANT_EL_LIST,SCALES,output_file_path,RAXR_EL,exp_const,rgh,re,auc,z_min=0,z_max=29,SURFACE_PARMS={'delta1':0.,'delta2':0.1391},ref_height=21.397):
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms=SURFACE_PARMS)
    bl_dl={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
        '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
        '0_0':{'segment':[[0,13]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
        '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
        '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
    domain_number=len(sample.domain.keys())
    plot_data_container_experiment={}
    plot_data_container_model={}
    plot_raxr_container_experiment={}
    plot_raxr_container_model={}
    A_list_Fourier_synthesis=[]
    P_list_Fourier_synthesis=[]
    HKL_list_raxr=[[],[],[]]

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

    spectra_index=0
    for data_set in data:
        if data_set.x[0]<15:
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            #make dumy hkl and f to make the plot look smoother
            #l_dumy=np.arange(l[0],l[-1]+0.1,0.1)
            left,right=np.arange(0,l[0]-0.2,-0.145)[1:],np.arange(0,l[-1]+0.2,0.145)[1:]
            left.sort()
            l_dumy=np.append(left,right)
            N=len(l_dumy)
            h_dumy=np.array([h[0]]*N)
            k_dumy=np.array([k[0]]*N)
            LB_dumy=[]
            dL_dumy=[]
            f_dumy=[]

            for i in range(N):
                key=None
                if l_dumy[i]>=0:
                    key=str(int(h[0]))+'_'+str(int(k[0]))
                else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                for ii in bl_dl[key]['segment']:
                    if abs(l_dumy[i])>=ii[0] and abs(l_dumy[i])<ii[1]:
                        n=bl_dl[key]['segment'].index(ii)
                        LB_dumy.append(bl_dl[key]['info'][n][1])
                        dL_dumy.append(bl_dl[key]['info'][n][0])
            LB_dumy=np.array(LB_dumy)
            dL_dumy=np.array(dL_dumy)
            rough_dumy = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l_dumy-LB_dumy)/dL_dumy)**2)**0.5
            h_dumy_,k_dumy_,l_dumy_=formate_hkl(h_dumy,k_dumy,l_dumy)
            if h_dumy[0]==0 and k_dumy[0]==0:
                q=np.pi*2*unitcell.abs_hkl(h_dumy_,k_dumy_,l_dumy_)
                pre_factor=(np.exp(-exp_const*rgh.mu/q))*(4*np.pi*re/auc)*3e6
                f_dumy = pre_factor*SCALES[0]*rough_dumy*sample.calc_f4_specular(h_dumy_, k_dumy_, l_dumy_,RAXR_EL)
            else:
                #f_dumy = rough_dumy*sample.calc_f4(formate_hkl(h_dumy,k_dumy,l_dumy)[0],formate_hkl(h_dumy,k_dumy,l_dumy)[1],formate_hkl(h_dumy,k_dumy,l_dumy)[2])
                f_dumy = rough_dumy*sample.calc_f4(h_dumy_, k_dumy_, l_dumy_)

            label=str(int(h[0]))+str(int(k[0]))+'L'
            #if label=="10L":
            #    print l_dumy
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis]),axis=1)
        else:#to be finished for plotting RAXR models here
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            x = data_set.x
            y = data_set.extra_data['Y']
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            h,k,y=formate_hkl(h,k,y)
            #rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            #f=abs(sample.cal_structure_factor_hematite_RAXR(spectra_index,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h, k, y, x, E0, F1F2,SCALES,rough))
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            f=sample.cal_structure_factor_hematite_RAXR(spectra_index,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h, k, y, x, E0, F1F2,SCALES,rough)
            A_list,P_list=[],[]
            for index_resonant_el in range(len(RESONANT_EL_LIST)):
                A_list_domain=0
                P_list_domain=0
                if RESONANT_EL_LIST[index_resonant_el]!=0:
                    A_list_domain=getattr(VARS['rgh_raxr'],'A_D'+str(index_resonant_el+1)+'_'+str(spectra_index+1))
                    P_list_domain=getattr(VARS['rgh_raxr'],'P_D'+str(index_resonant_el+1)+'_'+str(spectra_index+1))
                A_list.append(A_list_domain)
                P_list.append(P_list_domain)
            A_list_Fourier_synthesis.append(A_list)
            P_list_Fourier_synthesis.append(P_list)
            HKL_list_raxr[0].append(h[0])
            HKL_list_raxr[1].append(k[0])
            HKL_list_raxr[2].append(y[0])
            label=str(int(h[0]))+'_'+str(int(k[0]))+'_'+str(y[0])
            plot_raxr_container_experiment[label]=np.concatenate((x[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_raxr_container_model[label]=np.concatenate((x[:,np.newaxis],f[:,np.newaxis]),axis=1)
            #if spectra_index==0:print f[6]
            spectra_index+=1
    A_list_formated,P_list_formated={},{}
    if A_list_Fourier_synthesis!=[]:
        shape_A_P=len(A_list_Fourier_synthesis[0])
        for each_A_P in range(shape_A_P):
            A_list_formated['Domain'+str(each_A_P)]=np.array(A_list_Fourier_synthesis)[:,each_A_P]
            P_list_formated['Domain'+str(each_A_P)]=np.array(P_list_Fourier_synthesis)[:,each_A_P]

    A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_hematite(HKL_list_raxr[0],HKL_list_raxr[1],HKL_list_raxr[2],RAXR_EL)
    #dump CTR data and profiles
    hkls=['00L','02L','10L','11L','20L','22L','30L','2-1L','21L']
    plot_data_list=[]
    for hkl in hkls:
        try:
            plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
        except:
            pass
    pickle.dump(plot_data_list,open(os.path.join(output_file_path,"temp_plot"),"wb"))
    #dump raxr data and profiles
    pickle.dump([plot_raxr_container_experiment,plot_raxr_container_model],open(os.path.join(output_file_path,"temp_plot_raxr"),"wb"))
    #dump electron density profiles
    #e density based on model fitting
    sample.plot_electron_density_hematite(sample.domain,file_path=output_file_path,z_min=z_min,z_max=z_max,raxs_el=RAXR_EL,height_offset=ref_height)#dumpt file name is "temp_plot_eden"
    #e density based on Fourier synthesis with fit A and P

    #z_plot,eden_plot,eden_domains=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_Fourier_synthesis).transpose(),np.array(A_list_Fourier_synthesis).transpose(),z_min=z_min,z_max=z_max,resonant_el=RAXR_EL,resolution=1000,water_scaling=0.33
    z_plot,eden_plot,eden_domains=sample.fourier_synthesis_hematite(np.array(HKL_list_raxr),P_list_formated,A_list_formated,z_min=z_min,z_max=z_max,resonant_el=RAXR_EL,resolution=1000,water_scaling=0.33)
    pickle.dump([np.array(z_plot)-ref_height,eden_plot,eden_domains],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis"),"wb"))
    #e density based on Fourier synthesis from calculated A and P
    z_plot_sub,eden_plot_sub,eden_domains_sub=sample.fourier_synthesis_hematite(np.array(HKL_list_raxr),P_list_calculated_sub,A_list_calculated_sub,z_min=z_min,z_max=z_max,resonant_el=RAXR_EL,resolution=1000,water_scaling=0.33)
    pickle.dump([np.array(z_plot_sub)-ref_height,eden_plot_sub,eden_domains_sub],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis_sub"),"wb"))
    return None

def print_help_doc():
    print(
    """
    running_mode(bool)
        if true then disable all the I/O function
    SORBATE(list of list with each list item containing the sorbate element in each domain)
        element symbol for sorbate
        the shape of SORBATE is the same as pickup_index
    BASAL_EL(a list of elements to specify the anchor reference to the ternary complex)
        only used in the domain containing ternary complex species
        The first item in each item list is alway None, since the first one is referenced to the substrate surface
        By default each item after the first one has a referenced element from the previous element)
    pickup_index(a list of index list with items from the match index table above)
        representative of different binding configurations for different domains
        make sure the half layer indexes are in front of the full layer indexes
        In this new version, you can have multiple sites being assigned simultaneously on the same domain
        For example,in the case of [[0,6,6],[4],[10,14]] there are three sites assinged to domain1, i.e. bidentate site and the other two outer-sphere site
    sym_site_index(a list of list of [0,1])
        a way to specify the symmetry site on each domain
        you may consider only site pairs in this version ([0,1])
        The shape is the same as pickup_index, except that the inner-most items are [0,1] instead of match index number
        It will be set up automatically
    full_layer(a list of either 0 or 1 with 0 for short and 1 for long slab)
        used to specify the step for full layer termination, the items in this list must have a one to one corresponding to the items appearing in the pick_up_index for FL
    half_layer(a list of either 2 or 3 with 2 for short and 3 for long slab)
        Analogous to full_layer but used for half layer termination case
    full_layer_pick(a list of value of either None, or 0 or 1)
        used to specify the full layer type, which could be either long slab (1) or short slab (0)
        don't forget to set None for the half layer termination domain
        Again Nones if any must be in front of numbers (Half layer domains in front of full layer domains)
        concerns about None has been automatically setup in this new version
    half_layer_pick(a list of value of either None, or 2 or 3)
        Analogous to full_layer_pick but used for half layer termination
    OS_X(Y,Z)_REF(a list of None,or any number)
        set the reference coordinate xyz value for the outer-sphere configuration, which could be on either HL or FL domain
        these values are fractional coordinates of sorbates
        if N/A then set it to None
        such setting is based on the symmetry operation intrinsic for the hematite rcut surface, which have the following relationship
        x1+x2=0.5/1.5, y1-y2=0.5 or -0.5, z1=z2
        The shape is like [[],[]], each item corresponds to different domains
        The number of items within each domain is twice (considering symmetry site pair) the number of sorbate for that domain
    DOMAIN_GP(a list of list of domain indexs)
        use this to group two domains with same surface termination (HL or FL) together
        the associated atom groups for both surface atoms and sorbates will be created (refer to manual)
        This feature is not necessary and so not supported anymore in this version.
    water_pars(a lib to set the interfacial waters quickly)
        This water molecules are regarded as adsorbed water molecules with lateral and vertical ordering which will have effect on both the specular and offspecular rods
        you may use default which has no water or turn the switch off and set the number and anchor points
    layered_water_pars(a lib to set layered water structure)
        layered water structure factor only have effect on the specular rod
        Based on the equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
        key of 'yes_OR_no':a list of 0 or 1 to specify whether or not considering the layered water structure
        key of 'ref_layer_height' is a list of atom ids (domain information not needed) to specify the reference height for the layered water heights
    layered_sorbate_pars(a lib to set layered sorbate structure)
        pretty much the same as layered_water_pars
        key of 'el' is the symbol for the resonant element
    USE_BV(bool)
        a switch to apply bond valence constrain during surface modelling
    TABLE_DOMAINS(list of 0 or 1, the length should be higher than the total domain number)
        specify whether or not generate the associated pars for each domain
        [0,1,1] means only generate the pars for last two domains
    RAXR_EL(resonant element)
    NUMBER_SPECTRA(number of RAXR spectras)
        Note each spectra, there will be an independent set of fitting parameters (a,b,A,P)
    RESONANT_EL_LIST(a list of integer number (either 1 or 0))
        Used to specify the domain containing resonant element
        0 means no resonant element on the domain
        1 means considering resonant element on the domain
    E0=13000
        Center of Scan energy range for RAXR data
    F1F2_FILE="Pb.f1f2"
        Absolute file path for the f1f2 file containing anomalous correction items at each energy
    F1F2=None
        Global variable to hold the f1f2 values after loading the f1f2 file
    COVALENT_HYDROGEN_RANDOM(bool)
        a switch to not explicitly specify the protonation of surface functional groups
        different protonation scheme (0,1 or 2 protons) will be tried and compared, the one with best bv result will be used
    BV_OFFSET_SORBATE(a list of number)
        it is used to define the acceptable range of bond valence sum for sorbates
        [bv_eachbond*N_bonds-offset,bv_eachbond*N_bonds] will be the range
        set a random number for a clean surface (no sorbate), but don't miss that
    SEARCH_RANGE_OFFSET(a number)
        used to set the searching range for an atom, which will be used to calculate the bond valence sum of sorbates
        the radius of the searching sphere will be the ideal bond length plus this offset
    commands(a list of str to be executed inside sim function)
        eg. ['gp_O1O2_O7O8_D1.setoc(gp_Fe4Fe6_Fe10Fe12_D1.getoc())']
        used to expand the funtionality of grouping or setting something important
    USE_COORS(a list of [0,0] or [1,1] with two items for two symmetry sites)
        you may want to add sorbates by specifying the coordinates or having the program calculate the position from the geometry setting you considered
        eg1 USE_COORS=[[0,0]]*len(pickup_index) not use coors for all domains
        eg2 USE_COORS=[[1,1]]*len(pickup_index) use coors for all domains
        eg3 USE_COORS=[[0,0],[1,1],[1,1]] use coors for only domain2 and domain3
    COORS(a lib specifying the coordinates for sorbates)
        keys of COORS are the domain index and site index, ignore domain with no sorbates
        len(COORS[(i,j)]['sorbate'])=1 while len(COORS[(i,j)]['oxygen'])>=1, which is the number of distal oxygens
        make sure the setup matches with the pick_up index and the sym_site_index as well as the number of distal oxygens
        if you dont consider oxygen in your model, you still need to specify the coordinates for the oxygen(just one oxygen) to avoid error prompt
    O_NUMBER_HL/FL(a list of list of [a,b],where a and b are integer numbers)
        one to one corresponding for the number of distal oxygens, which depend on local structure and binding configuration
        either zero oxygen ligand or enough ligands to complete coordinative shell
    O_NUMBER_HL/FL_EXTRA(used to define the distal oxygen number for a surface species binding to the distal oxygen of basal element)
    MIRROR(a list of true or false)
        Used to specify the way you add a distal oxygen to a surface complex with monodentate or bidentate binding configuration
        Or in a case of tridentate binding mode with octahedral local structure
    SORBATE_NUMBER_HL/FL(a list of list of [a], a can be either 1 or 2 or 0 for clean surface)
        If considering two symmetry sites, then a=2
        If considering one site (distribute the two on two different domains), then a=1
        If considering clean surface, then a=0
    SORBATE_NUMBER_HL/FL_EXTRA(used to specify the number of outer-part of ternary complex species)
    COUNT_DISTAL_OXYGEN(bool)
        True then consider bond valence also for distal oxygen,otherwise skip the bv contribution from distal oxygen
    ADD_DISTAL_LIGAND_WILD(list of bool)
        the distal oxygen could be added by specifying the pars for the spherical coordinate system (r, theta, phi), which is called wild here, or be added
        in a specific geometry setting for a local structure (like tetrahedra)
        you can specify different case for different domains
        and this par is not applicable to outersphere mode, which should be set to None for that domain
    DOMAINS_BV(a list of integer numbers)
        Domains being considered for bond valence constrain, counted from 0
    BOND_VALENCE_WAIVER(a list of oxygen atom ids [either surface atoms or distals] with domain tag)
        When each two of thoes atoms in the list are being considered for bond valence, the valence effect will be ignored no matter how close they are
        Be careful to select atoms as bond valence waiver
    GROUPING_SCHEMES(a list of lists with two items, with each item being the domain index starting from 0)
        Define how you want to group the surface atoms together from two different domains with same termination type
        A function will generate all the associated commands to do the grouping
        [[0,1]] means group surface atoms from the first (0) and second domain(1)
        If you dont want to do any grouping, set this to be []
    GROUPING_DEPTH(a list of integers less than 10)
        Define how deep you want to group your atoms. You can define a maximum grouping depth to 10
        You should count the atom layers upward from the 10th atom layer
        [6,10] means you want to group the 5th atom layer to 10th atom layer for domain 1 and group all top ten atom layers together for domain2
        Don't forget that you have a Iron layer which is explicitly included in HL but the occ set to 0 to account for the missing Fe sites
        So you should count that atom layer too when considering the grouping depth
    #############################################code update logs############################################################################
    ##version 1##
    consider sorbate (Pb and Sb) of any combination
    #########################the order of building up the interfacial structure#############################
    ##surface atoms-->hydrogen atoms for surface oxygen atoms-->sorbate sets (metal and distal oxygens)-->hydrogen for distal oxygens-->water and the associated hydrogen atoms###
    #########################naming rules###################
        ###########atm ids################
        sorbate: Pb1_D1A, Pb2_D2B, HO1_Pb1_D1A,HO2_Pb1_D1A, HO1_Sb2_D1A
        water: Os1_D1A
        surf atms: O1_1_0_D1A (half layer), O1_1_1_D1A(atoms of second slab)
        hydrogen atoms:
            HB1_O1_1_0_D1A, HB2_O1_1_0_D1A(doubly protonated surface oxygen)-->r[phi,theta]_H_1_1, r[phi,theta]_H_1_2 (first number in the tag is the index of surface oxygen, and the second the index of hydrogen atom)
            HB1_HO1_Pb1_D1A, HB2_HO1_Pb1_D1A(doubly protonated distal oxygen)-->r[phi,theta]_H_D_1_1, r[phi,theta]_H_D_1_2 (first number in the tag is the index of distal oxygen, and the second the index of hydrogen atom)
            HB1_Os1_D1A, HB2_Os1_D1A(doubly protonated water oxygen)-->r[phi,theta]_H_W_1_1_1, r[phi,theta]_H_W_1_1_2 (first number in the tag is the index of water set (single or paired), and the second the index of water in each set (at most 2 for water pair), and the last one for index of hydrogen atom)
        ############group names###########
        gp_sorbates_set1_D1
            group first set (what set1 means, the set index has an increment of 1) of sorbates (including metals and distal oxygens) together without considering symmetry relationship (used to set equal occupancy for metal and its distal oxygens).
        gp_Pb_set1_D1(group two symmetry related Pb atoms together (4 in total if considering those for the symmetry related domains))
            Pb can be replaced with another other element symbol
            set1 means first set consisting of two symmetry related atom within each domain
            you can have multiple sets if you consider multiple sites being occupied simultaneously
            note that the adjacent set indexes are 2 apart, so it goes from set1 to set3 to set5 and so on
        gp_HO1_set1_D1(group two symmetry related distal oxygen atoms together (4 in total if considering those for the symmetry related domains))
            the set index is the same as that described above for the sorbate
            the number after HO specify the distal oxygen, so if there are 3 distal oxygens coordinated with the sorbate, then we use _HO1_, _HO2_ and _HO3_ to distinguish those
        gp_HO_set1_D1
            group all the distal oxygens for the first sorbate (like Pb1, what set1 means here).
            So the set index is different from those defined above (increment of 2) in that it starts from 1 and with an increment of 1.
            Corresponding to gp_sorbates_set1_D1, it is used to set equal u or oc for the distal oxygens associating with the symmetry related sorbate from two different domain (domainA and domainB).
        gp_waters_set1_D1(discrete grouping for each set of water at same layer, group u, oc and dz)
        gp_O1O7_D1(discrete grouping for surface atms, group dx dy in symmetry)
        gp_O1O2_O7O8_D1(sequence grouping for u, oc, dy, dz, or dx in an equal opposite way for O1O2 and O7O8)
        gp_O1O2_O7O8_D1_D2(same as gp_O1O2_O7O8_D1, but group each set of atoms from two different domains, you need to set DOMAIN_GP to have it work)

        some print examples
        #print domain_creator.extract_coor(domain1A,'O1_1_0_D1A')
        #print domain_creator.extract_coor(domain1B,'Pb1_D1B')
        #print_data(N_sorbate=4,N_atm=40,domain=domain1A,z_shift=1,save_file='D://model.xyz')
        #print domain_class_1.cal_bond_valence1(domain1A,'Pb1_D1A',3,False)
    ###########explanation of some global variables###################
    #pars for interfacial waters##
        WATER_NUMBER: must be even number considering 2 atoms each layer
        V_SHIFT: vertical shiftment of water molecules, in unit of angstroms,two items each if consider two water pair
        R=: half distance bw two waters at each layer in unit of angstroms
        ALPHA: alpha angle used to cal the pos of water molecule

        DISCONNECT_BV_CONTRIBUTION=[{('O1_1_0','O1_2_0'):'Pb2'},{}]
        ##if you have two sorbate within the same unit cell, two sorbates will be the coordinative ligands of anchor atoms
        ##but in fact the sorbate cannot occupy the adjacent sites due to steric constrain, so you should delete one ligand in the case of average structure
        ##However, if you consider multiple domains with each domain having only one sorbate, then set the items to be {}
        ##in this case, bv contribution from Pb2 won't be account for both O1 and O2 atom.

        ANCHOR_REFERENCE=[[None],[None]]
        ANCHOR_REFERENCE_OFFSET=[[None],[None]]
        #we use anchor reference to set up a binding configuration in a more intelligible way. We only specify the anchor ref when the two anchor points are
        #not on the same level. The anchor reference will be the center(Fe atom) of the octahedral coordinated by ligands including those two anchors.
        #phi=0 will means that the sorbate is binded in a most feasible way.
        #To ensure the bending on two symmetry site are towards right position, the sorbate attach atom may have reversed order.
        #eg. [O1,O3] correspond to [O4px,O2] rather than [O2,O4px].

        COHERENCE
        #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
        #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
        #value of SF will be calculated followed by being summed up
        #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]

        #IF the covalent_hydrogen_random set to True, then we wont specifically define the number of covalent hydrogen. And it will try [0,1,2] covalent hydrogens
        COVALENT_HYDROGEN_RANDOM=True
        POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]

        If covalent_hydrogen_random is set to false then explicitly define the number of covalent hydrogen here

        ADD_DISTAL_LIGAND_WILD=True means adding distal oxygen ligand in a spherical coordinated system with specified r, theta and phi. Otherwise the distal oxygen are added based on the return value of geometry module

        COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]
        COVALENT_HYDROGEN_NUMBER=[[1,1],[1,1]]
        ##means in domain1 both O1 and O2 will accept one covalent hydrogen (bv contribution of 0.8)

        HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0'],['O1_1_0','O1_2_0']]
        HYDROGEN_NUMBER=[[1,1,1,1],[1,1]]
        ##means in domain1 O1 to O4 will accept one covalent hygrogen (bv contribution of 0.2)

    ##########################some examples to set up sorbates binding under different configurations#######################
    SORBATE=["Pb","Sb"]
    eg1.SORBATE_NUMBER=[[1,0],[0,1]]#two domains:one has one Pb sorbate and the other one has one Sb sorbate
    eg2.SORBATE_NUMBER=[[0,0],[0,1]]#two domains:first one is clean surface, the other one has one Sb sorbate
    eg3.SORBATE_NUMBER=[[1,1],[1,1]]#two domains and each domain has one Pb and one Sb sorbate
    eg4.SORBATE_NUMBER=[[2,0],[0,1]]#first domain has two Pb sorbate

    #len(O_NUMBER)= # of domains
    #len(O_NUMBER[i])= # of sorbates
    O_NUMBER=[[[1],[0]],[[0],[0]]]#Pb sorbate has one oxygen ligand in domain1, while no oxygen ligands in domain2
    O_NUMBER=[[[1,2],[0]],[[0],[3,5]]]#1st Pb sorbate has one oxygen ligand and 2nd Pb has two oxygen ligands in domain1, while in domain2 1st Sb has 3 oxygen ligands and 2nd has 5 oxygen ligands
    SORBATE_LIST=create_sorbate_el_list(SORBATE,SORBATE_NUMBER)
    BV_SUM=[[1.33,5],[1.33,5.]]#pseudo bond valence sum for sorbate

    #len(SORBATE_ATTACH_ATOM)=# of domains
    #len(SORBATE_ATTACH_ATOM[i])=# of sorbates in domaini
    SORBATE_ATTACH_ATOM=[[['O1_1_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0']]]
    SORBATE_ATTACH_ATOM_OFFSET=[[[None,None]],[[None,None,None]]]
    TOP_ANGLE=[[1.38],[1.38]]
    PHI=[[0],[0]]
    R_S=[[1],[1]]
    MIRROR=False
    ##########################offset symbols######################
    None: no translation
    '+x':translate along positive x axis by 1 unit
    '-x':translate along negative x axis by 1 unit
    SAME deal for translation along y axis
    """)
    return None
