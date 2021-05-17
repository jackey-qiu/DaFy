#bidentate case for trigonalpyramid motif
BD_CASE_TP = {'ids':str(['Pb1', 'O1']), 
            'els': str(['Pb', 'O']), 
            'anchor_id': 'Pb1', 
            'substrate_domain':'surface_1', 
            'attach_atm_ids':str(['O1_1_0','O1_2_0']),
            'offset':str([None,None]),
            'anchor_ref':None,
            'anchor_offset':None,
            'mode':'BD',
            'mirror':'False', 
            'switch':'False',
            'top_angle':str(70.),
            'phi':str(0.),
            'edge_offset':str([0,0]), 
            'angle_offset':str(0),
            'lat_pars':str([5.038, 5.434, 7.3707, 90, 90, 90]),
            'T':'None',
            'T_INV':'None'}

TD_CASE_TP = {'ids':str(['Pb1']), 
            'els': str(['Pb']), 
            'anchor_id': 'Pb1', 
            'substrate_domain':'surface_1', 
            'attach_atm_ids':str(['O1_1_0','O1_2_0','O1_3_0']),
            'offset':str([None,None,None]),
            'mode':'TD',
            'mirror':'False', 
            'top_angle':str(70.),
            'lat_pars':str([5.038, 5.434, 7.3707, 90, 90, 90]),
            'T':'None',
            'T_INV':'None'}

BD_CASE_TH = {'ids':str(['As1', 'O1', 'O2']), 
            'els': str(['As', 'O', 'O']), 
            'anchor_id': 'As1', 
            'substrate_domain':'surface_1', 
            'attach_atm_ids':str(['O1_1_0','O1_2_0']),
            'offset':str([None,None]),
            'anchor_ref':None,
            'anchor_offset':None,
            'mode':'BD',
            'top_angle_offset':str(0.),
            'phi':str(0.),
            'edge_offset':str([0,0]), 
            'angle_offset':str([0,0]),
            'lat_pars':str([5.038, 5.434, 7.3707, 90, 90, 90]),
            'T':'None',
            'T_INV':'None'}

TD_CASE_TH = {'ids':str(['As1', 'O1']), 
            'els': str(['As', 'O']), 
            'anchor_id': 'As1', 
            'substrate_domain':'surface_1', 
            'attach_atm_ids':str(['O1_1_0','O1_2_0','O1_3_0']),
            'offset':str([None,None,None]),
            'mode':'TD',
            'mirror':'True',
            'center_offset':str(0.),
            'edge_offset':str(0), 
            'lat_pars':str([5.038, 5.434, 7.3707, 90, 90, 90]),
            'T':'None',
            'T_INV':'None'}
'''
#corner-sharing of type 1 in short Half layer termination
CS1_TP_SHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_7_0','O1_8_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

#corner-sharing of type 1 in long Half layer termination
CS1_TP_LHL ={**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_1_0','O1_2_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}} 

#corner-sharing of type 1 in short Full layer termination
CS1_TP_SFL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_5_0','O1_6_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}} 

#corner-sharing of type 1 in long Full layer termination
CS1_TP_LFL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_11_t','O1_12_t'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

#edge-sharing of type 1 in short half layer termination
ES1_TP_SHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_7_0','O1_9_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

ES1_TP_LHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_1_0','O1_3_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

ES1_TP_SFL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_8_0','O1_6_0'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

ES1_TP_LHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_2_0','O1_12_t'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None})}}

#edge-sharing of type 2 in short half layer termination
ES2_TP_SHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_7_0','O1_10_0'],'offset':[None,'+y'],'anchor_ref':None,'anchor_offset':None})}}

ES2_TP_LHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_1_0','O1_4_0'],'offset':[None,'+y'],'anchor_ref':None,'anchor_offset':None})}}

ES2_TP_SFL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_5_0','O1_8_0'],'offset':[None,'+x'],'anchor_ref':None,'anchor_offset':None})}}

ES2_TP_LHL = {**BD_CASE, **{'anchored_ids':str({'attach_atm_ids':['O1_11_t','O1_2_0'],'offset':[None,'-x'],'anchor_ref':None,'anchor_offset':None})}}
'''