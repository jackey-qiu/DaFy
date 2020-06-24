import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

charge = pd.read_excel('/Users/wulingqun/OneDrive/Co_pH_project/charge_transfer_summary.xlsx')
fig,ax = plt.subplots() 
cv1 = ax.bar(np.array(range(6))-0.18,charge['cv1'],width = 0.35) 
cv2 = ax.bar(np.array(range(6))-0.18,charge['cv2'],bottom = charge['cv1'],width = 0.35) 
hor_1 = ax.bar(np.array(range(6))+0.18,charge['hor_1'],width = 0.35) 
ver_1 = ax.bar(np.array(range(6))+0.18,charge['ver_1'],bottom = charge['hor_1'],width = 0.35) 
hor_2 = ax.bar(np.array(range(6))+0.18,charge['hor_2'],bottom = charge['cv1'],width = 0.35) 
ver_2 = ax.bar(np.array(range(6))+0.18,charge['ver_2'],bottom = charge['cv1']+charge['hor_2'],width = 0.35) 
plt.ylabel('Charge (mC)') 
plt.xticks(range(6),['pH10','pH13','pH8','pH13','pH7','pH13']) 
plt.legend((cv1[0],cv2[0],hor_1[0],ver_1[0],hor_2[0],ver_2[0]),('cv1','cv2','hor_1','ver_1','hor_2','ver_2')) 
plt.show()
