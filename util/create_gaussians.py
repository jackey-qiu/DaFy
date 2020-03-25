import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

data = {}
y = 0
x_values = np.linspace(-10, 15, 1000)
for mu, sig in [(-1, 1), (0, 2), (2, 0.5),(5,2)]:
    y += gaussian(x_values, mu, sig)
noise = np.random.normal(0,0.02,1000)
y = y + noise
data['L'] = x_values
data['H'] = x_values*0
data['K'] = x_values*0
data['Y'] = x_values*0
data['I'] = y
data['error'] = noise
data['dL'] = x_values*0
data['LB'] = x_values*0
data_pd=pd.DataFrame(data)
data_pd.to_csv('/Users/cqiu/app/DaFy/dump_files/gaussian_data.csv',sep="\t",columns=['L','H','K','Y','I','error','LB','dL'],\
                                 index=False, header=['#L','H','K','Y','I','error','LB','dL'])
plt.plot(x_values,y)
plt.show()