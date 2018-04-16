
import os
import numpy as np
import matplotlib.pyplot as plt


dir = '/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/finals/GEP2M_LinearPolicy/'

ind_test = []
for i in range(1000):
    ind_test.append(3+12*i)

perfs_100 = []
for i, f in enumerate(os.listdir(dir)):
    if 'Cheetah' in f:
        all_perfs = np.loadtxt(dir+f)
        perfs_100.append(all_perfs[-100:].mean())
        plt.figure()
        plt.hist(all_perfs[-100:])
        plt.savefig(dir+f[19:]+'_hist_perfs100')
        perfsToPlot = []
        for j in ind_test:
            # print(j)
            perfsToPlot.append(all_perfs[j:j+10].mean())
        np.savetxt(dir+f[19:]+'_perfsToPlot', np.array(perfsToPlot))

np.savetxt(dir+'perfs100', np.array(perfs_100))
