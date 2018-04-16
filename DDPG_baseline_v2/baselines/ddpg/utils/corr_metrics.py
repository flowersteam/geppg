import numpy as np
from scipy.stats import ks_2samp, ttest_ind, pearsonr
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
font = {'family' : 'normal',
        'size'   : 45}


matplotlib.rc('font', **font)
data_path='/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/finals/all_runs/'

scores_our = []
scores_litt=[]
for f in os.listdir(data_path):
    if len(f)<4:
        scores = np.loadtxt(data_path+f+'/scores')
        scores_our.append(scores[1,0])
        scores_litt.append(scores[0, 0])

r = pearsonr(np.array([scores_litt]).squeeze(), np.array([scores_our]).squeeze())
fig2 = plt.figure(figsize=(22, 13), frameon=False)
plt.scatter(scores_litt, scores_our,s=100,color='k')
scores_litt=np.array(scores_litt)
slope, intercept, r_value, p_value, std_err = stats.linregress(scores_litt, scores_our)
print(slope)
print(intercept)
print(r_value)
print(p_value)
plt.plot(scores_litt, slope*scores_litt +intercept, 'r-', linewidth=3.)
plt.annotate(r'$abs = 1.04 \times fin + 265$',xy=(0.5, 0.15), xycoords='axes fraction')
plt.annotate(r'$r=0.99, p=2.8 \times 10^{-124}$', xy=(0.5, 0.05), xycoords='axes fraction')

# plt.annotate(r'$abs = 0.18 \times fin + 76$', xy=(0.5, 0.15), xycoords='axes fraction')
# plt.annotate(r'$r=0.33, p=0.011$', xy=(0.5, 0.05), xycoords='axes fraction')

plt.xlabel('final metric')
plt.ylabel('absolute metric')
plt.title('Evaluation metrics on HC')
plt.savefig(data_path+'scatter_metrics.png',bbox_inches = 'tight')

np.savetxt(data_path+'correlation_metrics', np.array(r))

