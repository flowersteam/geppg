"""
This script is made of plotting multiple curves.
data_folder must refer to a directory that contains eval_performances and scores_absolute files ordered as such:
1_eval_performances_algo1, 2_eval_performances_algo2, ... , 3_scores_absolute_algo1, 4_scores_absolute_algo2, ...
the variables env_name and env should be set to 'HC' for HalfCheetah and CMC for Continuous Mountain Car
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import copy
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


font = {'family' : 'normal',
        'size'   : 45}

matplotlib.rc('font', **font)


env_name ='HC' #'CMC'
env=env_name.lower()

# put the eval_performances files in a folder, order the name 1_name1, 2_name2
data_folder = '../results/plot_folder/'

gep=True
gep_switch = [500] #number of gep episodes before switch


saving_name = 'perf'
legends=['DDPG, param. pert.', 'GEP-PG, param. pert.']
title = env_name+': '+ 'comparisons of DDPG and GEP-PG'


x_step = 2 # resolution of 2k steps for HC and CMC
error_type = 'sem' # 90#  100# 'std'#  'sem'#   80#  'std'
main_curve='mean'#  'median'#
list_files = sorted(os.listdir(data_folder))
gep_colors = [[0.3010,0.745,0.933],[0,0.447,0.7410],[222/255,21/255,21/255],[0.635,0.078,0.184],[0.494,0.1844,0.556], [0.466,0.674,0.188], ]
matlab_colors = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.494,0.1844,0.556],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
matlab_colors2 = [[0.494,0.1844,0.556],[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]

if gep:
    colors=gep_colors
else:
    colors = matlab_colors2
fig = plt.figure(figsize=(22, 13), frameon=False)
ax  = fig.add_subplot(111)
plt.xlabel(r"epochs")
plt.ylabel("performance")
for i in range(len(legends)):
    legends[i] = legends[i]
plt.title(title)
ind = 0
scores_absolute = []
scores_final = []

for f in list_files:
    if 'absolute' in f:
        scores_absolute.append(np.loadtxt(data_folder+f))
    elif 'eval' in f:
        eval_perfs = np.loadtxt(data_folder+f)
        n_runs = eval_perfs.shape[0]
        if eval_perfs.shape[1]%2==0: eval_perfs = np.concatenate([np.zeros([n_runs,1]), eval_perfs],axis=1)
        steps = range(0,eval_perfs.shape[1]*x_step,x_step)

        if main_curve == 'mean':
            toPlot_av = np.nanmean(eval_perfs, axis=0)
        elif main_curve == 'median':
            toPlot_av = np.nanmedian(eval_perfs, axis=0)
        else:
            raise NotImplementedError

        if error_type == 'std':
            toPlot_error_add = np.nanstd(eval_perfs, axis=0)
            toPlot_error_sub = np.nanstd(eval_perfs, axis=0)
        elif error_type == 'sem':
            toPlot_error_add = np.nanstd(eval_perfs, axis=0) / np.sqrt(n_runs)
            toPlot_error_sub = np.nanstd(eval_perfs, axis=0) / np.sqrt(n_runs)
        elif type(error_type) == int:
            assert (error_type <= 100 and error_type > 0), 'error_type must be between 0 and 100'
            toPlot_error_add = np.nanpercentile(eval_perfs, int(100 - (100 - error_type) / 2), axis=0) - toPlot_av
            toPlot_error_sub = -np.nanpercentile(eval_perfs, int((100 - error_type) / 2), axis=0) + toPlot_av
        else:
            raise NotImplementedError

        if type(error_type) == int:
            error_type = str(error_type) + '%'

        maincolor = list(colors[ind])
        maincolor.append(1)
        facecolor = list(colors[ind])
        facecolor.append(0.5)
        edgecolor = list(colors[ind])
        edgecolor.append(0.7)
        plt.figure(fig.number)
        ax.plot(steps, toPlot_av, label="label", c=maincolor)
        ax.fill_between(steps, toPlot_av - toPlot_error_sub, toPlot_av + toPlot_error_add,
                         alpha=0.5, edgecolor=edgecolor, facecolor=facecolor)

        if env_name == 'HC':
            plt.xlim([0,10000])
            plt.ylim([-1000, 8000])
        elif env_name == 'CMC':
            plt.xlim([0,500])
            plt.ylim([-100, 100])

        ind+=1
        scores_final.append(eval_perfs[:,-10:].mean(axis=1))
if gep:
    for s in gep_switch:
        plt.axvline(x=s, linestyle='--', color='k')

plt.legend(legends)
leg = plt.legend(legends)
leg_lines = leg.get_lines()
plt.setp(leg_lines, linewidth=10)
plt.savefig(data_folder + 'results.png', bbox_inches='tight')

# plot histogram of absolute performances
scores_array = np.array(scores_absolute)
if env_name == 'HC':
    range_scores = [scores_array.min(), 7500] #scores_array.max()]
elif env_name == 'CMC':
    range_scores = [scores_array.min(),100]
fig4 = plt.figure(figsize=(20,13), frameon=False)
ax = fig4.add_subplot(111,projection='3d')
for c, z ,i in zip(colors[:len(legends)], [0, 10, 20, 30, 40, 50],[0,1,2,3,4, 5]):
    fig_tmp = plt.figure()
    ys, bins= np.histogram(scores_absolute[i], range=range_scores)
    xs = (bins[:-1] + bins[1:]) / 2
    plt.figure(fig4.number)
    cs = [c] * len(xs)
    if env_name == 'HC':
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8, width=500)
    elif env_name == 'CMC':
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8, width=8)
    ti=ax.get_xticks()

if env_name == 'HC':

    ax.set_xticks(list(range(2000, 7001, 1000)))
    ax.set_xticklabels(range(2,8,1))
    ax.set_xlim(range_scores)
else:
    ax.set_xticks(list(range(-20, 101, 20)))
    ax.set_xlim([-20,100])

ax.set_yticks([])
ax.set_zticks(list(range(0,20,4)))
ax.set_zlim([0.4,20])
# ax.set_xlabel('performance returns')
leg = ax.legend(legends,loc='upper left',fontsize=35)
plt.savefig(data_folder+'hist_3d_'+saving_name+'_'+env+'_our_measure', bbox_inches = 'tight')
