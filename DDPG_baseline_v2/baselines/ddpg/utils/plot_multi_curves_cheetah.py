
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import copy
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
font = {'family' : 'normal',
        'size'   : 35}

matplotlib.rc('font', **font)


env_name ='HC' #'CMC'#
env='hc'#'cmc'#
# put the eval_performances files in a folder, order the name 1_name1, 2_name2
data_folder = '/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/finals/figures/noise_plot/'

gep=False

saving_name = 'noise'
legends=['No noise', 'Action pert.: OU ($\sigma=0.3$)','Action pert.: OU decreasing ($\sigma=0.6$)','Parameter pert.']
title = env_name+': '+ 'undirected exploration on DDPG'

error_type = 'sem' # 90#  100# 'std'#  'sem'#   80#  'std'
main_curve='mean'#  'median'#
hist_superposed = False
list_files = sorted(os.listdir(data_folder))
gep_colors = [[0.3010,0.745,0.933],[0,0.447,0.7410],[222/255,21/255,21/255],[0.635,0.078,0.184],[0.466,0.674,0.188]]
matlab_colors = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.494,0.1844,0.556],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
matlab_colors2 = [[0.494,0.1844,0.556],[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]

if gep:
    colors=gep_colors
else:
    colors = matlab_colors2
fig = plt.figure(figsize=(22, 13), frameon=False)
ax  = fig.add_subplot(111)
plt.xlabel(r"time steps ($\times 10³$)")
plt.ylabel("performance")
for i in range(len(legends)):
    legends[i] = legends[i]#+main_curve+' ('+error_type+')'
plt.title(title)
ind = 0
fig2 = plt.figure(figsize=(22, 13), frameon=False)
ax2  = fig.add_subplot(111)
plt.xlabel('performance returns')
plt.title(env_name+': performances (final metric)')
fig3 = plt.figure(figsize=(22, 13), frameon=False)
plt.xlabel('performance returns')
plt.title(env_name+': performances (absolute metric)')
scores_our = []
scores_litt = []

for f in list_files:
    if 'scores' in f:
        print(f)
        scores_our.append(np.loadtxt(data_folder+f))
    elif 'eval' in f:
        eval_perfs = np.loadtxt(data_folder+f)
        n_runs = eval_perfs.shape[0]
        if eval_perfs.shape[1]%2==0: eval_perfs = np.concatenate([np.zeros([n_runs,1]), eval_perfs],axis=1)
        steps = range(0, eval_perfs.shape[1] * 2, 2)

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
        plt.ylim([-500, 5300])
        plt.xlim([0,500])
        if env_name=='CMC':
            plt.ylim([-100, 110])
        ind+=1
        scores_litt.append(eval_perfs[:,-10:].mean(axis=1))
if gep:
    if env_name == 'CMC':
        range_switch = np.loadtxt(data_folder+'range_switch')
        plt.fill_between([range_switch[0]/1000, range_switch[1]/1000], [-110, -110], [110, 110],
                         alpha=0.2, edgecolor=(0, 0, 0, 0.05), facecolor=(0, 0, 0, 0.05))
    else:
        plt.axvline(x=500, linestyle='--', color='k')




if hist_superposed:
    for i in range(len(legends)):
        plt.figure(fig3.number)
        c = list(colors[i])
        c.append(0.5)
        plt.hist(scores_our[i], color=c)
        plt.figure(fig2.number)
        plt.hist(scores_litt[i], color=colors[i]+[0.5])
else:
    plt.figure(fig3.number)
    plt.hist(tuple(scores_our),  color=colors[:len(legends)])
    plt.figure(fig2.number)
    plt.hist(tuple(scores_litt),  color=colors[:len(legends)])
    # plt.xlim([1010, 7100])


plt.figure(fig.number)
ax.legend(legends, loc='lower right')
handles, labels = ax.get_legend_handles_labels()
handles = [copy.copy(ha) for ha in handles ]
[ha.set_linewidth(5) for ha in handles ]
h = mpatches.Patch(color='white', label='legend as in (b)')
leg = plt.legend(handles=[h],frameon=False,  loc='lower right')

plt.savefig(data_folder+saving_name+'_'+env+'.png', bbox_inches = 'tight')

plt.figure(fig2.number)
plt.legend(legends, loc='upper right',frameon=False)
plt.savefig(data_folder+'hist_'+saving_name+'_'+env+'_litt_measure', bbox_inches = 'tight')

plt.figure(fig3.number)
plt.legend(legends, loc='upper right',frameon=False)
plt.savefig(data_folder+'hist_'+saving_name+'_'+env+'_our_measure', bbox_inches = 'tight')

fig4 = plt.figure(figsize=(20,13), frameon=False)
ax = fig4.add_subplot(111,projection='3d')
for c, z ,i in zip(colors[:len(legends)], [0, 10, 20, 30],[0,1,2,3]):
    # xs = np.arange(1000,7000,1000)+4
    fig_tmp = plt.figure()
    xs = []
    ranges=[1500,7500]
    tmp=plt.hist(scores_our[i], range=ranges)[1]
    for j in range(len(tmp)-1):
        xs.append(int((tmp[j]+tmp[j+1])/2))
    xs=np.array(xs)
    ys = plt.hist(scores_our[i], range=ranges)[0]
    plt.close()
    plt.figure(fig4.number)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    # cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8, width=500)
    ti=ax.get_xticks()

    # plt.xticks([range(0,50,10)], [],[])
ax.set_xticks(list(range(2000,7100, 1000)))
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim([1500,7500])
ax.set_zlim([0,20])
# plt.show()
# ax.set_xlabel('performance returns')
ax.legend(legends,loc='upper right',fontsize=35)
plt.savefig(data_folder+'hist_3d_'+saving_name+'_'+env+'_our_measure', bbox_inches = 'tight')
