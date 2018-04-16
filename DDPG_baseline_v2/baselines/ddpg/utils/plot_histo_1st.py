import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
font = {'family' : 'normal',
        'size'   : 40}

matplotlib.rc('font', **font)

file_location1 = '/home/flowers/Desktop/Scratch/experiments/first_successes/CMC_successes_simpleLinear_policy'
file_location2 = '/home/flowers/Desktop/Scratch/experiments/first_successes/CMC_successes_complex_policy'
file_location4 = '/home/flowers/Desktop/Scratch/experiments/first_successes/1st_success_steps_OU'
file_location3 = '/home/flowers/Desktop/Scratch/experiments/first_successes/1st_success_Plappert'

n_steps_success = []
n_steps_success.append(np.loadtxt(file_location1)/1000)
n_steps_success.append(np.loadtxt(file_location2)/1000)
n_steps_success.append(np.loadtxt(file_location3)/1000)
n_steps_success.append(np.loadtxt(file_location4)/1000)
n_runs = len(n_steps_success)

fig = plt.figure(figsize=(20,13), frameon=False)
legends = ['GEP, linear policy', 'GEP, complex policy', 'DDPG, parameter pert.','DDPG, action pert.' ]
plt.xlabel(r'time steps $(\times 10³)$')
plt.title('CMC: number of steps before reaching the first reward',fontsize=40)
colors = [[0,0.447,0.7410,0.6],[0.85,0.325,0.098,0.5],[0.466,0.674,0.188],[0.494,0.1844,0.556,0.7],[0.929,0.694,0.125,0.5],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
colors2 = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.494,0.1844,0.556],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]

for i in range(n_runs):
    if i==2 or i==3:
        n_hist = np.argwhere(n_steps_success[i]!=50).shape[0]
        nbelow10k = np.argwhere(n_steps_success[i]<10).shape[0]/10
        print(nbelow10k)
        print('Only '+str(int(n_hist/10))+ '% of the runs achieved a success before 50k steps for '+legends[i])
        n_steps_success[i] = n_steps_success[i][np.argwhere(n_steps_success[i]!=50)]
    print('mean number of steps before reaching a reward for '+legends[i]+': ', n_steps_success[i].mean())
    plt.hist(n_steps_success[i], 30, color=tuple(colors[i]))
    plt.ylim([0,390])

plt.legend(legends)

plt.savefig('/home/flowers/Desktop/Scratch/experiments/first_successes/hist2',bbox_inches = 'tight')


fig3 = plt.figure(figsize=(20,13), frameon=False)
ax = fig3.add_subplot(111,projection='3d')
for c, z ,i in zip(colors[:4], [30, 20, 10, 0],[0,1,2,3]):
    xs = np.arange(0,50,5)+3
    fig_tmp = plt.figure()
    # xs = []
    # tmp=plt.hist(n_steps_success[i])[1]
    # for j in range(len(tmp)-1):
    #     xs.append(int((tmp[j]+tmp[j+1])/2))
    # xs=np.array(xs)
    ys = plt.hist(n_steps_success[i])[0]
    plt.close()
    plt.figure(fig3.number)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    # cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8, width=4)
    ti=ax.get_xticks()
    # plt.xticks([range(0,50,10)], [],[])
ax.set_xticks(list(range(0,51,10)))
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim([0,50])
# plt.show()
# ax.set_xlabel(r'time steps $(\times 10³)$')
ax.legend(legends,loc='upper right',fontsize=35)
# plt.title('CMC: number of steps before reaching the first reward',fontsize=40)

plt.savefig('/home/flowers/Desktop/Scratch/experiments/first_successes/hist_depth', bbox_inches = 'tight')



