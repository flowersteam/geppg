import json
import os
import matplotlib.pyplot as plt
import numpy as np

data_path='/home/sigaud/PycharmProjects/ddpg_baseline_openAI_fork/results/MountainCarContinuous-v0/'
name_run = 'DDPG_2M_CMC'
name_algo = 'DDPG'

def extract_performances(filename):

    with open(filename) as json_data:
        lines = json_data.readlines()
    eval_rewards = [0]
    steps = [0]
    for line in lines:
        episode_data = json.loads(line)
        if 'eval/return' in episode_data:
            step = episode_data['total/epochs']*2000
            perf = episode_data['eval/return']
            eval_rewards.append(perf)
            steps.append(step)
    return steps, eval_rewards


def plot_vs(data_path):
    run = {}
    run['perfs'] = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial) < 3:
            print('Extracting: ', trial)
            filename = data_path + trial + '/progress.json'
            # print(filename)
            steps, eval_rewards = extract_performances(filename)
            run['perfs'].append(eval_rewards)

    n_runs = len(run['perfs'])
    eval_perfs = np.zeros([n_runs, len(run['perfs'][0])])
    for i in range(n_runs):
        eval_perfs[i, :] = run['perfs'][i]

    max_steps = 0
    for i in range(n_runs):
        if len(run['perfs'][i]) > max_steps:
            max_steps = len(run['perfs'][i])
    eval_perfs = np.empty([n_runs, max_steps]) * (np.nan)
    for i in range(n_runs):
        eval_perfs[i, :len(run['perfs'][i])] = run['perfs'][i]
    steps = range(0,max_steps*2000,2000)

    inds = np.array(range(n_runs))
    # np.random.shuffle(inds)

    # # modify GEP source
    # for i in range(20,40):
    #     eval_perfs[i][251:]=eval_perfs[i][:750]
    #     eval_perfs[i][0:251] = np.zeros([251])

    print(inds)
    toPlot_av1 = eval_perfs[inds[:int(n_runs/2)]].mean(axis=0)
    toPlot_std1 = eval_perfs[inds[:int(n_runs/2)]].std(axis=0)/10
    toPlot_av2 = eval_perfs[inds[int(n_runs / 2):]].mean(axis=0)
    toPlot_std2 = eval_perfs[inds[int(n_runs / 2):]].std(axis=0)/10
    fig = plt.figure(figsize=(20, 13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title("DDPG")
    # plt.plot(steps, toPlot_av1, label="label", c='#CC6600')
    # plt.fill_between(steps, toPlot_av1 - toPlot_std1, toPlot_av1 + toPlot_std1,
    #                  alpha=0.5, edgecolor='#FF9933', facecolor='#FFB266')
    plt.plot(steps, toPlot_av1, label="label", c='#CC0000')
    plt.fill_between(steps, toPlot_av1 - toPlot_std1, toPlot_av1 + toPlot_std1,
                     alpha=0.5, edgecolor='#FF3333', facecolor='#FF6666')
    plt.plot(steps, toPlot_av2, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av2 - toPlot_std2, toPlot_av2 + toPlot_std2,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    plt.legend(['DDPG 2M','DDPG 1M'],loc=4)

    plt.savefig(data_path+'DDPG_1M_2M', bbox_inches='tight')


def plot_test_variability(data_path,ind):
    run = {}
    run['perfs'] = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial) < 3:
            print('Extracting: ', trial)
            filename = data_path + trial + '/progress.json'
            # print(filename)
            steps, eval_rewards = extract_performances(filename)
            run['perfs'].append(eval_rewards)

    n_runs = len(run['perfs'])
    max_steps = 0
    for i in range(n_runs):
        if len(run['perfs'][i])>max_steps:
            max_steps = len(run['perfs'][i])
    eval_perfs = np.empty([n_runs, max_steps])*(np.nan)
    for i in range(n_runs):
        eval_perfs[i,:len(run['perfs'][i])] = run['perfs'][i]

    steps = range(0,max_steps*2000,2000)

    inds = np.array(range(n_runs))
    np.random.shuffle(inds)

    print(inds)
    toPlot_av1 = np.nanmean(eval_perfs[inds[:int(n_runs/2)]],axis=0)
    toPlot_std1 = np.nanstd(eval_perfs[inds[:int(n_runs/2)]],axis=0)/np.sqrt(int(n_runs/2))
    toPlot_av2 = np.nanmean(eval_perfs[inds[int(n_runs / 2):]],axis=0)
    toPlot_std2 = np.nanstd(eval_perfs[inds[int(n_runs / 2):]],axis=0)/np.sqrt(int(n_runs/2))
    fig = plt.figure(figsize=(20, 13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title(name_algo)
    # plt.plot(steps, toPlot_av1, label="label", c='#CC6600')
    # plt.fill_between(steps, toPlot_av1 - toPlot_std1, toPlot_av1 + toPlot_std1,
    #                  alpha=0.5, edgecolor='#FF9933', facecolor='#FFB266')
    plt.plot(steps, toPlot_av1, label="label", c='#CC0000')
    plt.fill_between(steps, toPlot_av1 - toPlot_std1, toPlot_av1 + toPlot_std1,
                     alpha=0.5, edgecolor='#FF3333', facecolor='#FF6666')
    plt.plot(steps, toPlot_av2, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av2 - toPlot_std2, toPlot_av2 + toPlot_std2,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    plt.legend(['group 1','group 2'],loc=4)

    plt.savefig(data_path+name_run+'_variability_'+str(ind), bbox_inches='tight')


def plot_all(data_path):
    fig = plt.figure(figsize=(20,13), frameon=False)
    run={}
    run['perfs']=[]
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial) < 3:
            print('Extracting: ', trial)
            filename = data_path + trial + '/progress.json'
            print(filename)
            steps, eval_rewards = extract_performances(filename)
            run['perfs'].append(eval_rewards)
            plt.xlabel('steps')
            plt.ylabel('performance')
            plt.title(name_algo)
            plt.plot(steps,eval_rewards)

    plt.savefig(data_path+'all_'+name_run+'_runs', bbox_inches='tight')
    n_runs = len(run['perfs'])
    max_steps = 0
    for i in range(n_runs):
        if len(run['perfs'][i])>max_steps:
            max_steps = len(run['perfs'][i])
    eval_perfs = np.empty([n_runs, max_steps])*(np.nan)
    for i in range(n_runs):
        eval_perfs[i,:len(run['perfs'][i])] = run['perfs'][i]

    steps = range(0,max_steps*2000,2000)

    toPlot_av = np.nanmean(eval_perfs,axis=0)
    toPlot_std = np.nanstd(eval_perfs,axis=0)
    fig = plt.figure(figsize=(20,13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title(name_algo)
    plt.plot(steps, toPlot_av, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av - toPlot_std, toPlot_av + toPlot_std,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    #plt.legend()

    plt.savefig(data_path+name_run, bbox_inches='tight')

    plt.show()

plot_all(data_path)

# plot_vs(data_path)

# test variability
# for ind in range(10):
#     plot_test_variability(data_path,ind)



