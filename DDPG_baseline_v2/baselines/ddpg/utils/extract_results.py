import json
import os
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf
from scipy.stats import ks_2samp, ttest_ind, pearsonr
# https://github.com/facebookincubator/bootstrapped
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats


data_path='/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/GEP-FPG/900k/'
gep=False

name_run = 'GEP_FDDPG'
name_algo = 'DDPG noisy actions'
env_id='HalfCheetah-v1'

results = 'compute_scores'#  'plot_all'# 'mean_scores'#   'correlation_gep_ddpg'# 'compute_stats_from_eval'#    'compute_stats_vs' #   'plot_gep'#  'test_variability'#  'plot_vs'#       'compute_scores'#     'correlation_gep_ddpg'#
error_type = 'sem' # 90#  100# 'std'#  'sem'#   80#  'std'
main_curve='mean'#  'median'#
n1 = 20 #size sample 1
n2 = 20 #size sample 2



## stuff for replaying actor
# initialize environment
env = gym.make(env_id)

# extract space
action_space = np.array([env.action_space.low, env.action_space.high]).T
observation_space = np.array([env.observation_space.low, env.observation_space.high]).T
if len(np.array(env.reward_range).shape)==1:
    reward_space = np.array(env.reward_range).reshape(1,-1)
else:
    reward_space = np.array(env.reward_range)

reward_dim = reward_space.shape[0]
observation_dim = observation_space.shape[0]
action_dim = action_space.shape[0]
state_size = observation_dim

def extract_performances(filename):

    with open(filename) as json_data:
        lines = json_data.readlines()
    eval_rewards = [0]
    steps = [0]
    score_litt=0
    for line in lines:
        episode_data = json.loads(line)
        if 'eval/return' in episode_data:
            step = episode_data['total/epochs']*2000
            perf = episode_data['eval/return']
            eval_rewards.append(perf)
            steps.append(step)
    return steps, eval_rewards


def plot_vs(data_path, n1, n2, error_type, main_curve, gep):
    run = {}
    run['perfs'] = []
    scores_our = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial)<4:
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
    # np.random.shuffle(inds)
    print(inds)
    assert n1 + n2 == n_runs

    if gep:
        # modify GEP source
        for i in range(n1):#,n_runs):
            eval_perfs[i,251:] = eval_perfs[i,:750]
            if os.path.exists(data_path + 'buffer_perfs'):
                gep_perfs = np.loadtxt(data_path+'buffer_perfs')
                eval_perfs[i,0] = 0
                eval_perfs[i,1:251] = gep_perfs[i]
            else:
                eval_perfs[i, :251] = np.zeros([251])
        eval_perfs_gep = np.loadtxt('/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/finals/GEP_2M_LinearPolicy/eval_performances_GEP2M_linear_policy')
        eval_perfs_gep=np.concatenate([np.zeros([20,1]),eval_perfs_gep], axis=1)

    if main_curve=='mean':
        toPlot_av1 = np.nanmean(eval_perfs[:n1], axis=0)
        toPlot_av2 = np.nanmean(eval_perfs[inds[n1:]], axis=0)
        if gep:
            toPlot_av3 = np.nanmean(eval_perfs_gep,axis=0)
    elif main_curve=='median':
        toPlot_av1 = np.nanmedian(eval_perfs[:n1],axis=0)
        toPlot_av2 = np.nanmedian(eval_perfs[inds[n1:]],axis=0)
        if gep:
            toPlot_av3 = np.nanmedian(eval_perfs_gep,axis=0)
    else:
        raise NotImplementedError

    if error_type == 'std':
        toPlot_error1_add = np.nanstd(eval_perfs[:n1], axis=0)
        toPlot_error1_sub = np.nanstd(eval_perfs[:n1], axis=0)
        toPlot_error2_add = np.nanstd(eval_perfs[n1:], axis=0)
        toPlot_error2_sub = np.nanstd(eval_perfs[n1:], axis=0)
        if gep:
            toPlot_error3_sub = np.nanstd(eval_perfs_gep,axis=0)
    elif error_type == 'sem':
        toPlot_error1_add = np.nanstd(eval_perfs[:n1], axis=0) / np.sqrt(n1)
        toPlot_error1_sub = np.nanstd(eval_perfs[:n1], axis=0) / np.sqrt(n1)
        toPlot_error2_add = np.nanstd(eval_perfs[n1:], axis=0) / np.sqrt(n2)
        toPlot_error2_sub = np.nanstd(eval_perfs[n1:], axis=0) / np.sqrt(n2)
        if gep:
            toPlot_error3_sub = np.nanstd(eval_perfs_gep,axis=0) / np.sqrt(20)
            toPlot_error3_add = np.nanstd(eval_perfs_gep,axis=0) / np.sqrt(20)

    elif type(error_type) == int:
        assert (error_type<=100 and error_type>0), 'error_type must be between 0 and 100'
        toPlot_error1_add = np.nanpercentile(eval_perfs[:n1],int(100-(100-error_type)/2), axis=0)-toPlot_av1
        toPlot_error1_sub = -np.nanpercentile(eval_perfs[:n1],int((100-error_type)/2), axis=0)+toPlot_av1
        toPlot_error2_add = np.nanpercentile(eval_perfs[n1:],int(100-(100-error_type)/2), axis=0)-toPlot_av2
        toPlot_error2_sub = -np.nanpercentile(eval_perfs[n1:], int((100-error_type)/2), axis=0)+toPlot_av2
        if gep:
            toPlot_error3_add = np.nanpercentile(eval_perfs_gep,int(100-(100-error_type)/2),axis=0)-toPlot_av3
            toPlot_error3_sub = np.nanpercentile(eval_perfs_gep,int((100-error_type)/2),axis=0)+toPlot_av3

    else:
        raise NotImplementedError
    if type(error_type) == int:
        error_type = str(error_type)+'%'
    fig = plt.figure(figsize=(20, 13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title("DDPG (noisy parameters), GEP-DDPG (noisy parameters) and GEP")

    plt.plot(steps, toPlot_av1, label="label", c='#CC0000')
    plt.fill_between(steps, toPlot_av1 - toPlot_error1_sub, toPlot_av1 + toPlot_error1_add,
                     alpha=0.5, edgecolor='#FF3333', facecolor='#FF6666')
    plt.plot(steps, toPlot_av2, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av2 - toPlot_error2_sub, toPlot_av2 + toPlot_error2_add,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    if gep:
        plt.plot(steps, toPlot_av3, label="label", c=(0.929,0.694,0.125))
        plt.fill_between(steps, toPlot_av3 - toPlot_error3_sub, toPlot_av3 + toPlot_error3_add,
                         alpha=0.5, edgecolor=(1,0.75,0.25), facecolor=(1,0.8,0.35))
        plt.axvline(x=500000, linestyle = '--', color='k')

    legend = ['GEP-DDPG (noisy parameters), '+main_curve+' ('+error_type+')','DDPG (noisy parameters), '+main_curve+' ('+error_type+')']
    if gep:
       legend.append('GEP, '+main_curve+' ('+error_type+')')
    plt.legend(legend,loc=4)

    plt.savefig(data_path+'DDPG_vs_GEP_DDPG_Plappert_vs_GEP'+main_curve+' ('+error_type+').png', bbox_inches='tight')


def compute_stats_vs(data_path, n1, n2, gep, save=True, var=False):
    run = {}
    run['perfs'] = []
    scores_our = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial)<4:
            # print('Extracting: ', trial)
            filename = data_path + trial + '/progress.json'
            # print(filename)
            steps, eval_rewards = extract_performances(filename)
            run['perfs'].append(eval_rewards)
            scores = np.loadtxt(data_path+trial+'/scores')
            scores_our.append(scores[1,0])

    n_runs = len(run['perfs'])
    assert n1+n2==n_runs
    max_steps = 0
    for i in range(n_runs):
        if len(run['perfs'][i])>max_steps:
            max_steps = len(run['perfs'][i])
    eval_perfs = np.empty([n_runs, max_steps])*(np.nan)
    for i in range(n_runs):
        eval_perfs[i,:len(run['perfs'][i])] = run['perfs'][i]

    # steps = steps[:700]
    inds = np.array(range(n_runs))
    if var:
        np.random.shuffle(inds)

    if gep:
        # modify GEP source
        for i in range(n1):#,n_runs):
            eval_perfs[i,251:] = eval_perfs[i,:750]
            eval_perfs[i,:251] = np.zeros([251])

    # print(inds)

    # compute statistics
    data1_litt = np.nanmean(eval_perfs[inds[:n1]][:,-10:],axis=1)
    data2_litt = np.nanmean(eval_perfs[inds[n1:]][:,-10:],axis=1)
    ks_litt, p_ks_litt = ks_2samp(data1_litt,data2_litt)
    ttest_litt, p_ttest_litt = ttest_ind(data1_litt,data2_litt, equal_var=False)
    data1_our = np.array(scores_our[:n1])
    data2_our = np.array(scores_our[n1:])
    ks_our, p_ks_our = ks_2samp(data1_our, data2_our)
    ttest_our, p_ttest_our = ttest_ind(data1_our, data2_our, equal_var=False)

    # estimation of confidence intervals with bootstrap method, https://github.com/facebookincubator/bootstrapped
    res_litt = bs.bootstrap_ab(data1_litt, data2_litt, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_litt = np.sign(res_litt.upper_bound)==np.sign(res_litt.lower_bound)
    res_our = bs.bootstrap_ab(data1_our, data2_our, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_our = np.sign(res_our.upper_bound) == np.sign(res_our.lower_bound)

    toSave=np.zeros([4,4])
    toSave[0:2,:] = np.array([[ks_litt, p_ks_litt, ttest_litt, p_ttest_litt],[ks_our, p_ks_our, ttest_our, p_ttest_our]])
    toSave[2,:] = np.array([res_litt.value,res_litt.lower_bound,res_litt.upper_bound,sign_litt*np.sign(res_litt.lower_bound)])
    toSave[3,:] = np.array([res_our.value,res_our.lower_bound,res_our.upper_bound,sign_our*np.sign(res_our.lower_bound)])
    if save:
        np.savetxt(data_path+'stats',toSave)

    if var:
        toReturn = np.array([toSave[0, 1], toSave[0, 3], toSave[1, 1], toSave[1, 3], toSave[2, 3], toSave[3, 3]])
        return toReturn


def plot_test_variability(data_path, ind, error_type, main_curve):
    run = {}
    run['perfs'] = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial)<4:
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

    if main_curve=='mean':
        toPlot_av1 = np.nanmean(eval_perfs[inds[:int(n_runs/2)]], axis=0)
        toPlot_av2 = np.nanmean(eval_perfs[inds[int(n_runs / 2):]], axis=0)
    elif main_curve=='median':
        toPlot_av1 = np.nanmedian(eval_perfs[inds[:int(n_runs/2)]],axis=0)
        toPlot_av2 = np.nanmedian(eval_perfs[inds[int(n_runs / 2):]],axis=0)
    else:
        raise NotImplementedError

    if error_type == 'std':
        toPlot_error1_add = np.nanstd(eval_perfs[inds[:int(n_runs/2)]], axis=0)
        toPlot_error1_sub = np.nanstd(eval_perfs[inds[:int(n_runs/2)]], axis=0)
        toPlot_error2_add = np.nanstd(eval_perfs[inds[int(n_runs / 2):]], axis=0)
        toPlot_error2_sub = np.nanstd(eval_perfs[inds[int(n_runs / 2):]], axis=0)
    elif error_type == 'sem':
        toPlot_error1_add = np.nanstd(eval_perfs[inds[:int(n_runs/2)]],axis=0)/np.sqrt(int(n_runs/2))
        toPlot_error1_sub = np.nanstd(eval_perfs[inds[:int(n_runs/2)]], axis=0) / np.sqrt(int(n_runs/2))
        toPlot_error2_add = np.nanstd(eval_perfs[inds[int(n_runs / 2):]], axis=0) / np.sqrt(int(n_runs/2))
        toPlot_error2_sub = np.nanstd(eval_perfs[inds[int(n_runs / 2):]], axis=0) / np.sqrt(int(n_runs/2))
    elif type(error_type) == int:
        assert (error_type<=100 and error_type>0), 'error_type must be between 0 and 100'
        toPlot_error1_add = np.nanpercentile(eval_perfs[inds[:int(n_runs/2)]],int(100-(100-error_type)/2), axis=0)-toPlot_av1
        toPlot_error1_sub = -np.nanpercentile(eval_perfs[inds[:int(n_runs/2)]],int((100-error_type)/2), axis=0)+toPlot_av1
        toPlot_error2_add = np.nanpercentile(eval_perfs[inds[int(n_runs / 2):]], int(100 - (100 - error_type) / 2), axis=0)-toPlot_av2
        toPlot_error2_sub = -np.nanpercentile(eval_perfs[inds[int(n_runs / 2):]], int((100 - error_type) / 2), axis=0)+toPlot_av2
    else:
        raise NotImplementedError
    if type(error_type) == int:
        error_type = str(error_type)+'%'

    fig = plt.figure(figsize=(20, 13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title(name_algo+', '+ error_type)
    # plt.plot(steps, toPlot_av1, label="label", c='#CC6600')
    # plt.fill_between(steps, toPlot_av1 - toPlot_std1, toPlot_av1 + toPlot_std1,
    #                  alpha=0.5, edgecolor='#FF9933', facecolor='#FFB266')
    plt.plot(steps, toPlot_av1, label="label", c='#CC0000')
    plt.fill_between(steps, toPlot_av1 - toPlot_error1_sub, toPlot_av1 + toPlot_error1_add,
                     alpha=0.5, edgecolor='#FF3333', facecolor='#FF6666')
    plt.plot(steps, toPlot_av2, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av2 - toPlot_error2_sub, toPlot_av2 + toPlot_error2_add,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    plt.legend(['group 1','group 2'],loc=4)

    plt.savefig(data_path+name_run+'_variability_'+str(ind), bbox_inches='tight')


def plot_all(data_path, error_type, main_curve, gep):
    fig = plt.figure(figsize=(20,13), frameon=False)
    run={}
    run['perfs']=[]

    if gep:
        gep_perfs = np.loadtxt(data_path+'buffer_perfs')

    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial)<4:
            print('Extracting: ',trial)
            filename = data_path + trial + '/progress.json'
            print(filename)
            steps, eval_rewards= extract_performances(filename)
            run['perfs'].append(eval_rewards)
            plt.xlabel('steps')
            plt.ylabel('performance')
            plt.title(name_algo)
            plt.plot(steps,eval_rewards)

    plt.savefig(data_path+'all_'+name_run+'_runs', bbox_inches='tight')
    n_runs = len(run['perfs'])
    max_steps = 0
    for i in range(n_runs):
        if len(run['perfs'][i]) > max_steps:
            max_steps = len(run['perfs'][i])
    max_steps = 1001
    eval_perfs = np.empty([n_runs, max_steps])*(np.nan)
    for i in range(n_runs):
        eval_perfs[i, :len(run['perfs'][i])] = run['perfs'][i]

    steps = range(0,max_steps*2000,2000)

    if gep:
        for i in range(n_runs):
            eval_perfs[i, 251:] = eval_perfs[i,:750]
            eval_perfs[i, 0] = 0
            eval_perfs[i, 1:251] = gep_perfs[i]

    if main_curve=='mean':
        toPlot_av = np.nanmean(eval_perfs, axis=0)
    elif main_curve=='median':
        toPlot_av = np.nanmedian(eval_perfs,axis=0)
    else:
        raise NotImplementedError

    if error_type == 'std':
        toPlot_error_add = np.nanstd(eval_perfs, axis=0)
        toPlot_error_sub = np.nanstd(eval_perfs, axis=0)
    elif error_type == 'sem':
        toPlot_error_add = np.nanstd(eval_perfs,axis=0)/np.sqrt(n_runs)
        toPlot_error_sub = np.nanstd(eval_perfs, axis=0) / np.sqrt(n_runs)
    elif type(error_type) == int:
        assert (error_type<=100 and error_type>0), 'error_type must be between 0 and 100'
        toPlot_error_add = np.nanpercentile(eval_perfs,int(100-(100-error_type)/2), axis=0)-toPlot_av
        toPlot_error_sub = -np.nanpercentile(eval_perfs,int((100-error_type)/2), axis=0)+toPlot_av
    else:
        raise NotImplementedError

    if type(error_type) == int:
        error_type = str(error_type)+'%'
    fig = plt.figure(figsize=(20,13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title(name_algo+' '+main_curve+' ('+error_type+ ')')
    plt.plot(steps, toPlot_av, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av - toPlot_error_sub, toPlot_av + toPlot_error_add,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    #plt.legend()
    if gep:
        plt.axvline(x=500000, linestyle = '--', color='k')
    plt.savefig(data_path+name_run+' '+main_curve+' ('+error_type+ ')', bbox_inches='tight')
    np.savetxt(data_path+'eval_performances'+name_run, eval_perfs)




def compute_scores_all(data_path, name_algo,  gep):
    fig = plt.figure(figsize=(20, 13), frameon=False)
    hist_litt_scores = []
    hist_our_scores = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):
        if len(trial) < 4:
            if os.path.exists(data_path + trial + '/scores'):
                tmp_scores = np.loadtxt(data_path + trial + '/scores')
                hist_litt_scores.append(tmp_scores[0,0])
                hist_our_scores.append(tmp_scores[1,0])
            else:
                # actor_folder = data_path + trial + '/tf_save/' + 'best1_5M/'
                # for f in os.listdir(actor_folder):
                #     os.rename(actor_folder+f, actor_folder+f[:-2])
                toSave = np.zeros([2,4])
                print('Computing score: ', trial)
                filename = data_path + trial + '/progress.json'
                print(filename)
                steps, eval_rewards = extract_performances(filename)
                if gep:
                    steps=steps[:750]
                    eval_rewards=eval_rewards[:750]
                    actor_folder = data_path + trial + '/tf_save/'+'best1_5M/'
                else:
                    actor_folder = data_path + trial + '/tf_save/'

                toSave[0,0] = np.array(eval_rewards[-10:]).mean()
                toSave[0, 1] = np.array(eval_rewards[-10:]).std()
                toSave[0, 2] = np.array(eval_rewards[-10:]).min()
                toSave[0, 3] = np.array(eval_rewards[-10:]).max()

                for f in os.listdir(actor_folder):
                    if 'meta' in f:
                        n_tests=100
                        score = np.zeros([n_tests])

                        with tf.Session() as sess:
                            init = tf.global_variables_initializer()
                            sess.run(init)
                            saver = tf.train.import_meta_graph(actor_folder+f)
                            print(actor_folder+f)

                            #replace string in checkpoint file
                            with open(actor_folder+'/checkpoint', 'r') as checkf:
                                s=checkf.read()
                            if 'projets' in s:
                                old_s = '/projets/flowers/cedric/ddpg_baseline_openAI_fork/results/HalfCheetah-v1/'
                                s=s.replace(old_s,data_path)
                                if gep:
                                    old_s = '/tf_save'
                                    s=s.replace(old_s, '/tf_save/'+'best1_5M')
                                with open(actor_folder+'/checkpoint', 'w') as checkf:
                                    checkf.write(s)

                            saver.restore(sess, tf.train.latest_checkpoint(actor_folder))
                            graph = tf.get_default_graph()
                            obs0 = graph.get_tensor_by_name("obs0:0")
                            actor_tf = graph.get_tensor_by_name("actor/Tanh:0")
                            for i in range(n_tests):
                                done = False
                                observations = [env.reset()]
                                actions = []
                                rewards = []
                                while not done:
                                    last_obs = observations[-1].squeeze()
                                    feed_dict = {obs0: [last_obs]}
                                    action = sess.run(actor_tf, feed_dict=feed_dict)
                                    actions.append(action)
                                    # env.render()
                                    out = env.step(actions[-1])
                                    observations.append(out[0])
                                    rewards.append(out[1])
                                    done = out[2]
                                score[i] = sum(rewards)
                        toSave[1, 0] = score.mean()
                        toSave[1, 1] = score.std()
                        toSave[1, 2] = score.min()
                        toSave[1, 3] = score.max()
                        break
                np.savetxt(data_path+trial+'/scores', toSave)
                hist_litt_scores.append(toSave[0,0])
                hist_our_scores.append(toSave[1,0])
    plt.figure()
    plt.hist(hist_litt_scores)
    plt.xlabel('performance returns')
    plt.title('Histogram of final performances for '+name_algo)
    plt.savefig(data_path+ name_algo+'_histogram_litt_measure', bbox_inches='tight')
    plt.figure()
    plt.hist(hist_our_scores)
    plt.xlabel('performance returns')
    plt.title('Histogram of final performances for '+name_algo)
    plt.savefig(data_path + name_algo+'_histogram_our_measure', bbox_inches='tight')

def correlation_gep_ddpg(data_path):
    litt_scores = []
    our_scores = []
    list_dir = list(range(101,121))+list(range(181,201))+list(range(401,421))
    gep_scores=np.loadtxt(data_path+'gep_scores')
    for i in range(len(list_dir)):
        trial=list_dir[i]
        print('Extracting: ', trial)
        array = np.loadtxt(data_path + str(trial) + '/scores')
        litt_scores.append(array[0,0])
        our_scores.append(array[1,0])
    our_r, our_p = pearsonr(np.array(our_scores), np.array(gep_scores))
    litt_r, litt_p= pearsonr(np.array(litt_scores), np.array(gep_scores))
    toSave = np.array([[our_r, our_p],[litt_r, litt_p]])
    np.savetxt(data_path+'correlations', toSave)

def plot_gep(data_path, error_type, main_curve, n1):
    max_steps = 1000
    eval_perfs = np.empty([20, 1000]) * np.nan
    steps = range(0,max_steps*2000,2000)
    ind = 0
    fig = plt.figure(figsize=(20, 13), frameon=False)
    for i, f in enumerate(os.listdir(data_path)):
        if 'perfsToPlot' in f:
            print(f)
            name = data_path+f
            perfs = np.loadtxt(name)
            eval_perfs[ind,:] = np.loadtxt(name)
            plt.xlabel('steps')
            plt.ylabel('performance')
            plt.title(name_algo)
            plt.plot(steps, eval_perfs[ind,:])
            ind+=1


    plt.savefig(data_path + 'all_' + name_run + '_runs', bbox_inches='tight')
    n_runs = eval_perfs.shape[0]
    print('n_runs :', n_runs)
    if main_curve=='mean':
        toPlot_av = np.nanmean(eval_perfs, axis=0)
    elif main_curve=='median':
        toPlot_av = np.nanmedian(eval_perfs,axis=0)
    else:
        raise NotImplementedError

    if error_type == 'std':
        toPlot_error_add = np.nanstd(eval_perfs, axis=0)
        toPlot_error_sub = np.nanstd(eval_perfs, axis=0)
    elif error_type == 'sem':
        toPlot_error_add = np.nanstd(eval_perfs,axis=0)/np.sqrt(n_runs)
        toPlot_error_sub = np.nanstd(eval_perfs, axis=0) / np.sqrt(n_runs)
    elif type(error_type) == int:
        assert (error_type<=100 and error_type>0), 'error_type must be between 0 and 100'
        toPlot_error_add = np.nanpercentile(eval_perfs,int(100-(100-error_type)/2), axis=0)-toPlot_av
        toPlot_error_sub = -np.nanpercentile(eval_perfs,int((100-error_type)/2), axis=0)+toPlot_av
    else:
        raise NotImplementedError

    if type(error_type) == int:
        error_type = str(error_type)+'%'
    fig = plt.figure(figsize=(20,13), frameon=False)
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title(name_algo+' '+main_curve+' ('+error_type+ ')')
    plt.plot(steps, toPlot_av, label="label", c='#0066CC')
    plt.fill_between(steps, toPlot_av - toPlot_error_sub, toPlot_av + toPlot_error_add,
                     alpha=0.5, edgecolor='#3999FF', facecolor='#66B2FF')
    #plt.legend()

    plt.savefig(data_path+'GEP_2M'+' '+main_curve+' ('+error_type+ ')', bbox_inches='tight')
    np.savetxt(data_path+'eval_performances_GEP2M_linear_policy', eval_perfs)

def compute_stat_from_eval_perfs(data_path,n1,n2):
    eval_perfs = np.empty([n1+n2,1001])*np.nan
    scores_our = []
    for i, f in enumerate(sorted(os.listdir(data_path))):
        print(f)
        if 'our' in f:
            scores = np.loadtxt(data_path+f)
            for j in range(scores.shape[0]):
                scores_our.append(scores[j])
        elif 'eval' in f:
            if '2M' in f:
                tmp = np.concatenate([np.zeros([20,1]),np.loadtxt(data_path+f)], axis=1)
            else:
                tmp = np.loadtxt(data_path+f)
            eval_perfs[i*20:(i+1)*n1,:] = tmp

    # compute statistics
    data1_litt = np.nanmean(eval_perfs[:n1][:,-10:],axis=1)
    data2_litt = np.nanmean(eval_perfs[n1:][:,-10:],axis=1)
    # data1_litt = np.nanmean(eval_perfs[:n1][:,100-10:100],axis=1)
    # data2_litt = np.nanmean(eval_perfs[n1:][:,100-10:100],axis=1)
    ks_litt, p_ks_litt = ks_2samp(data1_litt,data2_litt)
    ttest_litt, p_ttest_litt = ttest_ind(data1_litt,data2_litt, equal_var=False)
    data1_our = np.array(scores_our[:n1])
    data2_our = np.array(scores_our[n1:])
    ks_our, p_ks_our = ks_2samp(data1_our, data2_our)
    ttest_our, p_ttest_our = ttest_ind(data1_our, data2_our, equal_var=False)

    # estimation of confidence intervals with bootstrap method, https://github.com/facebookincubator/bootstrapped
    res_litt = bs.bootstrap_ab(data1_litt, data2_litt, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_litt = np.sign(res_litt.upper_bound)==np.sign(res_litt.lower_bound)
    res_our = bs.bootstrap_ab(data1_our, data2_our, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_our = np.sign(res_our.upper_bound) == np.sign(res_our.lower_bound)

    toSave=np.zeros([4,4])
    toSave[0:2,:] = np.array([[ks_litt, p_ks_litt, ttest_litt, p_ttest_litt],[ks_our, p_ks_our, ttest_our, p_ttest_our]])
    toSave[2,:] = np.array([res_litt.value,res_litt.lower_bound,res_litt.upper_bound,sign_litt*np.sign(res_litt.lower_bound)])
    toSave[3,:] = np.array([res_our.value,res_our.lower_bound,res_our.upper_bound,sign_our*np.sign(res_our.lower_bound)])

    np.savetxt(data_path + 'stats', toSave)

def mean_scores(data_path):
    scores_our = []
    scores_litt = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial) < 4:
            scores = np.loadtxt(data_path + trial + '/scores')
            scores_our.append(scores[1, 0])
            scores_litt.append(scores[0, 0])

    scores_our = np.array(scores_our)
    scores_litt = np.array(scores_litt)

    toSave = np.array([[scores_our.mean(), scores_our.std(), scores_our.min(), scores_our.max()],\
                      [scores_litt.mean(), scores_litt.std(), scores_litt.min(), scores_litt.max()]])
    np.savetxt(data_path+'stat_scores', toSave)
    np.savetxt(data_path+'scores_our', scores_our)


if 'compute_scores' in results:
    compute_scores_all(data_path, name_algo, gep=gep)

if 'plot_all' in results:
    plot_all(data_path, error_type, main_curve, gep)

if 'compute_stats_vs' in results:
    compute_stats_vs(data_path, n1,n2,gep, var=False, save=True)

if 'plot_vs' in results:
    plot_vs(data_path,n1,n2, error_type, main_curve, gep)

if 'test_variability' in results:
    # for ind in range(20):
    #     plot_test_variability(data_path,ind, error_type, main_curve)
    n_tests = 1000
    stats = np.zeros([n_tests,6])
    for ind in range(n_tests):
        print(ind)
        stats[ind,:] = compute_stats_vs(data_path,n1,n2, gep, var=True, save=False)
    sig = np.zeros([6])
    for j in range(6):
        for i in range(n_tests):
            if j<4:
                if stats[i,j]<0.05:
                    sig[j]+=1
            else:
                if stats[i,j]>=0.5 or stats[i,j]<=-0.5:
                    sig[j]+=1
    np.savetxt(data_path+'stats_variability', sig/n_tests)

if 'correlation_gep_ddpg' in results:
    correlation_gep_ddpg(data_path)

if 'plot_gep' in results:
    plot_gep(data_path, error_type, main_curve, n1)

if 'compute_stats_from_eval' in results:
    compute_stat_from_eval_perfs(data_path,n1,n2)

if 'mean_scores' in results:
    mean_scores(data_path)



