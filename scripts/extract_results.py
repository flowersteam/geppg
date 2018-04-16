"""
This script is made for extracting results from runs.
Please refer to the paper GEP-PG: Decoupling Exploration and Exploitation in Deep Reinforcement Learning Algorithms.

Use results='compute_plot_all' to extract all results and plots for a given algorithm:
- Extract performance of GEP part if gep=True and plot its evolution across learning (each run on one figure, mean+error on another)
- Plot the learning curve of DDPG for each run (seed) and the average over different seeds (add GEP part if gep=True)
- Compute their scores according to two metrics (final: average of last 100 episodes performance, which corresponds to
the 10 last policies used by DDPG or absolute: average return of the best policy found across learning over 100 test episodes).

For each algorithm, 'compute_plot_all' should be called with data_path refering to a folder organised as such:
 data_path = Algo1|
                  |_ trial_id1 (folder containing output of a unique run)
                  |_ trial_id2
                  |_ ..

This creates three files
 - eval_performance_** : contains for each seed, the evolution of offline test performances
 - scores_absolute: absolute metric of performance for each seed
 - scores_final: final metric of performance for each seed
 - stats_scores: first row: mean, std, min, max of the final metric of performance over the different seeds.
                 second row: same for absolute metric of performance.

Once this is done, if you want to compare two algorithms: move the scores_final and scores_absolute files
corresponding to the two algorithms into a unique folder. Rank them in alphabetical order like this:
    1_scores_final_algo1, 2_scores_final_aglo2, 3_score_absolute_algo1, scores_absolute_algo2
Then, change the data_path to this folder, and run the script with results='compare_stats'.
This produces a file 'stats':
- first row: KS statistics, KS p-value, ttest statistic, ttest p-value for the final metric
- second row: KS statistics, KS p-value, ttest statistic, ttest p-value for the absolute metric
- third row: mean of the bootstrap 95% confidence interval for the difference, lower bound, higher bound, test significance. This for the final metric
- fourth row: mean of the bootstrap 95% confidence interval for the difference, lower bound, higher bound, test significance. This for the absolute metric

"""

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
import pickle
import os

data_path='../results/algo_folder'

name_run = 'GEP_PG'
name_algo = 'GEP-PG with action perturbations'
env_id='HalfCheetah-v2' #'MountainCarContinuous-v0'  #

gep = False # set to True if study includes GEP bootstrap 'GEP_PG', False otherwise
gep_only = False # set to True if it's a run 'GEP'
if 'GEP' in name_run:
    gep=True
if 'PG' not in name_run:
    gep_only = True


results ='compute_plot_all'# 'compare_stats'#

# choose representation of error percentile as integer, 'std', 'sem'
error_type = 'sem'
main_curve='mean'#  'median'#
x_axis_step = 2000 # number of steps between two data points. Here 2000 steps = 2 episodes
# give sample sizes for computing statistical tests between different algorithms
n1 = 20 #size sample 1
n2 = 20 #size sample 2

## for replaying best policy (compute absolute metric)
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

def extract_gep(data_path, x_step, main_curve, error_type):
    """
    Extract and plot GEP learning curves.
    """
    print('Extract GEP performances..')
    eval_perfs = []
    final_perfs = []
    absolute_perfs = []
    fig = plt.figure(figsize=(20,13), frameon=False)
    for i, trial in enumerate(sorted(os.listdir(data_path))):
        if len(trial)<5: #only select trial_ids, not files or plot created after that
            with open(data_path+trial+'/save_gep.pk', 'rb') as f:
                data_gep = pickle.load(f)
            eval_perfs.append(list(data_gep['eval_perfs']))
            absolute_perfs.append(np.array(data_gep['final_eval_perfs'].tolist()).mean())
            final_perfs.append(np.array(data_gep['eval_perfs'][-10:]).mean())
            plt.xlabel('steps')
            plt.ylabel('performance')
            plt.title(name_algo)
            steps = range(0, len(data_gep['eval_perfs']) * x_step, x_step)
            plt.plot(steps, eval_perfs[-1])
            del data_gep

    plt.savefig(data_path + 'all_geps', bbox_inches='tight')
    np.savetxt(data_path+'buffer_perfs', np.asarray(eval_perfs))
    np.savetxt(data_path+'scores_absolute_gep', np.array(absolute_perfs))
    np.savetxt(data_path+'scores_final_gep', np.array(final_perfs))

    print('Mean GEP performance: ', np.mean(np.array(final_perfs)))
    n_runs = len(eval_perfs)
    eval_perfs = np.array(eval_perfs)
    steps = range(0, eval_perfs.shape[1]*x_step, x_step)
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

    plt.savefig(data_path+'GEP_plot.png', bbox_inches='tight')
    print('Done.')


def extract_performances(filename, x_step):
    """
    Extract DDPG performance from output files.
    """
    with open(filename) as json_data:
        lines = json_data.readlines()
    eval_rewards = [0]
    steps = [0]
    score_final=0
    for line in lines:
        episode_data = json.loads(line)
        if 'eval/return' in episode_data:
            step = episode_data['total/epochs']*x_step
            perf = episode_data['eval/return']
            eval_rewards.append(perf)
            steps.append(step)
    return steps, eval_rewards
#


def plot_all(data_path, x_step, error_type, main_curve, gep):
    """
    Plot DDPG learning curves, and GEP-PG learning curves if gep=True
    """
    print('Plotting learning curves..')
    fig = plt.figure(figsize=(20,13), frameon=False)
    run={}
    run['perfs']=[]

    if gep:
        # load gep performances
        gep_perfs = np.loadtxt(data_path+'buffer_perfs')
        size_buff = gep_perfs.shape[1]

    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial)<5:
            # print('Extracting: ',trial)
            filename = data_path + trial + '/progress.json'
            # print(filename)
            steps, eval_rewards= extract_performances(filename, x_step)
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
    if gep:
        max_steps += size_buff

    eval_perfs = np.zeros([n_runs, max_steps])
    eval_perfs.fill(np.nan)
    for i in range(n_runs):
        eval_perfs[i, :len(run['perfs'][i])] = run['perfs'][i]

    steps = range(0,max_steps*x_step,x_step)

    if gep:
        # shift DDPG curve and squeeze gep curve first.
        for i in range(n_runs):
            eval_perfs[i, gep_perfs.shape[1]+1:] = eval_perfs[i,:max_steps-gep_perfs.shape[1]-1]
            eval_perfs[i, 0] = 0
            eval_perfs[i, 1:gep_perfs.shape[1]+1] = gep_perfs[i]

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
        plt.axvline(x=gep_perfs.shape[1]*x_step, linestyle = '--', color='k')

    plt.savefig(data_path+name_run+'_learning_curve.png', bbox_inches='tight')
    np.savetxt(data_path+'eval_performances'+name_run, eval_perfs)
    print('Done.')



def compute_scores_all(data_path, x_step, name_algo,  gep):
    """
    Compute final and absolute metrics of performance for each run. For the absolute metric, the best policy found during
    DDPG training is replayed 100 times in the environment and the performance is the average return.
    """
    print('Compute metrics of performance, (this might take ~5 min)..')
    fig = plt.figure(figsize=(20, 13), frameon=False)
    hist_final_scores = []
    hist_absolute_scores = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):
        if len(trial) < 5:
            if os.path.exists(data_path + trial + '/scores'):
                tmp_scores = np.loadtxt(data_path + trial + '/scores')
                hist_final_scores.append(tmp_scores[0,0])
                hist_absolute_scores.append(tmp_scores[1,0])
            else:
                toSave = np.zeros([2,4])
                print('     Computing score of run #', trial)
                filename = data_path + trial + '/progress.json'
                # print(filename)
                steps, eval_rewards = extract_performances(filename, x_step)
                actor_folder = data_path + trial + '/tf_save/'

                # compute final metric
                toSave[0, 0] = np.array(eval_rewards[-10:]).mean()
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
                            # print(actor_folder+f)

                            #replace string in checkpoint file
                            with open(actor_folder+'/checkpoint', 'r') as checkf:
                                s=checkf.read()

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
                hist_final_scores.append(toSave[0,0])
                hist_absolute_scores.append(toSave[1,0])
    print('Done.')



def compare_stats(data_path, n1, n2):
    """
    Computes statistical tests to assess whether two algorithms perform statistically differently.
    data_path should include the scores_absolute and scores_final of the two algorithms as such:
    1_scores_final_algo1, 2_scores_final_aglo2, 3_score_absolute_algo1, scores_absolute_algo2
    These files are created by the ''compute_plot_all' function.
    We compute Kolmogorov Smirnov test, the ttest, and a bootstrap ocnfidence interval of the difference in performance
    for the absolute and final metrics.
    """
    print('Running statistical tests..')
    eval_perfs = np.empty([n1+n2,1001])*np.nan # 1001 is the length of an episode
    scores_absolute = []
    scores_final = []
    for i, f in enumerate(sorted(os.listdir(data_path))):
        # print(f)
        if 'absolute' in f:
            scores = np.loadtxt(data_path+f)
            for j in range(scores.shape[0]):
                scores_absolute.append(scores[j])
        if 'final' in f:
            scores = np.loadtxt(data_path+f)
            for j in range(scores.shape[0]):
                scores_final.append(scores[j])

    data1_absolute = np.array(scores_absolute[:n1])
    data2_absolute = np.array(scores_absolute[n1:])
    data1_final = np.array(scores_final[:n1])
    data2_final = np.array(scores_final[n1:])

    ks_final, p_ks_final = ks_2samp(data1_final,data2_final)
    ttest_final, p_ttest_final = ttest_ind(data1_final,data2_final, equal_var=False)

    ks_absolute, p_ks_absolute = ks_2samp(data1_absolute, data2_absolute)
    ttest_absolute, p_ttest_absolute = ttest_ind(data1_absolute, data2_absolute, equal_var=False)

    # estimation of confidence intervals with bootstrap method, https://github.com/facebookincubator/bootstrapped
    res_final = bs.bootstrap_ab(data1_final, data2_final, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_final = np.sign(res_final.upper_bound)==np.sign(res_final.lower_bound)
    res_absolute = bs.bootstrap_ab(data1_absolute, data2_absolute, bs_stats.mean,bs_compare.difference, num_iterations=10000)
    sign_absolute = np.sign(res_absolute.upper_bound) == np.sign(res_absolute.lower_bound)

    toSave=np.zeros([4,4])
    toSave[0:2,:] = np.array([[ks_final, p_ks_final, ttest_final, p_ttest_final],[ks_absolute, p_ks_absolute, ttest_absolute, p_ttest_absolute]])
    toSave[2,:] = np.array([res_final.value,res_final.lower_bound,res_final.upper_bound,sign_final*np.sign(res_final.lower_bound)])
    toSave[3,:] = np.array([res_absolute.value,res_absolute.lower_bound,res_absolute.upper_bound,sign_absolute*np.sign(res_absolute.lower_bound)])

    np.savetxt(data_path + 'stats', toSave)
    print('Done.')

def mean_scores(data_path):
    """
    Gather all performance metrics and save them at the algorithm folder level.
    Three files are created: stats_scores, scores_absolute, scores_final.
    """
    scores_absolute = []
    scores_final = []
    for i, trial in enumerate(sorted(os.listdir(data_path))):

        if len(trial) < 5:
            scores = np.loadtxt(data_path + trial + '/scores')
            scores_absolute.append(scores[1, 0])
            scores_final.append(scores[0, 0])

    scores_absolute = np.array(scores_absolute)
    scores_final = np.array(scores_final)

    toSave = np.array([[scores_absolute.mean(), scores_absolute.std(), scores_absolute.min(), scores_absolute.max()],\
                      [scores_final.mean(), scores_final.std(), scores_final.min(), scores_final.max()]])
    np.savetxt(data_path+'stat_scores', toSave)
    np.savetxt(data_path+'scores_absolute', scores_absolute)
    np.savetxt(data_path+'scores_final', scores_final)


if 'compute_plot_all' in results:
    if gep:
        extract_gep(data_path, x_axis_step, main_curve, error_type)
    if not gep_only:
        plot_all(data_path, x_axis_step, error_type, main_curve, gep)
        compute_scores_all(data_path, x_axis_step, name_algo, gep=gep)
        mean_scores(data_path)

if 'compare_stats' in results:
    compare_stats(data_path, n1, n2)




