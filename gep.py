import os
import sys
sys.path.append('./')
import numpy as np
import gym
import pickle
import argparse
from DDPG_baseline_v2.baselines.ddpg.main_config import run_ddpg
from DDPG_baseline_v2.baselines.ddpg.configs.config import ddpg_config
from controllers import NNController
from representers import CheetahRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *


saving_folder = './results/'
trial_id = 1
nb_runs = 1
env_id = 'HalfCheetah-v2' #'MountainCarContinuous-v0' #
study = 'DDPG' # 'GEP' # 'GEPPG' #
ddpg_noise = 'ou_0.3'# 'adaptive_param_0.2' #
nb_exploration = 500 # nb of episodes for gep exploration


def run_experiment(env_id, trial, noise_type, study, nb_exploration, saving_folder):

    # create data path
    data_path = create_data_path(saving_folder, env_id, trial_id)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # GEP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'GEP' in study:
        # get GEP config
        if env_id=='HalfCheetah-v2':
            nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer,\
            nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = cheetah_config()
        elif env_id=='MountainCarContinuous-v0':
            nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
            nb_rep, engineer_goal, goal_space, initial_space,  knn, noise, nb_weights = cmc_config()

        # overun some settings
        nb_explorations = nb_exploration
        nb_tests = 100
        offline_eval = (1e6, 10) #(x,y): y evaluation episodes every x (done offline)

        train_perfs = []
        eval_perfs = []
        final_eval_perfs = []

        # compute test indices:
        test_ind = range(int(offline_eval[0])-1, nb_explorations, int(offline_eval[0]))

        # define environment
        env = gym.make(env_id)
        nb_act = env.action_space.shape[0]
        nb_obs = env.observation_space.shape[0]
        nb_rew = 1
        action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
        observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
        reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)

        # bootstrap phase
        # # # # # # # # # # #
        for ep in range(nb_bootstrap):
            print('Bootstrap episode #', ep+1)
            # sample policy at random
            policy = np.random.random(nb_weights) * 2 - 1

            # play policy and update knn
            obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                        representer, knn)

            # save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))

            # offline tests
            if ep in test_ind:
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env,
                                    controller, eval_perfs)

        # exploration phase
        # # # # # # # # # # # #
        for ep in range(nb_bootstrap, nb_explorations):
            print('Random Goal episode #', ep+1)

            # random goal strategy
            policy = random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights)

            # play policy and update knn
            obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                        representer, knn)

            # save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))

            # offline tests
            if ep in test_ind:
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs)

        # final evaluation phase
        # # # # # # # # # # # # # # #
        for ep in range(nb_tests):
            print('Test episode #', ep+1)
            best_policy = offline_evaluations(1, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, final_eval_perfs)


        print('Final performance for the run: ', np.array(final_eval_perfs).mean())

        # wrap up and save
        # # # # # # # # # # #
        gep_memory = dict()
        gep_memory['actions'] = action_seqs.swapaxes(1, 2)
        gep_memory['observations'] = observation_seqs.swapaxes(1, 2)
        gep_memory['rewards'] = reward_seqs.swapaxes(1, 2)
        gep_memory['best_policy'] = best_policy
        gep_memory['train_perfs'] = np.array(train_perfs)
        gep_memory['eval_perfs'] = np.array(eval_perfs)
        gep_memory['final_eval_perfs'] = np.array(final_eval_perfs)
        gep_memory['representations'] = knn._X
        gep_memory['policies'] = knn._Y
        gep_memory['metrics'] = compute_metrics(gep_memory) # compute metrics for buffer analysis

        with open(data_path+'save_gep.pk', 'wb') as f:
            pickle.dump(gep_memory, f)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DDPG
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'PG' in study:



        # load ddpg config
        dict_args = ddpg_config(env_id=env_id,
                                study=study,
                                data_path=data_path,
                                noise=noise_type,
                                trial_id=trial_id,
                                seed=int(np.random.random() * 1e6),
                                nb_epochs=1000,
                                buffer_location=None,
                                gep_memory=None
                                )
        # provide GEP memory to DDPG to fill its replay buffer
        try:
            dict_args['gep_memory'] = gep_memory
        except:
            pass

        run_ddpg(dict_args)

    return np.array(final_eval_perfs).mean()

def play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller, representer, knn):
    """
    Play a policy in the environment for a given number of timesteps, usin a NN controller.
    Then represent the trajectory and update the inverse model.
    """
    obs = np.zeros([1, nb_obs, nb_timesteps + 1])
    act = np.zeros([1, nb_act, nb_timesteps])
    rew = np.zeros([1, nb_rew, nb_timesteps + 1])
    obs.fill(np.nan)
    act.fill(np.nan)
    rew.fill(np.nan)
    obs[0, :, 0] = env.reset()
    rew[0, :, 0] = 0
    done = False  # termination signal
    for t in range(nb_timesteps):
        if done:
            break
        act[0, :, t] = controller.step(policy, obs[0, :, t]).reshape(1, -1)
        out = env.step(np.copy(act[0, :, t]))
        # env.render()
        obs[0, :, t + 1] = out[0]
        rew[0, :, t + 1] = out[1]
        done = out[2]

    # convert the trajectory into a representation (=behavioral descriptor)
    rep = representer.represent(obs, act)

    # update inverse model
    knn.update(X=rep, Y=policy)

    return obs, act, rew

def offline_evaluations(nb_eps, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    best_policy = knn.predict(engineer_goal)[0, :]

    returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)
        obs = env.reset()
        rew[:, 0] = 0
        done = False
        for t in range(nb_timesteps):
            if done:
                break
            act = controller.step(best_policy, obs).reshape(1, -1)
            out = env.step(np.copy(act))
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
        returns.append(np.nansum(rew))
    eval_perfs.append(np.array(returns).mean())

    return best_policy


def random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights):
    """
    Draw a goal, find policy associated to its nearest neighbor in the representation space, add noise to it.
    """
    # draw goal in goal space
    goal = np.copy(sample(goal_space))
    # scale goal to [-1,1]^N
    goal = scale_vec(goal, initial_space)

    # find policy of nearest neighbor
    policy = knn.predict(goal)[0]

    # add exploration noise
    policy += np.random.normal(0, noise*2, nb_weights) # noise is scaled by space measure
    policy_out = np.clip(policy, -1, 1)

    return policy_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', type=int, default=trial_id)
    parser.add_argument('--env_id', type=str, default=env_id)
    parser.add_argument('--noise_type', type=str, default=ddpg_noise)  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
    parser.add_argument('--study', type=str, default=study) #'DDPG'  #'GEP_PG'
    parser.add_argument('--nb_exploration', type=int, default=nb_exploration)
    parser.add_argument('--saving_folder', type=str, default=saving_folder)
    args = vars(parser.parse_args())

    gep_perf = np.zeros([nb_runs])
    for i in range(nb_runs):
        gep_perf[i] = run_experiment(**args)
        print(gep_perf)
        print('Average performance: ', gep_perf.mean())






