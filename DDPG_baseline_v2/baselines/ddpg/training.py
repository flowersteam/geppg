import os
import pickle
import shutil
import time
from collections import deque
from os import path

import numpy as np
import DDPG_baseline_v2.baselines.common.tf_util as U
import tensorflow as tf
from mpi4py import MPI
from DDPG_baseline_v2.baselines import logger
from DDPG_baseline_v2.baselines.ddpg.ddpg import DDPG
from DDPG_baseline_v2.baselines.ddpg.memory import load_from_cedric, load_from_geppg

from DDPG_baseline_v2.baselines.ddpg.util import mpi_mean, mpi_std, mpi_sum


def run_agent(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, study,
    buffer_location, trial_id, data_path, nb_eval_episodes,  tau=0.01, eval_env=None, param_noise_adaption_interval=50,
    gep_memory=None, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    episode = 0
    best_score = -1000 # initialize to negative values to keep track of best score
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    saver_best = tf.train.Saver(max_to_keep=1)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        size_buffer = 0
        # load buffer if necessary
        if study == 'GEP_PG' or study == 'GEP_FPG':
            if study ==  'GEP_FPG':
                logger.info('This is a frozen buffer study')
            if gep_memory is not None:
                buffer_gep = load_from_geppg(gep_memory)
                del gep_memory
                logger.info('Load buffer from gep memory')
            else:
                assert buffer_location != '', 'study is ' + study + ', a buffer location should be provided.'
                # fill replay buffer
                buffer_gep = load_from_cedric(filename=buffer_location)
                logger.info('Load buffer from file')

            size_buffer = len(buffer_gep[0])
            for i in range(size_buffer):
                agent.store_transition(np.array(buffer_gep[0][i]), np.array(buffer_gep[1][i]),
                                       buffer_gep[2][i][0], np.array(buffer_gep[3][i]), buffer_gep[4][i])
            logger.info('Buffer of size ' + str(size_buffer) + ' has been loaded')

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = size_buffer # t statt at size buffer if a buffer is provided, 0 otherwise

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        max_steps = nb_rollout_steps*nb_epoch_cycles*nb_epochs

        for epoch in range(nb_epochs):
            if t>=max_steps:
                break
            for cycle in range(nb_epoch_cycles):
                if t>=max_steps:
                    break
                # Perform rollouts.
                if study != 'GEP_FPG':
                    agent, env, episode_reward, episode_step, epoch_actions, epoch_qs, epoch_episode_rewards, episode_rewards_history, \
                    epoch_episode_steps, epoch_episodes, episodes, t = rollout(nb_rollout_steps, agent, obs, rank, env, max_action,
                                                                               episode_reward, episode_step, epoch_actions,
                                                                               epoch_qs,epoch_episode_rewards, episode_rewards_history,
                                                                               epoch_episode_steps, epoch_episodes, episodes, render, t, epoch
                                                                               )
                # Train.
                agent, epoch_actor_losses, epoch_critic_losses, epoch_adaptive_distances = train(nb_train_steps, memory,
                                                                                                 batch_size, t,
                                                                                                 param_noise_adaption_interval, agent
                                                                                                 )
            # Evaluate.
            agent, eval_episode_rewards, eval_qs, best_score, eval_obs, eval_env, \
            eval_episode_rewards_history = evaluate(agent, eval_obs, max_action, eval_env, render_eval, sess, t, eval_episode_rewards_history,
                                                     best_score, data_path, saver_best, nb_eval_episodes, study, max_steps, size_buffer
                                                    )

            # update decreasing_ou if necessary
            try:
                action_noise.adapt()
                print(action_noise.sigma)
            except:
                pass

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Added statistics

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

            # Train statistics.
            try:
                combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
                combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
                combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)
            except:
                pass

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mpi_mean(eval_qs)
                combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)





def rollout(nb_rollout_steps, agent, obs, rank, env, max_action, episode_reward,
            episode_step, epoch_actions, epoch_qs,epoch_episode_rewards, episode_rewards_history,
            epoch_episode_steps, epoch_episodes, episodes, render, t, epoch):

    for t_rollout in range(nb_rollout_steps):
        # Predict next action.
        action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
        assert action.shape == env.action_space.shape

        # Execute next action.
        if rank == 0 and render:
            env.render()
        assert max_action.shape == action.shape
        new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
        t += 1
        if rank == 0 and render:
            env.render()
        episode_reward += r
        episode_step += 1
        #print ("rollout episode_step : ", episode_step, " t:", t)

        # Book-keeping.
        epoch_actions.append(action)
        epoch_qs.append(q)
        agent.store_transition(obs, action, r, new_obs, done)
        obs = new_obs

        if done:
            # Episode done.
            epoch_episode_rewards.append(episode_reward)
            episode_rewards_history.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            episode_reward = 0.
            episode_step = 0
            epoch_episodes += 1
            episodes += 1

            agent.reset()
            obs = env.reset()

    return agent, env, episode_reward, episode_step, epoch_actions,epoch_qs,epoch_episode_rewards, episode_rewards_history, \
            epoch_episode_steps, epoch_episodes, episodes, t


def train(nb_train_steps, memory, batch_size, t, param_noise_adaption_interval, agent):

    epoch_actor_losses = []
    epoch_critic_losses = []
    epoch_adaptive_distances = []
    for t_train in range(nb_train_steps):
        # Adapt param noise, if necessary.
        if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
            distance = agent.adapt_param_noise()
            epoch_adaptive_distances.append(distance)

        cl, al = agent.train()
        epoch_critic_losses.append(cl)
        epoch_actor_losses.append(al)
        agent.update_target_net()
    return agent, epoch_actor_losses, epoch_critic_losses, epoch_adaptive_distances


def evaluate(agent, eval_obs, max_action, eval_env, render_eval, sess, t, eval_episode_rewards_history,
              best_score, data_path, saver_best, nb_eval_episodes, study, max_steps, size_buffer):

    eval_episode_rewards = []
    eval_qs = []
    if eval_env is not None:
        eval_episode_reward = 0.
        for i in range(nb_eval_episodes): # 5 rollouts
            while True:
                eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                if render_eval:
                    eval_env.render()
                eval_episode_reward += eval_r
                eval_qs.append(eval_q)
               # print("evaluate eval_done : ", eval_done, " eval_episode_reward:", eval_episode_reward)
                if eval_done:
                    eval_obs = eval_env.reset()
                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_rewards_history.append(eval_episode_reward)
                    eval_episode_reward = 0.
                    break
        if mpi_mean(eval_episode_rewards) > best_score:
            best_score = mpi_mean(eval_episode_rewards)
            if not os.path.exists(data_path+'tf_save/'):
                os.mkdir(data_path+'tf_save/')
                os.mkdir(data_path+'tf_save/'+'best1_5M/') # optional, to save GEP data at 1.5M
            for file in os.listdir(data_path+'tf_save/'):
                if path.isfile(data_path+'tf_save/'+file): # optional, to save GEP data at 1.5M
                    os.remove(data_path+'tf_save/'+file)
            saver_best.save(sess, data_path+'tf_save/'+ 'best_actor_step'+str(t)+'_score'+str(int(best_score)))


    return agent, eval_episode_rewards, eval_qs, best_score, eval_obs, eval_env, \
            eval_episode_rewards_history
