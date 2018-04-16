
import numpy as np

def ddpg_config(env_id,
                study,
                data_path,
                noise='ou_0.3',
                trial_id=999,
                seed=int(np.random.random()*1e6),
                nb_epochs=1000,
                buffer_location=None,
                gep_memory=None,
                ):

    args_dict = dict(env_id=env_id,
                     render_eval=False,
                     layer_norm=True,
                     render=False,
                     normalize_returns=False,
                     normalize_observations=True,
                     seed=seed, #int(np.random.random()*1e6),
                     critic_l2_reg=1e-2,
                     batch_size=64,  # per MPI worker
                     actor_lr=1e-4,
                     critic_lr=1e-3,
                     popart=False,
                     gamma=0.99,
                     reward_scale=1.,
                     clip_norm=None,
                     nb_epochs=nb_epochs, # with default settings, perform 2M steps total
                     nb_epoch_cycles=20,
                     nb_train_steps=50, # per epoch cycle and MPI worker
                     nb_eval_steps=1000, # per epoch cycle and MPI worker
                     nb_eval_episodes=10,
                     nb_rollout_steps=100, # per epoch cycle and MPI worker
                     noise_type=noise,  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
                     evaluation=True,
                     study=study, # 'DDPG',  # 'GEP_PG', #  'GEP_FPG', #
                     buffer_location=buffer_location,
                     data_path=data_path,
                     trial_id=trial_id,
                     max_memory=1e6,
                     gep_memory=gep_memory
                     )
    if env_id=='MountainCarContinuous-v0':
        args_dict['nb_epochs'] = 250

    return args_dict


