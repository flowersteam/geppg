# geppg

### Implementation of the GEP-PG algorithm.

This is the codebase for the paper GEP-PG: Decoupling Exploration and Exploitation in Deep Reinforcement Learning Algorithms.
https://arxiv.org/pdf/1802.05054.pdf

GEP-PG aims at combining sequentially an efficient exploration strategy (Goal Exploration Process or GEP) with a state-of-the-art reinforcement learning algorithm (Deep Deterministic Policy Gradient or DDPG).

To launch a run of GEP-PG on Half Cheetah (OpenAI Gym, v2):

```
python3 gep.py --trial 1 --env_id HalfCheetah-v2 --study GEP_PG --nb_exploration 500 --noise_type ou_0.3
```

the options are: 
```
--trial: trial id. \n
--env_id: HalfCheetah-v2, MountainCarContinuous-v0.
--study: GEP, GEP_PG, DDPG.
--nb_exploration: number of exploration episodes performed by GEP before switching to DDPG.
--noise_type: type of noise used by DDPG ou_0.3 is Ornstein Uhlenbeck noise with sigma=0.3,
              adaptive_parameter_0.2 is an adaptive noise on parameters with sigma=0.2
              (as described in https://arxiv.org/abs/1706.01905).
              decreasing_ou_0.6: OU noise linearly annealed from 0.6 to 0
--saving_folder: folder in which to save the runs.
```
DDPG: This code integrates a modified version of the OpenAI Baselines and uses the included implementation of DDPG.

GEP: Goal Exploration Processes encompass a large variety of strategies endowed with efficient exploration abilities. The agent is intrinsically motivated to select goals in his environment and to try to reach them. The implementation we use here is the simplest form of GEP. More sophisticated variants can be implemented, see https://arxiv.org/abs/1708.02190.

### Reproducing the paper
To reproduce the paper, each algorithm tested should be run 20 times (with 20 different random seeds). The python files geppg/scripts/extract_results.py and geppg/scripts/plot_multiple_curves.py contain instructions on how to extract data, plot and compute statistical tests from the outputs of gep.py.

Please do not hesitate to contact me for questions, or to report bugs.

email: cedric.colas@inria.fr

