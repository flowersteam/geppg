import numpy as np

from controllers import NNController
from representers import CheetahRepresenter, CMCRepresenter
from inverse_models import KNNRegressor

from gep_utils import *

def cheetah_config():

    # run parameters
    nb_bootstrap = 50
    nb_explorations = 500
    nb_tests = 100
    nb_timesteps = 1000
    offline_eval = (1e6, 10)  # (x,y): y evaluation episodes every x (done offline)

    # controller parameters
    hidden_sizes = []
    controller_tmp = 0.15
    activation = 'relu'
    subset_obs = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]
    norm_values = np.array([[-0.18, 0.064, 0.14, 0.17, 0.079, -0.082, -0.08, -0.084, 0.30,
                             -0.011, 0.0078, 0.0065, -0.055, 0.014, -0.011, 0.073, 0.054],
                            [0.073, 0.25, 0.22, 0.22, 0.25, 0.35, 0.27, 0.25, 0.64, 0.38,
                             1.1, 2.1, 2.9, 3.5, 3.9, 3.5, 3.4]])  # to zscore observations
    scale = None
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, 6, norm_values, scale, activation)
    nb_weights = controller.nb_weights

    # representer
    representer = CheetahRepresenter()
    initial_space = representer.initial_space
    goal_space = np.array([[2., 6.], [-0.5, 0.]])  # space in which goal are sampled
    engineer_goal = np.array([6., -0.2])  # engineer goal
    # scale engineer goal to [-1,1]^N
    from gep_utils import scale_vec
    engineer_goal = scale_vec(engineer_goal, initial_space)

    nb_rep = representer.dim

    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = 0.1

    return nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval,  \
           controller, representer, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights

def cmc_config():

    # run parameters
    nb_bootstrap = 10
    nb_explorations = 50
    nb_tests = 100
    nb_timesteps = 1000
    offline_eval = (1e6, 10)  # (x,y): y evaluation episodes every x (done offline)

    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(2)
    norm_values = None
    scale = np.array([[-1.2,0.6],[-0.07,0.07]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, 1, norm_values, scale, activation)
    nb_weights = controller.nb_weights

    # representer
    representer = CMCRepresenter()
    initial_space = representer.initial_space
    goal_space = representer.initial_space  # space in which goal are sampled
    nb_rep = representer.dim
    engineer_goal = np.array([0.5, 1.5, 5.])  # engineer goal
    # scale engineer goal to [-1,1]^N
    engineer_goal = scale_vec(engineer_goal, initial_space)
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = 0.1

    return nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval,  \
           controller, representer, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights