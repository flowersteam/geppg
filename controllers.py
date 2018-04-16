import numpy as np
import torch

class NNController():

    def __init__(self, hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale,  activation):

        self._subset_obs = subset_obs
        self._controller_tmp = controller_tmp
        self._norm_values = norm_values
        self._scale = scale # None or space from which observation should be scale to [-1,1]^N

        if self._norm_values is not None and self._subset_obs is not None:
            self._norm_values = self._norm_values[:, self._subset_obs]

        if scale is not None:
            self._min = self._scale[:, 0]
            self._range = self._scale[:, 1] - self._min

        self._layer_sizes = [len(self._subset_obs)] + hidden_sizes + [nb_act]
        # compute number of parameters
        self._nb_weights = 0
        for i in range(len(self._layer_sizes) - 1):
            self._nb_weights += self._layer_sizes[i] * self._layer_sizes[i + 1]

        self._activation_function = activation
        self._dtype = torch.FloatTensor  # run on CPU
        self._weights = None # weights of the NN

    def step(self, policy, obs):

        obs_in = np.copy(obs.astype(np.float)).squeeze()
        policy_in = np.copy(policy).squeeze()

        # format weights
        self._weights = []
        index = 0
        for i in range(len(self._layer_sizes) - 1):
            ind_weights = np.arange(index, index + self._layer_sizes[i] * self._layer_sizes[i + 1])
            index = index + (self._layer_sizes[i]) * self._layer_sizes[i + 1]
            self._weights.append(torch.from_numpy(
                policy_in[ind_weights].reshape([self._layer_sizes[i], self._layer_sizes[i + 1]])).type(self._dtype))

        # take only a subset of observation as input to the NN
        if self._subset_obs is not None:
            obs_in = obs_in[self._subset_obs]

        # normalize observations (zscore)
        if self._norm_values is not None:
            obs_in = (obs_in - self._norm_values[0, :]) / self._norm_values[1, :]
        # or scale values to [-1,1]^N if scale is not None
        elif self._scale is not None:
            obs_in = ((obs_in - self._min) * 2*np.ones([2]) / self._range) - np.ones([2])


        x = torch.from_numpy(obs_in.reshape(1,-1)).type(self._dtype)

        y = x.mm(self._weights[0])


        for i in range(len(self._layer_sizes) - 2):
            if self._activation_function == 'relu':
                y = y.clamp(min=0)
            elif self._activation_function == 'tanh':
                y = np.tanh(y)
            elif self._activation_function == 'leakyrelu':
                y[y < 0] = 0.01 * y[y < 0]
            y = y.mm(self._weights[i + 1])
        y = y.numpy()
        y = np.tanh(np.longfloat(self._controller_tmp * y))

        self._action = y[0, :].astype(np.float64)
        if self._action.size == 1:
            self._action = np.array([self._action])

        self._action = np.clip(self._action, -1, 1) # just in case..

        return self._action




    @property
    def nb_weights(self):
        return self._nb_weights