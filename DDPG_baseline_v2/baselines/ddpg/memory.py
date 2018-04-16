import pickle

import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

def load_from_cedric(filename):
    """
    used to load a replay buffer saved under Cédric Colas' format into a replay buffer of Pierre Fournier's format
    """
    with open(filename, 'rb') as file:
        buffer = pickle.load(file)
        buffer_baseline = [[],[],[],[],[]]
        for i in range(len(buffer)):
            for idx in range(len(buffer[i])):
                buffer_baseline[0].append(buffer[i][idx]['state0'])
                buffer_baseline[1].append(buffer[i][idx]['action'])
                buffer_baseline[2].append(buffer[i][idx]['reward'])
                buffer_baseline[3].append(buffer[i][idx]['state1'])
                buffer_baseline[4].append(buffer[i][idx]['terminal1'])

    return buffer_baseline

def load_from_geppg(memory):
    """
    Used to bootstrap a replay buffer from a memory object filled by GEP.
    """
    buffer = [[],[],[],[],[]]
    action_seq = np.copy(memory['actions'])
    obs_seq = np.copy(memory['observations'])
    rew_seq = np.copy(memory['rewards'])
    n_eps = action_seq.shape[0]
    n_obs = obs_seq.shape[2]
    assert obs_seq.shape[0] == n_eps
    assert rew_seq.shape[0] == n_eps
    for i in range(n_eps):
        for j in range(action_seq[i, :, :].shape[0]):
            buffer[0].append(obs_seq[i, j, :])
            buffer[1].append(action_seq[i, j, :])
            buffer[2].append(rew_seq[i, j+1, :])
            buffer[3].append(obs_seq[i, j+1, :])
            try:
                if np.isnan(obs_seq[i,j+2,0]):
                    buffer[4].append(True)
                    break
                elif np.all(obs_seq[i,j+2,:]==np.zeros([n_obs])):
                    buffer[4].append(True)
                    break
                else:
                    buffer[4].append(False)
            except:
                buffer[4].append(True)
                break
    return buffer

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)
