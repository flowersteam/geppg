import os
import pickle
import numpy as np

path_buffers = '/media/flowers/3C3C66F13C66A59C/data_save/buffers_all/Cheetah/buffers900/'
name_env = 'Cheetah'
n_buffers = 20
buffers_id = list(range(1,1+n_buffers))

n_episodes = 900
n_eps2 = 900
ind_test = []
ind_train = []
for i in range(int(n_episodes/2)):
    ind_test.append(2+12*i)
    ind_train.append(12*i)
    ind_train.append(12*i+1)

assert len(ind_test)==int(n_episodes/2)

test_perfs = np.zeros([n_buffers, int(n_episodes/2)]) # perf each 2000 steps (2eps)

for b_id in range(n_buffers):
    # b='Cheetah_model_500_'+str(b_id+1)+'_buffer'
    b=name_env+'_model_'+str(n_eps2)+'_'+str(buffers_id[b_id])+'_buffer'
    print('Extracting file '+b)
    with open(path_buffers+b, 'rb') as f:
        buffer = pickle.load(f)
    n_eps = len(buffer)
    print('nb episodes: ', n_eps)
    perfs_buffer = []
    for i in range(len(ind_test)):
        tmp_perf = np.zeros([10])
        for j in range(10):
            ind_ep = ind_test[i]+j
            n_timesteps = len(buffer[ind_ep])
            count=0
            for k in range(n_timesteps):
                count+=buffer[ind_ep][k]['reward'][0]
            tmp_perf[j] = count
        test_perfs[b_id,i] = tmp_perf.mean()
    del buffer
    np.savetxt(path_buffers+'buffer_perfs', test_perfs)

# cut and rename
for b_id in range(n_buffers):
    # b='Cheetah_model_500_'+str(b_id+1)+'_buffer'
    b=name_env+'_model_'+str(n_eps2)+'_'+str(buffers_id[b_id])+'_buffer'

    print('Extracting file '+b)
    with open(path_buffers+b, 'rb') as f:
        buffer = pickle.load(f)
    buffer2=[]
    for i in ind_train:
        buffer2.append(buffer[i])
    del buffer
    n_eps = len(buffer2)
    print('nb episodes: ', n_eps)
    with open(path_buffers+name_env+'_buffer_'+str(n_eps2)+'_'+str(buffers_id[b_id]), 'wb') as f:
        pickle.dump(buffer2,f)



# path=path_buffers
# l = []
# steps = []
# for i in range(1,n_buffers+1):
#     name = 'CMC_buffer_500_'+str(20+i)
#     f=open(path+name,'rb')
#     buff=pickle.load(f)
#     le = 0
#     for j in range(len(buff)):
#         le+=len(buff[j])
#         if j%2!=0:
#             steps.append(le)
#     l.append(le)
#     np.savetxt(path+name+'_steps', np.array(steps))


