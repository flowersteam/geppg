

import numpy as np
folder='/media/flowers/3C3C66F13C66A59C/data_save/ddpg_study_baseline/data/HalfCheetah-v1/finals/GEP2M_ComplexPolicy/'
f1= '_2000_6464_21_40_scores'
f2 = 'eval_performances_GEP2M_complex_policy'
scores_our = np.loadtxt(folder+f1)
eval_perfs = np.loadtxt(folder+f2)
scores_litt =np.mean(eval_perfs[:,-10:],axis=1)
# toSave = np.array([[scores_our.mean(), scores_our.std(), scores_our.min(), scores_our.max()],\
#                       [scores_litt.mean(), scores_litt.std(), scores_litt.min(), scores_litt.max()]])
np.savetxt(folder+'scores_our', scores_our)
