import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np

import pickle

N_OPP = 3
SEQ_LEN = 1000

policy_vec_seq = []
adv_type_seq = []

for i in range(SEQ_LEN):
    selected_adv_idx = np.random.randint(0, N_OPP)
    policy_vec = np.zeros(N_OPP)
    policy_vec[selected_adv_idx] += 1
    policy_vec_seq.append(policy_vec)


result_dict = {}
result_dict['policy_vec_seq'] = policy_vec_seq
pickle.dump(result_dict, open('policy_vec_seq_' + str(N_OPP) + '.p', "wb"))

