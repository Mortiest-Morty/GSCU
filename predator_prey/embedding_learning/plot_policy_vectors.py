from collections import defaultdict
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import pickle
import pandas as pd 
from torch.autograd import Variable
import torch.nn.functional as F
from opponent_models_e import Encoder

import matplotlib.pyplot as plt
from utils.config_predator_prey import Config

import seaborn as sns
# sns.set_theme(style="whitegrid",font="Verdana")

N_SAMPLE = 3000

obs_dim = 16
# num_adv_pool = Config.NUM_ADV_POOL
num_adv_pool = len(Config.ADV_POOL_SEEN)
action_dim = 7
input_dim = int((obs_dim+action_dim)*50)
hidden_dim = Config.HIDDEN_DIM
latent_dim = Config.LATENT_DIM

"""
encoder:
v4 -- hidden_dim 128 learning_rate 1e-5
v6 -- hidden_dim 1024 learning_rate 1e-4

data:
v4/v6 -- N S W
v5 -- N S W E
"""

args = {
    "encoder": "v6",
    "data": "v5",
    "png": "v9",
}


def plot(args):
    encoder_path = '../model_params/VAE/encoder_e_param_' + args["encoder"] + '_29.pt'
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device("cpu")))
    encoder.eval()

    embedding_data = {}
    
    data_dir = Config.DATA_DIR

    data_file = data_dir + 'e_data_simple_tag_' + args["data"] + '.p'
    data = pd.read_pickle(data_file)
    tau_dict = data['data_tau']
    for key in tau_dict:
        tau_dict[key] = np.array(tau_dict[key])
        # {id: [?, 1150]}
    ## 固定策略
    for adv_id, tau in tau_dict.items():
        tau_tensor = Variable(torch.tensor(tau))
        embedding = encoder(tau_tensor.float())
        embedding = embedding.detach().numpy()
        embedding_data[adv_id] = embedding  # [?, 2]

    data = embedding_data
    mean = []
    label_all = []
    label = []

    for idx, i in enumerate(data):
        
        data_i = data[i]
        
        if idx == 0:
            data_all = data_i[:N_SAMPLE, :]
        else:
            data_all = np.vstack((data_all, data_i[:N_SAMPLE, :]))
        label_all += [i for _ in range(N_SAMPLE)]
        mean += [np.mean(data_i[:N_SAMPLE, :], 0)]
        label.append(i)


    print ('label',label)
    print ('data_all',data_all.shape)

    X_2d_all = np.array(data_all)
    X_2d = np.array(mean)
    # print(X_2d_all.shape)
    # print(X_2d.shape)
    # sys.exit()
    label_all = np.array(label_all)
    label = np.array(label)
    print ('label_all',label_all.shape)

    
    agent_names = ['N', 'S', 'W', 'E(U)']
    rgbs = ['lightseagreen','y','lightslategrey','deepskyblue']
    # rgbs = ['lime', 'g', 'darkgreen', 'springgreen', 'lightseagreen','lightskyblue','orange', 'tomato','r'] 
    # rgbs = ['lightseagreen','y','lime','steelblue','grey','orange','lightslategrey','r']  # for p0
    # rgbs = ['lightseagreen','y','lime','deepskyblue','steelblue','grey','orange','seagreen','r']  # for p1
    # rgbs = ['orange', 'tomato', 'lime', 'g', 'darkgreen', 'springgreen', 'lightseagreen','deepskyblue','skyblue','lightskyblue'] 
    
    
    for j in [0]:
        plt.rcParams["figure.figsize"] = (8,6)
        for a,b in zip([j],[j+1]):
            print ('axis:', a,b)
            fig, ax = plt.subplots()
            for i in label:
                c = rgbs[i]
                l = agent_names[i]
                #c = 'dimgrey'
                plt.text(X_2d[label == i, a], X_2d[label == i, b], l, fontsize=20)
                plt.scatter(X_2d_all[label_all == i, a], X_2d_all[label_all == i, b], c=c, label=l, s=8, alpha=0.5)
                # plt.scatter(X_2d_all[label_all == i, a], X_2d_all[label_all == i, b], c=c, label=l, s=10, alpha=0.8) # not for paper
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.legend(fontsize=20)
        # plt.xticks(fontsize=14) # not for paper
        # plt.yticks(fontsize=14) # not for paper
        plt.savefig('results/e_fig_sample_' + args["png"] +'.png', bbox_inches='tight')
        # plt.savefig('kp_emb_seen.pdf', bbox_inches='tight')
        # plt.savefig('results/sample/vae_fig_sample'+file_name+'_'+str(j)+'.jpg', bbox_inches='tight')
        # plt.savefig('results/emb_for_slides_.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

    return embedding_data

if __name__ == "__main__":
    plot(args)