import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import glob 
import seaborn as sns

sns.set_theme(style="whitegrid", palette=None, font="Verdana")

def moving_avg(x,N=5):
    return np.convolve(x, np.ones(N)/N, mode='same')

rl_files = sorted(glob.glob('../results/RL/return_rl_*.p'))
crl_files = sorted(glob.glob('../results/RL/return_crl_*.p'))

labels = ['RL','CRL']
color_list = [u'#1f77b4', u'#ff7f0e']
# labels = ['PPO','DRON','LIAM','GSCU-Greedy(Ours)']
# color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c','purple']
file_list = [rl_files, crl_files]

all_data = {}
for i in range(len(labels)):
    files = file_list[i]
    all_data['episode'] = []

    for seed in range(len(files)):

        data = pd.read_pickle(files[seed])
        label = labels[i]
        return_list = data['test_list_vae']
        episode_list = list(range(len(return_list)))

        n_point = len(return_list)

        if label not in all_data:
            all_data[label] = return_list
        else:
            all_data[label] += return_list
        all_data['episode'] += episode_list

print ('episode', len(all_data['episode']))
print ('RL', len(all_data['RL']))
all_data_df = pd.DataFrame.from_dict(all_data)

# print (len(all_data['episode']))

plt.figure(figsize=(8,6))
for i in range(len(labels)):
    sns.lineplot(data=all_data_df, x='episode', y=labels[i], alpha=0.8, linewidth=2.0, color=color_list[i])
plt.xlim(0,len(episode_list)-1)
plt.title('Predator Prey', fontsize=28)
plt.xlabel('Episode', fontsize=22)
plt.ylabel('Returns', fontsize=22)
plt.legend(labels, fontsize=18)
plt.yticks(fontsize=20)
xtick_list = [str(i*0.5+0.5)+'k' for i in np.arange(0, n_point, 4)]
plt.xticks(np.arange(0, n_point, 4), xtick_list, fontsize=20)
# plt.xticks(np.arange(n_point_per_opponet//2, n_point_per_opponet*n_opponent_per_img,20), ['adv'+str(j) for j in range(n_opponent_per_img)])
# plt.axis.xticks.tick_top()
# plt.show()
plt.savefig('results/pp_rl_training.png', bbox_inches='tight')
# plt.close()

