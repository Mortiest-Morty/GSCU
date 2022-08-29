import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from collections import namedtuple
import pickle
import logging
import torch
from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from embedding_learning.data_generation_e import get_all_adv_policies
from conditional_RL.ppo_model_e import PPO
from utils.multiple_test import play_multiple_times_train
from utils.config_predator_prey import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

N_ADV = 3
adv_pool = Config.ADV_POOL_SEEN
n_adv_pool = len(adv_pool)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_three_adv_all_policies(env, adv_pool):
    all_policies_idx0 = get_all_adv_policies(env, adv_pool, agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env, adv_pool, agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env, adv_pool, agent_index=2)
    all_policies = [all_policies_idx0, all_policies_idx1, all_policies_idx2]
    return all_policies

def main(args):
    Transition_e = namedtuple('Transition_e', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'obs_traj', 'act_traj'])

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gamma = 0.99
    hidden_dim = Config.HIDDEN_DIM
    num_episodes = 10000 
    actor_lr = args.lr1
    critic_lr = args.lr2
    window_size = Config.WINDOW_SIZW
    
    checkpoint_freq = 1000
    adv_change_freq = 50
    batch_size = 4096
    ppo_update_freq = 10
    test_freq = 500

    result_dir  = Config.RL_TRAINING_RST_DIR
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=False)

    exp_id = args.version
    settings = {}
    settings['exp_id'] = exp_id
    settings['hidden_dim'] = hidden_dim
    settings['actor_lr'] = actor_lr
    settings['critic_lr'] = critic_lr
    settings['batch_size'] = batch_size
    settings['ppo_update_freq'] = ppo_update_freq
    settings['adv_change_freq'] = adv_change_freq
    settings['seed'] = seed

    print(settings)
    scenario = scenarios.load(args.scenario).Scenario()
    world_e = scenario.make_world()
    
    env_e = MultiAgentEnv(world_e, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    
    env_e.seed(seed)
    np.random.seed(seed)
    
    state_dim = env_e.observation_space[env_e.n-1].shape[0]
    action_dim = env_e.action_space[env_e.n-1].n

    agent_e = PPO(state_dim + action_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma)
    agent_e.batch_size = batch_size
    agent_e.ppo_update_time = ppo_update_freq

    return_list = []
    test_return_list = []

    all_policies_e = get_three_adv_all_policies(env_e, adv_pool)
    
    selected_adv_idx = 0
    for i in range(50):
        for i_episode in range(int(num_episodes/50)):
            if i_episode % adv_change_freq == 0:
                policies_e = []
                selected_adv_idx = np.random.randint(0, n_adv_pool)
                for j in range(N_ADV):
                    policies_e.append(all_policies_e[j][selected_adv_idx])
                        
            episode_return_e = 0
            obs_n_e = env_e._reset()

            obs_traj = [np.zeros(state_dim)] * (window_size - 1)
            act_traj = [np.zeros(action_dim)] * (window_size)
            hidden = [torch.zeros((1, 1, hidden_dim)).to(device), torch.zeros((1, 1, hidden_dim)).to(device)]  # [1, 1, 1024]

            for _ in range(args.steps):
                act_n_e = []

                for j, policy in enumerate(policies_e):
                    act_e = policy.action(obs_n_e[j])
                    act_n_e.append(act_e)

                obs_traj.append(obs_n_e[3])
                obs_traj_tensor = torch.tensor([obs_traj], dtype=torch.float).to(device)  # [1, 8, 16]
                act_traj_tensor = torch.tensor([act_traj], dtype=torch.float).to(device)  # [1, 8, 5]
                act, act_index, act_prob = agent_e.select_action(obs_traj_tensor, act_traj_tensor, hidden, 2)

                act_n_e.append(act)
                next_obs_n_e, reward_n_e, _, _ = env_e._step(act_n_e)

                if len(obs_traj) >= window_size:
                    trans_e = Transition_e(obs_n_e[3], act_index, act_prob, reward_n_e[3], next_obs_n_e[3], obs_traj.copy(), act_traj.copy())
                    agent_e.store_transition(trans_e)
                    obs_traj.pop(0)
                    act_traj.pop(0)

                episode_return_e += reward_n_e[3]
                obs_n_e = next_obs_n_e
                act_traj.append(act[:-2])

            return_list.append(episode_return_e)

            if len(agent_e.buffer) >= agent_e.batch_size:
                agent_e.update()
                            
            current_episode = num_episodes / 50 * i + i_episode + 1
            if current_episode % checkpoint_freq == 0:
                agent_e.save_params(exp_id + '_rl_' + str(current_episode))
            
            if current_episode % test_freq == 0:
                play_episodes = 500
                test_returns = play_multiple_times_train(env_e, agent_e, all_policies_e, 'ppo', 'rule', adv_change_freq, play_episodes=play_episodes)
                mean_test_returns = np.mean(test_returns)
                test_return_list.append(mean_test_returns)

                logging.info("Average returns is {0} at the end of epoch {1}".format(mean_test_returns, current_episode))


                result_dict = {}
                result_dict['version'] = exp_id
                result_dict['num_episodes'] = len(return_list)
                result_dict['test_list_vae'] = test_return_list
                result_dict['settings'] = settings

                pickle.dump(result_dict, open(result_dir+'return_rl_' + exp_id + '.p', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=5e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=5e-4, help='Critic learning rate')
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-v', '--version', default='v0')
    args = parser.parse_args()

    main(args)