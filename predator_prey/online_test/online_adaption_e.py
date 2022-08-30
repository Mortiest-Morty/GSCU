import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import pickle
import torch
import logging

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from embedding_learning.data_generation_e import get_all_adv_policies
from conditional_RL.conditional_rl_model_e import PPO_E
from conditional_RL.ppo_model import PPO
from utils.multiple_test import *
from utils.config_predator_prey import Config
from online_test.unseen_detector import Detector

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

N_ADV = 3
seen_adv_pool = Config.ADV_POOL_SEEN
unseen_adv_pool =  Config.ADV_POOL_UNSEEN
mix_adv_pool = Config.ADV_POOL_MIX
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_three_adv_all_policies(env, adv_pool):
    all_policies_idx0 = get_all_adv_policies(env,adv_pool,agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env,adv_pool,agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env,adv_pool,agent_index=2)
    all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]
    return all_policies

def main(args):
    gamma = 0.99
    hidden_dim = Config.HIDDEN_DIM
    seed = args.seed
    adv_change_freq = 200
    n_opponent = 50
    log_per_n_opponent = 5
    num_episodes = adv_change_freq * n_opponent

    actor_lr = args.lr1
    critic_lr = args.lr2
    
    window_size = Config.WINDOW_SIZW
    encoder_input_dim = int((16+7)*50)
    
    adv_pool_type = args.opp_type

    rst_dir = Config.ONLINE_TEST_RST_DIR
    data_dir = Config.DATA_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False)
    
    offline_data_dir = data_dir + 'e_data_simple_tag_' + args.version + '.p'

    if adv_pool_type == 'mix':
        dataloader = open(data_dir+'policy_vec_seq_8.p', 'rb')
    elif adv_pool_type == 'seen' or adv_pool_type == 'unseen':
        dataloader = open(data_dir+'policy_vec_seq_3.p', 'rb')
    else:
        print('Please choose seen/unseen/mix')
    data = pickle.load(dataloader)
    policy_vec_seq = data['policy_vec_seq']

    test_id = args.version

    scenario = scenarios.load(args.scenario).Scenario()
    world_pi = scenario.make_world()
    world_e = scenario.make_world()

    env_pi = MultiAgentEnv(world_pi, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_e = MultiAgentEnv(world_e, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None, shared_viewer=False, discrete_action=True)
    env_pi.seed(seed)
    env_e.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    state_dim = env_e.observation_space[3].shape[0]
    action_dim = env_e.action_space[3].n
    embedding_dim = Config.LATENT_DIM
    encoder_weight_path = args.encoder_file

    agent_e = PPO_E(state_dim + action_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, encoder_input_dim)
    agent_pi = PPO(state_dim, hidden_dim, action_dim, 0.0, 0.0, gamma)
    ppo_e_path = args.rl_file
    ppo_pi_path = '../model_params/RL/params_pi.pt'
    agent_e.init_from_save(ppo_e_path)
    agent_pi.init_from_save(ppo_pi_path)
    
    if adv_pool_type == 'seen':
        selected_adv_pool = seen_adv_pool
    elif adv_pool_type == 'unseen':
        selected_adv_pool = unseen_adv_pool
    else:
        selected_adv_pool = mix_adv_pool
    
    all_policies_pi = get_three_adv_all_policies(env_pi, selected_adv_pool)
    all_policies_e = get_three_adv_all_policies(env_e, selected_adv_pool)
    
    Det = Detector(offline_data_dir, agent_e.encoder, len(seen_adv_pool))

    cur_adv_idx = 300 # just a random number to identify the sequence start point. can be any number between 0 to 800
    cur_n_opponent = 0

    return_list_e = []
    return_list_pi = []

    for i_episode in range(num_episodes):

        if i_episode % adv_change_freq == 0:
            policies_pi = []
            policies_e = []

            policy_vec = policy_vec_seq[cur_adv_idx]
            for j in range(N_ADV):
                adv_idx = np.argmax(policy_vec)
                policies_pi.append(all_policies_pi[j][adv_idx])  # 0～2 oppos are the same
                policies_e.append(all_policies_e[j][adv_idx])
            opp_name = selected_adv_pool[adv_idx]
        if i_episode == 0:
            tau_vec = np.zeros((N_ADV, encoder_input_dim))
        else:
            for j in range(N_ADV):
                tau_vec[j, :] = np.concatenate(temp_tau[j])
        
        selected_pos_idx = np.random.randint(0, N_ADV)  # randomly sample a position and encode its traj
        temp_tau = [[] for _ in range(N_ADV)]
        
        episode_return_pi = 0
        episode_return_e = 0

        obs_n_pi = env_pi._reset()
        obs_n_e = env_e._reset()
        
        tau_vec_tensor = torch.tensor(tau_vec, dtype=torch.float).to(device)

        obs_traj_e = [np.zeros(state_dim)]*(window_size-1)
        act_traj_e = [np.zeros(action_dim)]*window_size
        hidden_e = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

        for st in range(args.steps):
            act_n_pi = []
            act_n_e = []

            # pi_1^*
            for j, policy in enumerate(policies_pi):
                act_pi = policy.action(obs_n_pi[j])
                act_n_pi.append(act_pi)
            act_pi,_,_ = agent_pi.select_action(obs_n_pi[3], 2)
            act_n_pi.append(act_pi)
            next_obs_n_pi, reward_n_pi, _,_ = env_pi._step(act_n_pi)
            episode_return_pi += reward_n_pi[env_pi.n-1]
            obs_n_pi = next_obs_n_pi

            # ppo_e
            for j, policy in enumerate(policies_e):
                act_e = policy.action(obs_n_e[j])
                act_n_e.append(act_e)
                temp_tau[j].append(obs_n_e[j])
                temp_tau[j].append(act_e)
            latent = agent_e.encoder(tau_vec_tensor[selected_pos_idx:selected_pos_idx+1, :])  # [3, 2] -> [1, 2]
            
            obs_traj_e.append(obs_n_e[3])
            obs_traj_tensor_e = torch.tensor([obs_traj_e], dtype=torch.float).to(device)
            act_traj_tensor_e = torch.tensor([act_traj_e], dtype=torch.float).to(device)
            act_e,_,_ = agent_e.select_action(obs_traj_tensor_e, act_traj_tensor_e, hidden_e, latent, 2)
            act_n_e.append(act_e)
            
            next_obs_n_e, reward_n_e, done_n_e, _ = env_e._step(act_n_e)
            
            if len(obs_traj_e) >= window_size:
                obs_traj_e.pop(0)
                act_traj_e.pop(0)
            
            episode_return_e += reward_n_e[3]
            obs_n_e = next_obs_n_e
            act_traj_e.append(act_e[:-2])

        Det.add_adv_traj(np.concatenate(temp_tau[selected_pos_idx]))
        if (i_episode+1) % Det.detect_step == 0:
            is_unseen = Det.detect()
            if is_unseen:
                pass  # TODO: online fine-tuning
        
        # TODO: detection accuracy
        
        return_list_e.append(episode_return_e)
        return_list_pi.append(episode_return_pi)

        if (i_episode+1) % adv_change_freq == 0:
            logging.info("opp idx: {}, opp name: {}, ppo_e: {:.2f}, | pi: {:.2f}".format(
                        cur_n_opponent,opp_name,np.mean(return_list_e[-adv_change_freq:]),np.mean(return_list_pi[-adv_change_freq:])))

            if (cur_n_opponent+1) % log_per_n_opponent == 0:
                print ( 'total # of opp: ', cur_n_opponent+1,
                        ', avg ppo_e', np.mean(return_list_e), 
                        '| avg pi', np.mean(return_list_pi))
                print ('-'*10)

                result_dict = {}
                result_dict['opponent_type'] = adv_pool_type
                result_dict['version'] = test_id
                result_dict['n_opponent'] = cur_n_opponent+1
                result_dict['pi'] = return_list_pi
                result_dict['ppo_e'] = return_list_e
                pickle.dump(result_dict, open(rst_dir+'online_adaption_'+test_id+'_'+adv_pool_type+'.p', "wb"))
        
        
        if i_episode % adv_change_freq == 0 and i_episode > 0:
            cur_adv_idx += 1
            cur_n_opponent += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-o', '--opp_type', default='seen', 
                        choices=["seen", "unseen", "mix"], help='type of the opponents')
    parser.add_argument('-e', '--encoder_file', default='../model_params/VAE/encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-r', '--rl_file', default='../model_params/RL/params_demo.pt', help='conditional RL file')
    parser.add_argument('-l1', '--lr1', default=5e-3, help='Actor online learning rate')
    parser.add_argument('-l2', '--lr2', default=5e-3, help='Critic online learning rate')
    args = parser.parse_args()
    

    main(args)
