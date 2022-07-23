#!/usr/bin/env python
from collections import defaultdict
import pickle
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time
from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *
import multiagent.scenarios as scenarios
from utils.config_predator_prey import Config


N_ADV = 3

N_TRAINING_EPISODES = 5000
N_TESTING_EPISODES = N_TRAINING_EPISODES//10
SEED_TRAINING = 0
SEED_TESTING = 1

# ramdomly generate 3 adv policies and a agent 
def generate_policies(env, adv_pool,agent_pool):
    selected_advs_ids = np.random.choice(range(0, len(adv_pool)), size=N_ADV, replace=True)
    selected_agent_ids = np.random.choice(range(0, len(agent_pool)), size=1, replace=True)
    adv_policies = []
    agent_policies = []
    for idx, adv_id in enumerate(selected_advs_ids):
        policy = get_policy_by_name(env,adv_pool[adv_id],idx)
        adv_policies.append(policy)
    agent_policies = [eval(agent_pool[selected_agent_ids[0]] + "(env," + str(N_ADV) +")")]
    policies = adv_policies + agent_policies
    return policies, selected_advs_ids

def get_all_adv_policies(env,adv_pool,agent_index):
    all_policies = []
    for adv_id in range(len(adv_pool)):
        policy = get_policy_by_name(env,adv_pool[adv_id],agent_index)
        all_policies.append(policy)
    return all_policies

def get_policy_by_name(env,policy_name,agent_index):
    return eval(policy_name + "(env," + str(agent_index) +")")

def main(args):
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        shared_viewer=False, discrete_action=True)

    adv_pool = Config.ADV_POOL_SEEN
    agent_pool = ['ConservativePolicy'] # use the conservative policy to 
    data_dir = Config.DATA_DIR

    version = args.version
    
    testing_mode = [True, False]
    for is_test in testing_mode:
        if is_test:
            this_version = version + '_test'
            eposides = N_TESTING_EPISODES
            seed = SEED_TESTING
            print ('Generating testing data...')
        else:
            this_version = version
            eposides = N_TRAINING_EPISODES
            seed = SEED_TRAINING
            print ('Generating training data...')

        np.random.seed(seed)
        random.seed(seed)

        data_tau = defaultdict(list)
        
        n_step = 0
        n_sample = 0

        for e in range(eposides):
            policies, adv_ids = generate_policies(env, adv_pool, agent_pool)  # sampled 3 policy
            
            data_s = []
            data_a = []
            data_i = []
            temp_tau = [[] for _ in range(N_ADV)]
            
            # execution loop
            obs_n = env._reset()
            steps = args.steps
            for _ in range(steps):
                # query for action from each agent's policy
                act_n = []
                for i, policy in enumerate(policies):
                    act = policy.action(obs_n[i])
                    act_n.append(act)
                    # collect the obs/a/policy_index for opponent policies
                    if i < N_ADV:
                        data_s.append(obs_n[i])
                        data_a.append(act)
                        data_i.append(adv_ids[i])
                        temp_tau[i].append(obs_n[i])
                        temp_tau[i].append(act)
                        # print("obs_n[i]: ", obs_n[i])
                        # print("act: ", act)
                        # print("adv_ids[i]: ", adv_ids[i])

                        n_sample += 1

                # step environment
                obs_n, reward_n, done_n, _ = env._step(act_n)
                n_step += 1
            

            for i in range(N_ADV):
                data_tau[adv_ids[i]].append(np.concatenate(temp_tau[i]))
            
            if (e+1) % (eposides//10) == 0:
                print ('n eposides',e+1, ' | n sample',n_sample)

        for adv, taus in data_tau.items():
            print("adv: ", adv)
            print("num_tau: ", len(taus))
            print("shape_tau: ", np.array(taus).shape)
        
        e_data = {
            'data_tau': data_tau,
            }

        pickle.dump(e_data, 
            open(data_dir+'e_data_simple_tag_' + this_version + '.p', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, type=int, help='Num of steps in a single run')
    parser.add_argument('-v', '--version', default='v0')
    args = parser.parse_args()


    main(args)

