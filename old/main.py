import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tqdm
import os
import pickle

import datetime
import copy

now = datetime.datetime.now()


from create_graph_1 import *
from path_loss_probability import *
from age_calculation import *
from itertools import product  
from network import *
from drl import *

import sys

# random.seed(3)

# define reward function in network.py

EPISODES = 40_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

SHOW_EVERY = 2000 # episodes
# MIN_REWARD = -20  # episodes, For model save, not used

final_age = 0 # to store the age at the BS in the final episode

reward_fn = 0 # will be given a string value in the main fn
# all these are values to be maximized
# these are the rewards collected in the steps, the reward at the end of the episode will always be the sum AoI at BS as that is what we are measuring as the system performance.

# redundeant marked as sum and average are proportional so they are the same

# "BS-AoI-diff"     -> difference of previous AoI and current AoI at BS ; DONE
# "BS-AoI-sum"      -> sum of current AoI at BS                         ; DONE
# "BS-AoI-avg"      -> avg of current AoI at BS                         ; redundant
# "BS-NegAoI-sum"   -> negative of sum of current AoI at BS             ; DONE
# "BS-NegAoI-avg"   -> negative of avg of current AoI at BS             ; redundant
# "UAV-AoI-sum"     -> sum of AoI at UAV                                ; DONE
# "UAV-NegAoI-sum"  -> negative of sum of AoI at UAV
# "AoI-sum"         -> sum of current AoI at BS and UAV
# "AoI-avg"         -> avg of current AoI at BS and UAV                 ; redundant
# "Neg-AoI-sum"     -> negative of sum of current AoI at BS and UAV     ; DONE
# "Neg-AoI-avg"     -> negative of avg of current AoI at BS and UAV     ; redundant


age_BS = [] # collect sum age for every episode, i.e. the final sum AoI at BS
age_UAV = [] # collect sum age for every episode, i.e. the final sum AoI at BS

age_dist_UAV = {} # collect AoI progression for every episode per user
age_dist_BS = {}

ep_rewards = []

transmission_attempts_UAV = {} # will store the number of transmission attempt per user at every steps for all the episodes
transmission_attempts_BS = {}

min_reward = 0
max_reward = 0
average_reward = 0

if __name__ == '__main__':
    
    # if len(str(sys.argv))==1:
    # print(f'len is {((sys.argv))}')
    
    if len((sys.argv))==1: # default reward scheme
        reward_fn = "BS-NegAoI-sum"
    else:
        reward_fn = str(sys.argv[1])
        
    print(f'passed argument is {reward_fn}', flush=True) 

    now_str_1 = now.strftime("%Y-%m-%d %H:%M")
    folder_name = 'models/' + reward_fn + now_str_1

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    print("execution started at ", now_str_1, file = open(folder_name + "/results.txt", "a"), flush = True)

    print("EPISODES = ",EPISODES, ", epsilon = ", epsilon, ", EPSILON_DECAY = ", EPSILON_DECAY," , MIN_EPSILON = ",MIN_EPSILON, ", SHOW_EVERY = ",SHOW_EVERY, ", MAX_STEPS = ", MAX_STEPS, ", REPLAY_MEMORY_SIZE = ", REPLAY_MEMORY_SIZE, ", reward_fn = ", reward_fn, ", MIN_REPLAY_MEMORY_SIZE = ", MIN_REPLAY_MEMORY_SIZE, ", MINIBATCH_SIZE = ", MINIBATCH_SIZE, ", DISCOUNT = ", DISCOUNT, ", UPDATE_TARGET_EVERY = ", UPDATE_TARGET_EVERY, file = open(folder_name + "/results.txt", "a"), flush = True)

    users_list = [3] #, 50, 100, 150, 250, 350, 450]
    drones_needed = []
    deployment = "MDS" # "random" "MDS"
    scheduling = "DQL" # "greedy" "random" "DQL"

    print("results for ", deployment, " deployment", " and ", scheduling, " scheduling  follows - ", file = open(folder_name + "/results.txt", "a"), flush = True)

    for users in users_list:
        # each user density case will have multiple episodes
        
        net = Network(users, coverage_capacity)
        
        # coverage_capacity defined in create_graph_1.py
        net.start(deployment) # deploy UAV as per the required configuration
        drones_needed.append(net.UAV)
        # drones_needed is for plotting drones needed in different placement scenarios, no role in learning
        max_users_covered = net.max_coverage()
        assert max_users_covered <= coverage_capacity
        # max_users_covered to verify no drone is holding more users than 
        
        print(f'for {users} users, {net.UAV} UAVs are needed, max user covered is {max_users_covered}, coverage is {net.cover}, max coverage is {net.coverage_cap}, subcarrier is {net.UAV_capacity}')
        
        agent = DLModel(int(max_users_covered), int(net.action_size)) # 2 DL Agents created with the same architecture
        # input arguments to create the NN involves number of users and action size as they decide the number of neurons in the NN

        """
        deployment has been completed, next part is scheduling
        """

        if scheduling == "DQL":
            for episode in tqdm.tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
                done = False
                agent.Tensorboard.step = episode
                step = 1

                for k in range(net.action_size): # for the logic, see tx_attempts_UAV below
                    if episode == 1: 
                        if step==1:
                            net.preference[k] = [0]
                    else:
                        net.preference[k].append(0)
                    


                for k in (net.BS_age.keys()): # for every user
                    if episode == 1: 
                        if step==1:
                            net.tx_attempts_UAV[k] = [0]
                            net.tx_attempts_BS[k]  = [0]
                # at the first time step of first episode, set up the array with 0 and keep adding to it. future ages will be appended here
                        
                    else: # will be activated at the beginning of each episode after the 1st episode
                        net.tx_attempts_UAV[k].append(0)
                        net.tx_attempts_BS[k].append(0)
                        # once 0 has been appended, in take_RL_action the values will be increased. appended item will be at index episode-1 as episode starts from 1

                # transmission_attempts_UAV and transmission_attempts_BS are dictionaries with an array corresponding to each user. At transmission_attempts_UAV[k], the array will store the number of attempts of the kth user and the value at the jth place is its tx attempts in the jth episode

                # for the first time the code is run, step = 1 and episode = 1. Here the array is set up. Then at each time step = 1 in the beginning of an episode, the first element in each array is initialized to 0 and in network.py, they will be incremented for each step. As episode gets incremented automatically in main.py, using that to index the attempts in network.py will be sufficient. Imptt: episode starts from 1 here so in network.py, use episode-1 to reference it


                # age not resetted until now, save the time progression of each user's age at the end of every episode at BS
                for k in (net.BS_age.keys()):
                    if episode == 1:
                        age_dist_UAV[k] = []# at the episode, set up the array and future ages will be appended here
                        age_dist_BS[k]  = []
                    else:# net.BS_Age and net.UAV_Age has not been reset so at episode 2, episode 1 details will be appended
                        age_dist_BS[k].append(net.BS_age[k]) 
                        age_dist_UAV[k].append(net.UAV_age[k])

                # print(f'age_dist at beginning of episode {episode} = {age_dist}')# working 
                # print("\n\ncurrent state at beginning of the step before action is ") 
                current_state = net.reset_network() # dont comment as will update current state #
                # reset the network in the beginning of each episode, resetting involves just the re-initialization of the ages at UAV and BS

                # print("CURRENT EPISODE = ", episode) # working
                # print(f'current status just after initialization- UAV {net.UAV_age}, BS {net.BS_age}, overall state {net.getstate()}, current reward {net.get_reward()}, done {done}') # working

                if episode%SHOW_EVERY==0 and episode!=1:
                    print(f'Episode - {episode} :  average sum AoI at the BS for the past {SHOW_EVERY} episodes has been {np.mean(age_BS[-SHOW_EVERY:])}\n\n', flush=True)
                    # will show nan for first run as data hasn't yet been added

                # network reset at every start of episode
                # print(f'current state after reset {np.shape(current_state)}') # (10,)

                
                # done will become True when the take_RL_action will return that it has achieved the terminal state, meaning MAX_STEPS
                # if net.episode_step > MAX_STEPS*net.UAV:
                # # each UAV will have MAX_STEPS on their own
                #     done = True
                while not done: 
                    # done will be made true inside the while loop if one UAV crosses MAX_STEPS
                    for UAV in range(net.UAV):
                        # print(f'\n\nstep {step} of episode {episode} for UAV {UAV}') # biplav
                        a = np.random.random()
                        NN_predicted_q = 'random action' # will change to a numerical value if q-value based action chosen
                        arg_max = 'not used' # will change to a numerical value if q-value based action chosen
                        if a > epsilon:
                            NN_predicted_q = agent.get_qs(current_state)
                            action = np.argmax(NN_predicted_q) # getqs ??
                            arg_max = np.argmax(NN_predicted_q)
                            # epsilon is pretty high initially ?? initially take lot of random action
                            # get the NN to predict the current q values (instead of the usual q-table) and take the best action among them
                            # print(f'for {UAV} UAV, epsilon in {step} step is {a} and taking RL action - {action}')
                            
                        else:
                            action = np.random.randint(low = 0, high = net.action_size)
                            # print(f'for {UAV} UAV, epsilon in {step} step is {a} and taking random action - {action}') # working


                        # print(f'current_q as NN predicted {NN_predicted_q}, arg_max = {arg_max}') # working
                        # print(f'action - {action}') # working
                        
                        # actual_action = net.map_actions(action) # not used as action used directly below
                        net.BS_age_prev = copy.deepcopy(net.BS_age) # used for age diff at BS
                        new_state, reward, done = net.take_RL_action(UAV, action, step, reward_fn, episode) 
                        # take RL action means both the random action and argmax action as per Q-eqn
                        # done will be true when last step reached
                        # print(f'reward as per {reward_fn} is {reward}') # biplav

                        # reward for each UAV action is taken for NN working but to record -> once all UAV has taken action, age_BS will be added the average age
                        transition = (current_state, action, reward, new_state, done)
                        
                        # print(f'transition is {transition}') # working # biplav
                        # print(f'UAV_Age={net.UAV_age}, BS_age={net.BS_age}') # biplav
                        # print(f'np.shape(new_info) = {np.shape(new_info)}') # (5,)
                        agent.update_replay_memory(transition)


                        agent.train(done, step)
                        current_state = new_state
                        step += 1

                # print("the current sum AoI at BS is ")
                age_BS.append(net.get_reward("BS-AoI-sum")) # IMPTT - always the sum AOI at BS
                # this doesn't impact learning, just to plot to see the long term AoI at BS. 
                # age_BS should be average age ??
                age_UAV.append(net.get_reward("UAV-AoI-sum")) 

                ep_rewards.append(net.get_reward(reward_fn)) # to see if learning happening
                # print(f'tx_attempts_BS = {net.tx_attempts_BS}')
                # print(f'tx_attempts_UAV = {net.tx_attempts_UAV}')
                # print(f'preference was {net.preference}') # biplav


                if episode == EPISODES: # store the final episode's age to see with average
                    final_age = np.sum(list(net.BS_age.values()))
                
                if not episode % SHOW_EVERY or episode == 1: # for TensorBoard
                    # print("type(age_BS[0]) = ", (type(age_BS[0])))
                    average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
                    min_reward = min(ep_rewards[-SHOW_EVERY:])
                    max_reward = max(ep_rewards[-SHOW_EVERY:])
                    agent.Tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Decay epsilon
                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)

            agent.model.save(f'{folder_name}/{reward_fn}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        elif scheduling == "random":
            for step in range(MAX_STEPS):
                for UAV in range(net.UAV):
                    if step ==1:
                        print("random scheduling for UAV ", UAV)
                    net.take_random_action(UAV, step)
                # store rewards only after all UAVs have acted    
                age_BS.append(net.get_reward("BS-AoI-sum")) # IMPTT - always the sum AOI at BS
                age_UAV.append(net.get_reward("UAV-AoI-sum"))
                # this doesn't impact learning, just to plot to see the long term AoI at BS. 

        pickle.dump(net.tx_attempts_BS, open(folder_name + '/tx_attempts_BS_' + scheduling + "_" + deployment + ".pickle", "wb"))
        pickle.dump(net.tx_attempts_UAV, open(folder_name + '/tx_attempts_UAV_' + scheduling + "_" + deployment + ".pickle", "wb"))
        pickle.dump(net.preference, open(folder_name + '/preference_' + scheduling + "_" + deployment + ".pickle", "wb")) 
        pickle.dump(ep_rewards, open(folder_name + '/ep_rewards_' + scheduling + "_" + deployment + ".pickle", "wb"))
        

    print(f'final episode\'s age = ', final_age, file=open(folder_name + "/results.txt", "a"), flush = True)

    print("deployment =", deployment, ",scheduling = ", scheduling, ", overall average sum BS age = ", np.mean(age_BS), file=open(folder_name + "/results.txt", "a"), flush = True)

    pickle.dump(age_BS, open(folder_name + '/age_BS_' + scheduling + "_" + deployment + ".pickle", "wb")) 
    pickle.dump(age_UAV, open(folder_name + '/age_UAV_' + scheduling + "_" + deployment + ".pickle", "wb"))
    pickle.dump(age_dist_BS, open(folder_name + '/age_dist_BS_' + scheduling + "_" + deployment + ".pickle", "wb"))
    pickle.dump(age_dist_UAV, open(folder_name + '/age_dist_UAV_' + scheduling + "_" + deployment + ".pickle", "wb"))
    # for variable users per drone, all these above codes have to be taken inside the scheduling loop like the tx_attempt pickle

    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    print("execution ended at ", now_str, file=open(folder_name + "/results.txt", "a"), flush = True)
    print("\n\n\n", file=open(folder_name + "/results.txt", "a"), flush = True)
