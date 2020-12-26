from collections import UserList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tqdm
import os
import pickle

import datetime
import copy
import time

# from tf_environment import *
# from comet_ml import Experiment

# experiment = Experiment("HsbMT2nT816RPUXC1LLkVvEe0")

now = datetime.datetime.now()

from create_graph_1 import *
# from path_loss_probability import *
import itertools
from itertools import product  
from tf_reinforce import *
from tf_dqn import *
from tf_c51 import *
from tf_sac import *
from random_scheduling import *
from greedy_scheduling import *
from mad_scheduling import *

import sys

from joblib import Parallel, delayed
import multiprocessing as mp

from parameters import *

random.seed(42)


def distributed_run(arguments):
  
    print(f"passed arguments are {arguments}\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    print(f"passed arguments are {arguments}")

    # pool.starmap(do_scheduling, [(arg[0], arg[1], arg[2]) for arg in arguments]) ## this enable multiprocessing but I am getting memroy allocation and other CUDA related errors with this, so now using sequential execution
    
    for j in arguments:
        do_scheduling(j[0],j[1],j[2])
    
#############################################################

def do_scheduling(deployment, I, scheduler):
    
    
    deployment_options = ["MDS", "RP"]
    scheduler_options  = ["random", "greedy", "MAD", "dqn", "c51"]
    assert(deployment in deployment_options and scheduler in scheduler_options)

    random.seed(42) ## this seed ensures same location of users in every case, keep both seeds
    
    if test_case:
        
        ## exp 15
        print(f"under experiment {experiment}, normal runing", file = open(folder_name + "/results.txt", "a"), flush = True)

        drones_needed           = 1
        users_per_drone         = [8]
        assert drones_needed    ==len(users_per_drone)
        
        drones_coverage         = []
        
        count = 10 # user IDs will start from this. and this also ensured that UAV and users have different IDs. Ensure number of UAVs is less than the count
        for i in range(drones_needed):
            individual_drone_coverage = [x for x in range(count, count+users_per_drone[i])]
            print(f"individual_drone_coverage = {individual_drone_coverage}", file = open(folder_name + "/results.txt", "a"), flush = True)
            count = individual_drone_coverage[-1]+1
            
            drones_coverage.append(individual_drone_coverage)
        
        print(f"drones_coverage = {drones_coverage}", file = open(folder_name + "/results.txt", "a"), flush = True)

        user_list = []
        UAV_list = np.arange(drones_needed)
        for i in drones_coverage:
            for j in i:
                if j!=0: ## user will not contain 0
                    user_list.append(j)
        
        print(f"user_list = {user_list}, UAV_list = {UAV_list}", file = open(folder_name + "/results.txt", "a"), flush = True)
        assert (max(user_list) - min(user_list))+1 == sum(users_per_drone)
        # time.sleep(10)

                    
        if periodic_generation:
            periodicity = {x:random.choice([3,4]) for x in user_list}
        else:
            periodicity = {x:1 for x in user_list}
        
        I = len(user_list) # changed to the needed value

        if packet_loss == True:
            packet_update_loss  = {yy : round(random.random(),2) for yy in user_list}
            packet_sample_loss  = {yy : round(random.random(),2) for yy in user_list}
        else:
            packet_update_loss  = {yy : -1 for yy in user_list}
            packet_sample_loss  = {yy : -1 for yy in user_list}
            
    else: ## user defined UAV and user configuration
                        
        # I is number of users, L length and B breadth
        x_vals = random.sample(range(1, L-1), I) # x-coordinates for users
        y_vals = random.sample(range(1, B-1), I) # y-coordinates for users
        z_vals = [0]*I

        user_coordinates = list(zip(x_vals,y_vals))

        x_grid_nos = int(L/r) + 1 # number of different values the grid takes for x axis
        y_grid_nos = int(B/r) + 1 # number of different values the grid takes for y axis

        grid_x = np.linspace(0, L, num = x_grid_nos) # generate evenly spaced x positions for grid
        grid_y = np.linspace(0, B, num = y_grid_nos) # generate evenly spaced y positions for grid
        
        grid_coordinates = list(itertools.product(grid_x , grid_y))

        drones_needed, drones_coverage = create_graph_1(user_coordinates, grid_coordinates, deployment)       
    
        user_list = [] ## this is not the same user_list as defined in the environment, this is just used to index the packet loss and sample loss
        UAV_list  = np.arange(drones_needed)
        
        for i in drones_coverage:
            for j in i:
                if j!=0:
                    user_list.append(j)
                    
         
        if periodic_generation:
            periodicity = {x:random.choice([3,4]) for x in user_list}
        else:
            periodicity = {x:1 for x in user_list}


        if packet_loss == True:
            packet_update_loss = {yy : round(random.random(),2) for yy in user_list}
            packet_sample_loss = {yy : round(random.random(),2) for yy in user_list}
        else:
            packet_update_loss = {yy : -1 for yy in user_list}
            packet_sample_loss = {yy : -1 for yy in user_list}
            

    print(f"\n\n{deployment} deployment for {I} users under {scheduler} scheduling\n", file=open(folder_name + "/results.txt", "a"), flush=True)

            
    print(f'Under test_case = {test_case}, drones_needed = {drones_needed}, UAV_list = {UAV_list}, drones_coverage = {drones_coverage}, user_list = {user_list}, periodicity = {periodicity} for {deployment} deployment for {I} users under {scheduler} scheduling, update loss = {packet_update_loss}, sampling loss = {packet_sample_loss}, user_list = {user_list}, UAV_list = {UAV_list}, CSI_as_state = {CSI_as_state}, sample_error_in_CSI = {sample_error_in_CSI}\n', file=open(folder_name + "/results.txt", "a"), flush=True)  
    

    str_x = str(deployment) + " placement with " + str(I) + " users needs " + str(scheduler) + " scheduler and "  + str(drones_needed) + " drones\n"
    print(f'{str_x}', file=open(folder_name + "/drones.txt", "a"), flush=True)
    
    
    if scheduler == "greedy":
        t1 = time.time()
        #with tf.device('/CPU:0'):
        greedy_overall[I], greedy_final[I], greedy_all_actions[I] = greedy_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity)  
        t2 = time.time()
        print("greedy for ", I, " users took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
        pickle.dump(greedy_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_overall.pickle", "wb")) 
        pickle.dump(greedy_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_final.pickle", "wb"))
        pickle.dump(greedy_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "_greedy_all_actions.pickle", "wb")) 
    
    if scheduler == "random":
        t1 = time.time()
        #with tf.device('/CPU:0'):
        random_overall[I], random_final[I], random_all_actions[I] = random_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity)
        t2 = time.time()
        print("random for ", I, " users took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
        pickle.dump(random_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_overall.pickle", "wb")) 
        pickle.dump(random_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_final.pickle", "wb")) 
        pickle.dump(random_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_all_actions.pickle", "wb")) 
        
        
    if scheduler == "MAD":
        t1 = time.time()
        #with tf.device('/CPU:0'):
        mad_overall[I], mad_final[I], mad_all_actions[I] = mad_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity)
        t2 = time.time()
        print("MAD for ", I, " users took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
        pickle.dump(mad_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_overall.pickle", "wb")) 
        pickle.dump(mad_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_final.pickle", "wb"))
        pickle.dump(mad_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_all_actions.pickle", "wb")) 
        
    

    if scheduler == "dqn":
        t1 = time.time()
        dqn_overall[I], dqn_final[I], dqn_all_actions[I] = tf_dqn(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity)
        t2 = time.time()
        print("DQN for ", I, " users took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
        pickle.dump(dqn_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_overall.pickle", "wb")) 
        pickle.dump(dqn_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_final.pickle", "wb"))
        pickle.dump(dqn_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_all_actions.pickle", "wb"))


    if scheduler == "c51":
        t1 = time.time()
        c51_overall[I], c51_final[I], c51_all_actions[I] = tf_c51(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity)
        t2 = time.time()
        print("c51 for ", I, " users took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
        pickle.dump(c51_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_overall.pickle", "wb")) 
        pickle.dump(c51_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_final.pickle", "wb")) 
        pickle.dump(c51_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_all_actions.pickle", "wb")) 
        

    print(f"{I} users under {scheduler} scheduling and {deployment} placement are over\n\n", file=open(folder_name + "/results.txt", "a"), flush=True)
    print(f"{I} users under {scheduler} scheduling and {deployment} placement are over\n\n")

#############################################################
    
if __name__ == '__main__':
        

    now_str_1 = now.strftime("%Y-%m-%d %H:%M")
    folder_name = 'models/' +  now_str_1
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n")
    
    folder_name_MDS = folder_name + "/MDS"
    folder_name_random = folder_name + "/RP" ## RP means random placement

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        os.makedirs(folder_name_MDS)
        os.makedirs(folder_name_random) 
        
    print("execution started at ", now_str_1, file = open(folder_name + "/results.txt", "a"), flush = True)

    print("num_iterations = ",num_iterations, ", random_episodes = ", random_episodes,", BS_capacity = ", BS_capacity, ", UAV_capacity = ", UAV_capacity,",  MAX_STEPS = ", MAX_STEPS, " gamma = ", set_gamma, ", learning_rate = ", learning_rate, ", fc_layer_params = ", fc_layer_params, ", replay_buffer_capacity = ", replay_buffer_capacity, ", coverage_capacity = ", coverage_capacity, ", L = ", L, ", B = ", B, ", R = ", R, ", r = ", r,  "\n", file = open(folder_name + "/results.txt", "a"), flush = True)

    deployments = ["RP"] #, "RP"] #, "MDS"]
    
    schedulers  = ["MAD", "dqn", "random", "greedy"] ##     scheduler_options  = ["random", "greedy", "MAD", "dqn", "c51"]
    
    limit_memory = False ## enabling this makes the code notbeing able to find CUDA device
    
#############################

    experiment = 1

    if experiment == 1:
        test_case           = True
        packet_loss         = False
        periodic_generation = False

    elif experiment == 2:
        test_case           = True
        packet_loss         = True
        periodic_generation = False
        
    elif experiment == 3:
        test_case           = False
        packet_loss         = False
        periodic_generation = False

    elif experiment == 4:
        test_case           = False
        packet_loss         = True
        periodic_generation = False

    elif experiment == 5:
        test_case           = True
        packet_loss         = False
        periodic_generation = True

    elif experiment == 6:
        test_case           = True
        packet_loss         = True
        periodic_generation = True
        
    elif experiment == 7:
        test_case           = False
        packet_loss         = False
        periodic_generation = True

    elif experiment == 8:
        test_case           = False
        packet_loss         = True
        periodic_generation = True
    
    if test_case:
        users = [8] ## will get changed accordingly inside the loop above
    else:
        users = [8,10]

#############################

    
    arguments = list(itertools.product(deployments, users, schedulers)) ## deployment, I, scheduler
    
    dqn_overall = {}
    dqn_final = {}
    dqn_all_actions = {}
    
    c51_overall = {}
    c51_final = {}
    c51_all_actions = {}
    
    reinforce_overall = {}
    reinforce_final = {}
    reinforce_all_actions = {}
    
    random_overall = {} ## sum of age at BS for all of the MAX_STEPS time steps
    random_final   = {} ## sum of age at BS for step =  MAX_STEPS i.e. last time step
    random_all_actions = {}
    
    greedy_overall = {}
    greedy_final   = {}
    greedy_all_actions = {}
    
    mad_overall = {}
    mad_final   = {}
    mad_all_actions = {}

    pool = mp.Pool(mp.cpu_count())
    print(f"pool is {pool} \n\n", file = open(folder_name + "/results.txt", "a"))
    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}", file = open(folder_name + "/results.txt", "a"))
    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}")

    distributed_run(arguments)

    pool.close()    
