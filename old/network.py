import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import combinations

from create_graph_1 import *
from path_loss_probability import *
from age_calculation import *
from itertools import product  
# from utils import  *
from drl import *
import scipy
import pickle

# random.seed(3)

xx = 0 # user whose age will be shown, made a global variable

MAX_STEPS = 5 #20 # 100 # step for each UAV per episode

class Network:
    def __init__(self, n_users, coverage_capacity):
        '''
        param n_users        : int, number of users
        param L              : int, length of area in meters
        param B              : int, breadth of area in meters
        param BW             : int, bandwidth
        param UAV_capacity   : int, number of users UAV can support in the uplink and backhaul. same for now.
        param r              : int, grid intervals 
        param R              : float, UAV coverage, edge between UAV and user exists only if their distance less than R
        user_locs            : list, locations of the users
        grid                 : list, grid points where the UAVs can be deployed
        UAV_loc              : list, UAV deployment positions
        cover                : list of list, list of list containing users covered by the drone in the index position
        UAV                  : int, number of UAV needed
        UAV_age              : dict, age at UAV
        BS_age               : dict, age at BS
        BS_age_prev          : dict, age at BS in the previous step
        state                : list, state of the system - contains all ages at BS and UAV
        # reward               : float, negative of average of age at BS 
        agent                : Object class, the DL agent that will be shared among all the UAVs
        actions              : list, possible actions
        coverage_cap         : int, max users one UAV can give provide coverage
        action_size          : int, number of possible actions
        episode_step         : int, current step of the ongoing episode. One episode gets over after MAX_STEP   steps
        tx_attempts_BS       : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was updated
        tx_attempts_UAV      : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was sampled
        preference           : dict, at each index indicated by action is an array and the array has episode wise count of how many times the action was selected. Analogous to visualizing the q-table



        '''
        self.n_users        = n_users
        self.list_of_users  = []
        self.L              = 21
        self.B              = 21
        self.UAV_capacity   = 1
        self.r              = 20  
        self.R              = round(r*(2**0.5), 3)
        self.user_locs      = []
        self.grid           = []
        self.UAV_loc        = []
        self.cover          = []
        self.UAV            = 0 # initialized to 0 but updated in create_topology()
        self.UAV_age        = {}
        self.BS_age         = {}
        self.BS_age_prev    = {}
        self.state          = np.array([])
        # self.reward         = 0
        self.actions        = [] # initialized once the coverage is calculated
        self.coverage_cap   = coverage_capacity
        self.action_size    = int(2*scipy.special.comb(coverage_capacity, self.UAV_capacity)) # coverage_capacity in create_graph_1
        self.episode_step   = 0
        # self.agent          = DLModel(self.max_coverage(), self.action_size)
        self.tx_attempts_BS = {}
        self.tx_attempts_UAV= {}
        self.preference     = {}

    def create_user_locations(self):
        '''
        generate user locations randomly in the given area
        '''

        x_vals = random.sample(range(1, self.L-1), self.n_users) # x-coordinates for users
        y_vals = random.sample(range(1, self.B-1), self.n_users) # y-coordinates for users
        z_vals = [0]*self.n_users

        self.user_locs = list(zip(x_vals,y_vals))

    def create_grid_locations(self):
        '''
        generate the grid over the network and these are the potential UAV deployment locations
        '''

        x_grid_nos =(self.L/self.r) + 1 # number of different values the grid takes for x axis
        y_grid_nos = (self.B/self.r) + 1 # number of different values the grid takes for y axis

        grid_x = np.linspace(0, self.L, num = x_grid_nos) # generate evenly spaced x positions for grid
        grid_y = np.linspace(0, self.B, num = y_grid_nos) # generate evenly spaced y positions for grid
 
        self.grid = list(product(grid_x , grid_y)) 
    
    def create_topology(self, str):
        '''
        creates the networkx graph based on random or MDS UAV deployment to cover all users
        str = "random" or "MDS"
        '''
        if str=="random":
            self.UAV, self.cover = create_graph_1(self.user_locs, self.grid, str)
        
        if str=="MDS":
            self.UAV, self.cover = create_graph_1(self.user_locs, self.grid, str)

    def initialize_age(self):
        self.list_of_users = sorted([x for y in self.cover for x in y])
        for i in self.list_of_users:
            # initial age put 1 and not 0 as if 0, in first time step whethere sampled or not, all users age at UAV becomes 1 but for 1, it is different - 2 for not sampled and 1 for sampled
            self.UAV_age[i] = 1
            self.BS_age[i]  = 1

            self.BS_age_prev[i] = 1 # special case for first step ??
            
    def create_action_space(self):
        '''
        for 1 UAV once the coverage has been decided, create the action space
        '''
        # sampled_user = random.sample(self.list_of_users, self.coverage_cap)
        # sampled_user = random.sample(self.list_of_users, self.coverage_cap)

        # has to be changed for multiple users ?? drones user or overall user. 
        sampled_user_possibilities = list(combinations(self.list_of_users, self.UAV_capacity))
        updated_user_possibilities = list(combinations(self.list_of_users, self.UAV_capacity))
        self.actions = list(product(sampled_user_possibilities, updated_user_possibilities))
        self.action_size = len(self.actions)
        print(f'action space is {self.actions}, number of actions are {(self.action_size)}')

    def start(self, str):
        self.create_user_locations()
        self.create_grid_locations()
        self.create_topology(str)
        self.initialize_age()
        self.create_action_space()

    def max_coverage(self):
        '''
        get the maximum number of users covered by any UAV, needed to decide the number of states and hence the NN first layer size
        '''
        users_per_drone = []
        for i in self.cover:
            users_per_drone.append(len(i))
        max_users =  max(users_per_drone)
        return int(max_users)

    def getstate(self): # 
        # doesn't change anything, just returns the current state. Ages have been updated in the take_RL_action, here the new state is returned
        state_UAV = np.array(list(self.UAV_age.values()))
        state_BS  = np.array(list(self.BS_age.values()))
        self.state = np.concatenate((state_UAV, state_BS), axis=None) 
        # print(f'UAV_age = {self.UAV_age}, BS_age = {self.BS_age}') # biplav
        # print(f'inside getstate state = {self.state}') # biplav
        # type(self.state) = {type(self.state)}')
        return (self.state)
        

    def take_RL_action(self, UAV, action, step, reward_fn, episode):
        # called per UAV so no don't loop over all the UAVs
        '''
        param UAV     - int, the UAV which is taking the action
        param action  - tuple, the action that the UAV is taking. action means whom to sample and update
        param step    - int, step of the current episode. needed for special case of step=0 when no update
        '''
        # print("action is ", action)
        self.preference[action][episode-1] = self.preference[action][episode-1] + 1
        sampled_users = list(self.actions[action][0])
        updated_users = list(self.actions[action][1])
        # print(f'action is {action}, sampled users are {sampled_users} and updated_users are {updated_users}') # working # biplav
        self.episode_step +=1
        # this episode_step will track the steps and return True if steps exceed MAX_STEPS
        users_covered = self.cover[UAV]

        # unlike in random scheduling, here the sampled and updated users are already decided so no extra code for that

        # biplav
        # print("step = ", step, " results before action follows")
        # print("sampled_users = ", sampled_users)
        # print("Age at UAV ", UAV," is ", self.UAV_age)
        # print("updated_users = ", updated_users)
        # print("Age at BS ", self.BS_age)

        assert self.episode_step == step # because step starts from 0 and episode_step incremented starts from 0 but incremented once before this line

        if step==1: # step 1 so BS has nothing to get from UAV
            for i in users_covered:
                self.BS_age[i] = self.BS_age[i]+1

        else:
            for i in users_covered:
                if i in updated_users:
                    # print("user ", i, " was updated")
                    self.BS_age[i] = self.UAV_age[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                    self.tx_attempts_BS[i][episode-1] = self.tx_attempts_BS[i][episode-1] + 1
                else:
                    # print("user ", i, " was not updated")
                    self.BS_age[i] = self.BS_age[i] + 1

        for i in users_covered:
            if i in sampled_users:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
                self.tx_attempts_UAV[i][episode-1] = self.tx_attempts_UAV[i][episode-1] + 1
            else:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = self.UAV_age[i] + 1

        # if step==0: # fix an user for the entire simulation to track
        #     global xx
        #     xx = np.random.choice(users_covered)

        ### to see age progression, comment out the following lines # biplav
        # print("step = ", step, " results after action follows")
        # print("sampled_users = ", sampled_users)
        # print("Age at UAV ", UAV," is ", self.UAV_age)
        # print("updated_users = ", updated_users)
        # print("Age at BS is", self.BS_age)


        # print("results for user ", xx, " will be shown ")
        # check age evolution of user at each time step
        # if xx in sampled_users:
        #     print("step = ", step, ", user ", xx, " was sampled and age at UAV is ", self.UAV_age[xx])
        # else:
        #     print("step = ", step, ", user ", xx, " NOT sampled and age at UAV is ", self.UAV_age[xx])

        # if xx in updated_users:
        #     print("step = ", step, ", user ", xx, " was updated and age at  BS is ", self.BS_age[xx])
        # else:
        #     print("step = ", step, ", user ", xx, " NOT updated and age at  BS is ", self.BS_age[xx])
        
        
        if self.episode_step == MAX_STEPS*self.UAV: # each UAV will take MAX_STEPS, need to change for multiple UAVs ??
            done = True
            # print("done for step") # working
        else:
            done = False

        # net.UAV_age and net.BS_age already updated in the above lines
        self.getstate() # will update net.state, also returns the new state but that is not needed
        
        reward = self.get_reward(reward_fn)
        ## reward whole system age or only age of covered users ?? only covered users as one drone's action doesn't affect other drone's users ??
        
        return (self.state, reward, done)


    def take_random_action(self, UAV, step): 
        '''
        # this random is not the exploratory random of RL, it is the random scheduling without any objective. Will be compared to DRL based scheduling
        param UAV    - int, ID of the UAV
        param step   - int, analogous to time step 
        '''
        # print("random action for UAV ", UAV, " follows ")
        sampled_users = [] # list of users sampled by the UAV
        updated_users = [] # list of users updated by the UAV

        # print("step = ", step, " results before action follows")
        # print("sampled_users = ", sampled_users)
        # print("Age at UAV ", UAV," is ", self.UAV_age)
        # print("updated_users = ", updated_users)
        # print("Age at BS ", self.BS_age)

        # sampling by UAV part starts

        users_covered = self.cover[UAV] # list containing all users covered by the passed UAV
        if len(users_covered) <= self.UAV_capacity: # all users under drone j is sampled
            # print('for drone ', j,", covered users = ", len(users_covered), " which is less than UAV capacity of ", UAV_capacity)
            for user in users_covered:
                sampled_users.append(user)
                # print("sampled_users = ", sampled_users)
        
        else: 
            to_sample = random.sample(users_covered, self.UAV_capacity) # selected users to sample
            for user in to_sample:
                sampled_users.append(user)
                    # print("sampled_users = ", sampled_users)

        # sampling by UAV part ends

        # print("sampling process for each drone done, sampled users are ", sampled_users)
        # if time==run_time-1:
        #     print(len(sampled_users)," users sampled out of ", users, " in the random sampling and updating policy")
        # every drone has added which users to sample in sampled_users, next is to select which user will get updated

        # updating to BS starts

        if step==1:
            for i in users_covered:
                self.BS_age[i] = self.BS_age[i]+1 #  # at the first slot no user is updated and age becomes 2 at the BS for all of the users, see table 1 of paper - Optimal Scheduling Policy for Minimizing Age of Information with a Relay


        else: 
            if len(users_covered) <= self.UAV_capacity: # all users under drone j is updated
                for user in users_covered:
                    updated_users.append(user)
        
            else: # all users can't be served only UAV_capacity users can be served
                to_update = random.sample(users_covered, self.UAV_capacity) # selected users to sample
                for user in to_update:
                    updated_users.append(user)

            # age at BS is updated before age at UAV as age at BS of the next slot depends on the age at UAV of current slot, and updating age at UAV before will overwrite the original value
            for i in users_covered:
                if i in updated_users:
                    # print("user ", i, " was updated")
                    self.BS_age[i] = self.UAV_age[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                else:
                    # print("user ", i, " was not updated")
                    self.BS_age[i] = self.BS_age[i] + 1

        # updating to BS ends

        # age at BS is updated before age at UAV as age at BS of the next slot depends on the age at UAV of current slot, and updating age at UAV before will overwrite age at UAV

        # print("updating process for each drone done, updated users are ", updated_users)
        # if time==run_time-1:
        #     print(len(updated_users)," users updated] out of ", users, " in the random sampling and updating policy")
        # which of these two will happen first ? seems like updating has to change first ??

        for i in users_covered:
            if i in sampled_users:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
            else:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = self.UAV_age[i] + 1

        # fix a user and show his age at each stage
        if step==1:
            global xx
            xx = np.random.choice(users_covered)

        # ### to see age progression, comment out the following lines
        # print("step = ", step, " results after action follows")
        # print("sampled_users = ", sampled_users)
        # print("Age at UAV ", UAV," is ", self.UAV_age)
        # print("updated_users = ", updated_users)
        # print("Age at BS ", self.BS_age)


        # print("results for user ", xx, " will be shown ")
        # # check age evolution of user at each time step
        # if xx in sampled_users:
        #     print("step = ", step, ", user ", xx, " was sampled and age at UAV is ", self.UAV_age[xx])
        # else:
        #     print("step = ", step, ", user ", xx, " NOT sampled and age at UAV is ", self.UAV_age[xx])

        # if xx in updated_users:
        #     print("step = ", step, ", user ", xx, " was updated and age at  BS is ", self.BS_age[xx])
        # else:
        #     print("step = ", step, ", user ", xx, " NOT updated and age at  BS is ", self.BS_age[xx])
        


    def map_actions(self, action):  # not used
        '''
        convert the single integer action to specific sampling and updating tasks
        '''
        actual_action = self.actions[action]
        # print(f'action space is {self.actions}, selected action is {action} which maps to {actual_action}')
        return actual_action

    def reset_network(self):
        # when an episode starts, reset the network
        self.initialize_age()
        self.episode_step = 0
        return self.getstate()

    def get_reward(self, reward_fn):
        # doesn't change anything, just returns
        # averages are redundant, behavior same as sum
        '''
        options for reward_fn
        BS-AoI-diff     -> difference of previous AoI and current AoI at BS
        BS-AoI-sum      -> sum of current AoI at BS                         ; DONE
        BS-AoI-avg      -> avg of current AoI at BS                         ; DONE
        BS-NegAoI-sum   -> negative of sum of current AoI at BS             ; DONE
        BS-NegAoI-avg   -> negative of avg of current AoI at BS             ; DONE
        AoI-sum         -> sum of current AoI at BS and UAV
        AoI-avg         -> avg of current AoI at BS and UAV
        Neg-AoI-sum     -> negative of sum of current AoI at BS and UAV
        Neg-AoI-avg     -> negative of avg of current AoI at BS and UAV

        '''
        # print(f'BS_age = {list(self.BS_age.values())}')
        # if reward_fn == "BS-NegAoI-avg":
        #     award = -np.average(list(self.BS_age.values()))
            # print(f'awards selected via BS-NegAoI-avg')

        if reward_fn == "BS-NegAoI-sum":
            award = -np.sum(list(self.BS_age.values()))
            # print(f'awards selected via BS-NegAoI-sum')

        elif reward_fn == "BS-AoI-sum":
            award = np.sum(list(self.BS_age.values()))
            # print(f'awards selected via BS-AoI-sum')

        # elif reward_fn == "BS-AoI-avg":
        #     award = np.average(list(self.BS_age.values()))
        #     # print(f'awards selected via BS-AoI-avg')

        elif reward_fn == "AoI-sum":
            award = np.sum(list(self.state))
            # print(f'awards selected via AoI-sum')

        # elif reward_fn == "AoI-avg":
        #     award = np.avg(list(self.state))
        #     # print(f'awards selected via AoI-avg')

        elif reward_fn == "Neg-AoI-sum":
            award = -np.sum(list(self.state))
            # print(f'awards selected via Neg-AoI-sum')

        # elif reward_fn == "Neg-AoI-avg":
        #     award = -np.avg(list(self.state))
        #     # print(f'awards selected via Neg-AoI-avg')

        elif reward_fn == "BS-AoI-diff":
            award = np.sum(list(self.BS_age_prev.values())) - np.sum(list(self.BS_age.values()))
            # print(f'awards selected via BS-AoI-diff')

        elif reward_fn == "UAV-AoI-sum":
            award =  np.sum(list(self.UAV_age.values()))
            # print(f'awards selected via UAV-AoI-sum')

        elif reward_fn == "UAV-NegAoI-sum":
            award =  -np.sum(list(self.UAV_age.values()))
            # print(f'awards selected via UAV-NegAoI-sum')


        # print(f'self.BS_age_prev = {self.BS_age_prev}, self.BS_age = {self.BS_age}')
        # print(f'in get_reward, reward is {award}')
        return (award) #*1.00/len(BS_age)
        # working for random scheduling






        

