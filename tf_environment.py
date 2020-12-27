from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

import random
from itertools import combinations

# from path_loss_probability import *
# from age_calculation import *
# from drl import *

from itertools import product  
import itertools
from create_graph_1 import *

import time

import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import copy
import base64
# import imageio

import matplotlib.pyplot as plt
import os
# import reverb
import tempfile
import functools
import operator

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network


tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

tempdir = tempfile.gettempdir()
random.seed(42)

################## PARAMETERS NEEDED TO BE CONFIGURED EVERY RUN

## shell files ntasks = |users| * |scheduler| * |placement| = 4*6*2 = 48, use 4 nodes, in scheduler only consider the ML ones. greedy mad and random are done in less than a minute
limit_memory = False
verbose = False
# verbose = True
comet = True
# comet = True
CSI_as_state = False
sample_error_in_CSI = False ## has no meaning if CSI_as_state=False; if CSI_as_state True, this True will mean only UAV CSI is included in state

MAX_STEPS = 10 # 20  

set_gamma = 0.99
UAV_capacity = 5 # L, sample

BS_capacity = 3 # K, update
random_episodes = 100_000
coverage_capacity = 3 # max users 1 UAV can cover, used in create_graph_1

num_iterations = 1_000_000
#@param {type:"integer"} # number of times collect_data is called, log_interval and eval_interval are used here. number of times the collect_episodes(...) will run. each collect_episode(...) runs for collect_episodes_per_iteration episodes to fill the buffer. once one iteration is over, the train_env is run on it and then buffer is clear. This value doesn't add to the returns that is showed as the final performance.

# if packet failure occurs independently at each user-UAV and UAV_BS link, the greedy will still keep targetting the most aged user and therefore greedy will still perform well which can be seen in Optimal Scheduling Policy for Minimizing Age of Information with a Relay. Eg with a failure rate 0.5, greedy will keep targetting the most aged which might still fail and it could have been able to reduce the age by targetting some other user. Not sure what will happen with failure rates, lets see.

fc_layer_params = (1024, 1024) # for 5 user 2 SC (32,16) # for 3 user 1 SC


################### DON'T CHANGE THESE    

## https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"GPUs are {gpus}")
if limit_memory:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
# https://www.tensorflow.org/guide/gpu
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("\n\n",len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU\n\n")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(f"\n\nNo GPU\n\n")
    pass

##############

collect_episodes_per_iteration = 1 # @param {type:"integer"} # collect_episode runs this number of episodes per iteration

log_interval = 10_000 # @param {type:"integer"} # how frequently to print out in console

num_eval_episodes = 1 ## @param {type:"integer"} # this decides how many times compute_avg_return will run. compute_avg_return runs first in the beginning without any training to show random actions. Then collect_episode(...) starts filling the buffer and agent starts training. And in the training process if the current iteration % eval_interval, compute_avg_return(...) runs for num_eval_episodes on eval_env to see append the new rewards. Every time this is run, returns is appended and this is what is shown as the final performance

eval_interval = 100 #100 # @param {type:"integer"} # # compute_avg_return called every eval_interval, i.e avg_return calculated filled every eval_interval. this is what is shown in plot 

learning_rate = 1e-3 # @param {type:"number"}

replay_buffer_capacity = 200_000 # @param {type:"integer"} value of max_length, same for both


class UAV_network(py_environment.PyEnvironment):   # network of UAVs not just a single one
    
    def __init__(self, n_users, coverage, name, folder_name, packet_update_loss, packet_sample_loss, periodicity): # number of total user and the user-coverage for every UAV
        
        '''
        param n_users        : int, no of users.
        param n_UAVs         : int, number of UAVs in this scenario.
        param coverage       : list of lists, each list has users covered by the drone indicated by the index of the sublist.
        param user_list      : list, user list of all users.
        param UAV_list       : list, generic UAV list. e.g. for 2 UAV [1, 2].
        param users_UAVs     : dict, will contain the users covered by each jth UAV at jth index. user_UAVs[j+1] = self.coverage[j].
        param act_coverage   : dict, same as users_UAVs but 0s removed. actual coverage.
        param BW             : int, bandwidth.
        param UAV_capacity   : int, number of users UAV can support in the uplink.
        param BS_capacity    : int, number of UAVs BS can support in the backhaul.
        user_locs            : list, locations of the users.
        grid                 : list, grid points where the UAVs can be deployed.
        UAV_loc              : list, UAV deployment positions.
        cover                : list of list, list of list containing users covered by the drone in the index position.
        UAV_age              : dict, age at UAV.
        BS_age               : dict, age at BS.
        BS_age_prev          : dict, age at BS in the previous step.
        state                : list, state of the system - contains all ages at BS and UAV.
        agent                : Object class, the DL agent that will be shared among all the UAVs.
        actions              : list, possible actions.
        action_size          : int, number of possible actions.
        current_step         : step of the ongoing episode
        episode_step         : int, current step of the ongoing episode. One episode gets over after MAX_STEP number of steps. Note difference with current_step
        preference           : dict, at each index indicated by action is an array and the array has episode wise count of how many times the action was selected. Analogous to visualizing the q-table.
        name                 : string, distinguish between eval and train networks.
        age_dist_UAV         : dict, stores the episode ending age at UAV per user.
        age_dist_BS          : dict, stores the episode ending age at BS per user.
        tx_attempt_BS        : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was updated.
        tx_attempt_UAV       : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was sampled.
        
        attempt_sample       : list, index is the episode and the value is the number of times a sample attempt was made. since each sampling results in 1 packet, this value is the number of users selected to sample
        success_sample       : list, index is episode and value is the number of sampling attempts that were successful
        
        attempt_update       : list, index is the episode and the value is the number of times a sample attempt was made. since each sampling results in 1 packet, this value is the number of users selected to sample
        success_update       : list, index is episode and value is the number of sampling attempts that were successful
        
        
        
        '''
        
        self.n_users        = n_users
        self.periodicity    = periodicity
        self.coverage       = coverage
        self.n_UAVs         = len(coverage)
        self.UAV_capacity   = UAV_capacity
        self.BS_capacity    = BS_capacity
        self.users_UAVs     = {} # [i for i in range(1, self.n_users + 1)] # updated in start_network()
        self.act_coverage   = {} # updated in start_network()
        self.user_locs      = []
        self.grid           = []
        self.UAV_loc        = []
        self.cover          = []
        self.actions_space  = [] # initialized once the coverage is calculated
        self.action_size    = 1 # will be updated in start_network()
        self.episode_step   = 0
        self.preference     = {}
        self.current_step   = 1
        self.UAV_age        = {}
        self.BS_age         = {}
        self.BS_age_prev    = {}
        self.name           = name
        self.age_dist_UAV   = {}
        self.age_dist_BS    = {}    
        self.tx_attempt_BS  = {}
        self.tx_attempt_UAV = {}
        self.user_list      = []
        self.UAV_list       = []
        self.folder_name    = folder_name
        self.update_loss    = {} ## the dicts will be initialized in start_network
        self.sample_loss    = {}
        
        self.attempt_sample = []
        self.success_sample = []
        self.attempt_update = []
        self.success_update = []

        self.start_network(packet_update_loss, packet_sample_loss)
        
        # print(f"sample_loss = {self.sample_loss}, update_loss = {self.update_loss}")

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.action_size-1, name='sample_update')  # bounds are inclusive
        
        if CSI_as_state:
            if sample_error_in_CSI:
                self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + 3*n_users + len(self.UAV_list)), dtype=np.float32, minimum=0, maximum=MAX_STEPS+2, name='current_state')
                self._state = np.concatenate((list([self.current_step]), [1]*2*(n_users), list(self.sample_loss.values()), list(self.update_loss.values())), axis=None) #
            else:
                self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + 2*n_users + len(self.UAV_list)), dtype=np.float32, minimum=0, maximum=MAX_STEPS+2, name='current_state')
                self._state = np.concatenate((list([self.current_step]), [1]*2*(n_users), list(self.update_loss.values())), axis=None) #
            
        else:
            self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + 2*n_users), dtype=np.float32, minimum=0, maximum=MAX_STEPS+2, name='current_state')
            self._state = np.concatenate((list([self.current_step]), [1]*2*(n_users)), axis=None) #

        # if verbose:
        # print(f"initial state is {self._state} with length {np.shape(self._state)} when CSI_as_state = {CSI_as_state} and sample_error_in_CSI = {sample_error_in_CSI}")  
              
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    
    def update_act_coverage(self):
        ## remove the padding from the coverage 0s so that we get the actual coverage
    
        n = [i for i in list(self.users_UAVs.keys())] # UAV ids
        k = [i for i in list(self.users_UAVs.values())] # all users
        b = {} # this will be the act_coverage

        for i in range(len(self.coverage)): # for each UAV ## don't make it UAV_list as it has not been made yet
            old_list = self.users_UAVs[i] # users under UAV i
            new_list = [j for j in old_list if j!=0] # 0s are padding not actual user. user ID cannot be 0, drone ID can be 0
            b[i] = new_list

        # if verbose:
        #     print("n = ", n , "k = ", k, "b = ", b)
        return b
    
    def start_network(self, packet_update_loss, packet_sample_loss):
        
        for i in range(len(self.coverage)): ## don't make it UAV_list as it has not been made yet
            self.users_UAVs[i] = self.coverage[i]
            # index of users_UAVs will start from 0
            
        self.act_coverage = self.update_act_coverage()   
        self.user_list      = functools.reduce(operator.iconcat, list(self.act_coverage.values()), []) # list(self.act_coverage.values())
        self.UAV_list       = list(self.act_coverage.keys())
        
        if verbose:
            print(f"user list = {self.user_list}")
            print(f"UAV list  = {self.UAV_list}")
        
        self.update_loss    = packet_update_loss
        self.sample_loss    = packet_sample_loss


        if verbose:
            print(f'self.n_users = {self.n_users}, self.n_UAVs = {self.n_UAVs}, self.act_coverage = {self.act_coverage}, self.update_loss = {self.update_loss}, self.sample_loss = {self.sample_loss}, self.UAV_list = {self.UAV_list}, self.user_list = {self.user_list}')
            # time.sleep(15)

        self.create_action_space()

        # not doing in initialize_age() as initialize_age() is run every time net is reset so older values will be lost. start_network is run only once
        for i in self.user_list:
        
            self.age_dist_UAV[i]   = [] # ending age of every episode, updated in _reset
            self.age_dist_BS[i]    = []
            
            self.tx_attempt_BS[i]  = [] # number of attempts per episode, initialize in _reset and updated in _step
            self.tx_attempt_UAV[i] = []   
            
        for i in range(self.action_size):
            self.preference[i] = []
            
            
        if verbose:
            print(f'\n{self.name} started and age_dist_UAV = {self.age_dist_UAV}, age_dist_BS = {self.age_dist_BS}, tx_attempt_BS = {self.tx_attempt_BS}, tx_attempt_UAV = {self.tx_attempt_UAV}\n')

    
    def initialize_age(self):
        
        # before initializing, save the info    
        # for the first time it is run, BS_age and others haven't even been initialized
        
        if self.episode_step!=1: ## why !=1 ??
            
            self.attempt_sample.append(0)
            self.success_sample.append(0)
            
            self.attempt_update.append(0)
            self.success_update.append(0)
            
            
            for i in self.user_list:
                
                self.age_dist_UAV[i].append(self.UAV_age[i])
                self.age_dist_BS[i].append(self.BS_age[i])
                
                self.tx_attempt_BS[i].append(0) # 0 will be changed in _step for every attempt
                self.tx_attempt_UAV[i].append(0)
                
            for i in range(self.action_size):
                self.preference[i].append(0) 
                
            
            if verbose:
                print(f'\n{self.name} just before reset of {self.name} the age at UAV = {self.UAV_age}, age at BS = {self.BS_age} and these have used to update age_dist_UAV = {self.age_dist_UAV} and age_dist_BS = {self.age_dist_BS}\n')
                print(f'\n{self.name} in the same reset block, tx_attempts have been updated as tx_attempt_UAV = {self.tx_attempt_UAV} and tx_attempt_BS = {self.tx_attempt_BS}\n')
                # print(f"{self.name} preference is {self.preference}")
        
        for i in self.user_list:
            # initial age put 1 and not 0 as if 0, in first time step whethere sampled or not, all users age at UAV becomes 1 but for 1, it is different - 2 for not sampled and 1 for sampled
            self.UAV_age[i] = 1
            self.BS_age[i]  = 1
            self.BS_age_prev[i] = 1 # special case for first step ??

    def _reset(self):
        self.episode_step +=1
            
        if CSI_as_state: # csi as state needed
            if sample_error_in_CSI: # sampling error as state needed  
                self._state = np.concatenate((list([self.current_step]), [1]*2*(len(self.user_list)), list(self.sample_loss.values()), list(self.update_loss.values())), axis=None)
            else:
                self._state = np.concatenate((list([self.current_step]), [1]*2*(len(self.user_list)), list(self.update_loss.values())), axis=None)
        else:
            self._state = np.concatenate((list([self.current_step]), [1]*2*(len(self.user_list))), axis=None)
            
        self._episode_ended = False
        self.current_step = 1
        if verbose:
            print(f'\n{self.name} after reset, episode {self.episode_step} begins with self._state = {self._state} with shape {np.shape(self._state)} when CSI_as_state = {CSI_as_state} and sample_error_in_CSI = {sample_error_in_CSI}\n') 
            # time.sleep(10)
         # just before initializing age, this episode ending age to be saved   
        self.initialize_age()
        
        return ts.restart(np.array([self._state], dtype=np.float32))

        
    def map_actions(self, action):  
        '''
        convert the single integer action to specific sampling and updating tasks
        '''
        # print(f'inside  map_actions, action={action}, type(action)={type(action)}')
        # print(f'action={action},self.actions_space={self.actions_space}')
        actual_action = self.actions_space[action]
        if verbose:
            # print(f'action space is {self.actions_space}, length is {self.action_size}, array size is {len(self.actions_space)} selected action is {action} which maps to {actual_action}')
            pass
        return actual_action
    
    def get_current_state(self): # 
        # doesn't change anything, just returns the current state. Ages have been updated in the take_RL_action, here the new state is returned
        state_UAV = np.array(list(self.UAV_age.values()))
        state_BS  = np.array(list(self.BS_age.values()))
        
        if CSI_as_state:
            if sample_error_in_CSI:
                self._state = np.concatenate((list([self.current_step]), state_UAV, state_BS, list(self.sample_loss.values()) , list(self.update_loss.values())), axis=None) 
            else:
                self._state = np.concatenate((list([self.current_step]), state_UAV, state_BS, list(self.update_loss.values())), axis=None) 
                
        else:
            self._state = np.concatenate((list([self.current_step]), state_UAV, state_BS), axis=None) 

        if verbose:
            print(f'\nself._state from of {self.name} get_current_state() = {self._state} with shape = {np.shape(self._state)}\n') # debug
        return (self._state)
    
    def create_action_space(self):
        '''
        for 1 UAV once the coverage has been decided, create the action space
        2 data structures needed, a dict for storing sampling possibility at each UAV and a list to store the updating possibility
        '''
        
        update_user_possibilities = []

        if len(self.user_list) < self.BS_capacity: ## number of users available < number of users BS can update at anytime
            update_user_possibilities = list(combinations(self.user_list, len(self.user_list)))
        else:
            update_user_possibilities = list(combinations(self.user_list, self.BS_capacity)) # different combination of selecting UAVs to update
        if verbose:
            print(f"self.user_list = {self.user_list}, self.BS_capacity = {self.BS_capacity}, update_user_possibilities = {update_user_possibilities} with length {len(update_user_possibilities)}")
        
        update_user_possibilities = [list(i) for i in update_user_possibilities]
        
        ## update action part done
        sampling_choices = {}
        ##a dict with all combinations of sampling UAV_capacity users from each UAV irrespective of which UAV got updated
        for x in self.UAV_list:
            sampling_choices[x] = []
        
        for k in self.UAV_list:
            if len(self.act_coverage[k]) < self.UAV_capacity:
                sampling_choices[k].extend(list(combinations(self.act_coverage[k], len(self.act_coverage[k]))))
                # sampling_choices.extend(sampling_choice)
                
            else:
                sampling_choice = list(combinations(self.act_coverage[k], self.UAV_capacity))
                sampling_choices[k].extend(sampling_choice)
                # if verbose:
                #     print(f"users covered under UAV k are {self.act_coverage[k]}, sampling_choice = {sampling_choice}, and sampling choices has become {sampling_choices}")

        all_user_sampling_combinations = []
        for p in itertools.product(*sampling_choices.values()):
            y = [yy for zz in p for yy in zz]
            all_user_sampling_combinations.append(y)
            
        # print(f"update_user_possibilities = {update_user_possibilities}, {type(update_user_possibilities)}")
        # print(f"all_user_sampling_combinations = {all_user_sampling_combinations}, {type(all_user_sampling_combinations)}")
        
        # time.sleep(2)
        

        actions_space = list(itertools.product(update_user_possibilities, all_user_sampling_combinations))
        
        actions_space = [list(i) for i in actions_space]
            
        if verbose:
            print(f"sampling_choices = {sampling_choices} with length = {len(sampling_choices)} and \nall_user_sampling_combinations is {all_user_sampling_combinations} with length {len(all_user_sampling_combinations)}")
            print("\naction_size is ", len(actions_space)) #, " and they are actions_space = ", actions_space)
            # time.sleep(10)
            
            
        self.actions_space = actions_space
        self.action_size = len(self.actions_space)
        
        if verbose:
            print(f"\n{self.name} has a action_space of size ", np.shape(self.actions_space)) #, " and they are ", self.actions_space,  "\n")
            
        # print("\n action_space is of size ", np.shape(self.actions_space), file = open(self.folder_name + "/results.txt", "a"))
        # print("\n action_space is of size ", np.shape(self.actions_space), " and they are ", self.actions_space)
    
    def _step(self, action):
        # print("step ", self.current_step, " started") ## runs for MAX_STEPS steps
        # each step returns TimeStep(step_type, reward, discount, observation

                  
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            if verbose:
                print(f'for {self.name}, episode = {self.episode_step} at first reset')
            return self.reset()
        
        actual_action = self.map_actions(action)
        action = action.tolist()
        
        # print(f"\n env = {self.name}, self.current_step = {self.current_step}, self.episode_step = {self.episode_step}, action = {action}, type(action) = {type(action)}, (actual_action)={actual_action}, preference = {self.preference}") #, {type(self.preference[action])}") ## for some reason action is type nparray
        self.preference[action][-1] = self.preference[action][-1] + 1
            
        # updated_UAVs  = list(actual_action[0])
        updated_users = list(actual_action[0])
        # for j in updated_UAVs:
        #     updated_users.extend(self.act_coverage[j])
        sampled_users = list(actual_action[1])
        if verbose:
        
            print(f'\nfor {self.name}, current_step = {self.current_step}, selected action = {action}, actual_dqn_action={actual_action}, updated_users = {updated_users} sampled_users={sampled_users}\n') 
            # time.sleep(3)
            
            print(f"{self.name} tx_attempt_BS was {self.tx_attempt_BS}")
        
        if self.current_step==1: ## updating
        # step 1 so BS has nothing to get from UAV
            for i in self.user_list:
                self.BS_age[i] = self.BS_age[i]+1

        else: # not time step = 1
            for i in self.user_list: ## updating
                if i in updated_users:
                    ## find associated UAV
                    for kk in self.act_coverage:
                        if i in self.act_coverage[kk]:
                            associated_UAV = kk
                    ##
                    self.tx_attempt_BS[i][-1] = self.tx_attempt_BS[i][-1] + 1
                    self.attempt_update[-1] = self.attempt_update[-1] + 1
                    chance_update_loss = np.round(random.random(), 2)
                    if verbose:
                    #     print(f"user {i}'s associated UAV is {associated_UAV}")
                        print(f"for user {i}, chance_update_loss = {chance_update_loss} and {self.name}.update_loss = {self.update_loss[i]} ")
                    if chance_update_loss > self.update_loss[i]:
                        if verbose:
                            print("user ", i, " was updated")
                        self.BS_age[i] = self.UAV_age[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                        self.success_update[-1] = self.success_update[-1] + 1
                    else:
                        self.BS_age[i] = self.BS_age[i] + 1
                        if verbose:
                            print(f'user {i} was updated but had update failure')
                            
                else:
                    if verbose:
                        print("user ", i, " was not updated")
                    self.BS_age[i] = self.BS_age[i] + 1
                
        
        if verbose:
            print(f"{self.name} tx_attempt_BS has become {self.tx_attempt_BS}")
            ## for first step, even if some user was updated, it is not counted in the tx_attempt_BS as that tx has no role in minimizing AoI. Therefore for a scenario with 4 steps, tx_attempt_UAV will go to max 4 but tx_attempt_BS will go to 3 as only 3 attempts are counted
            print(f"{self.name} tx_attempt_UAV was {self.tx_attempt_UAV}")

        for i in self.user_list: ## sampling
            if i in sampled_users:
                chance_sample_loss = np.round(random.random(), 2)
                self.tx_attempt_UAV[i][-1] = self.tx_attempt_UAV[i][-1] + 1
                self.attempt_sample[-1] = self.attempt_sample[-1] + 1
                if verbose:
                    print(f" for user {i}, chance_sample_loss = {chance_sample_loss} and self.sample_loss = {self.sample_loss[i]} ")
                if chance_sample_loss > self.sample_loss[i]:
                    if (self.current_step-1)%self.periodicity[i]==0: ## -1 as here time starts from 1
                        if verbose:
                            print("slot = ", self.current_step-1, " - user ", i, " period = ", self.periodicity[i], " was selected to sample and sampled")
                        self.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
                        self.success_sample[-1] = self.success_sample[-1] + 1
                    else:
                        if verbose:
                            print("slot = ", self.current_step-1, " - user ", i, " period = ", self.periodicity[i], " was selected to sample but not sampled")                    
                else:
                    self.UAV_age[i] = self.UAV_age[i] + 1
                    if verbose:
                        print(f'user {i} was sampled but had sample failure')
                        
                    
            else:
                if verbose:
                    print("user ", i, " was not sampled")
                self.UAV_age[i] = self.UAV_age[i] + 1
        
        # print(f"slot {self.current_step} ended with state {self._state}")
        # time.sleep(20)
                
        if verbose:
            print(f"{self.name} tx_attempt_UAV has become {self.tx_attempt_UAV}")
                
        self._state = self.get_current_state() # update state after every action
        
       
        BS_sum_age = np.sum(list(self.BS_age.values()))
        
        self.current_step += 1
        
        award = -BS_sum_age
        
        
        # if verbose:
        #     print(f"attempt_sample = {self.attempt_sample}")
        #     print(f"success_sample = {self.success_sample}")
        #     print(f"attempt_update = {self.attempt_update}")
        #     print(f"success_update = {self.success_update}")
        #     time.sleep(5)
                   
        
        if verbose:
            print(f'new current_step = {self.current_step}')
            print(f"{self.name} tx_attempt_UAV has become {self.tx_attempt_UAV}")
            print(f"new state is {self.get_current_state()}")
            print(f'\nfor {self.name}, award is {award}\n') # debug
            # time.sleep(10)

        
        if self.current_step < MAX_STEPS + 1:        ## has to run for MAX_STEPS, i.e. an action has to be chosen MAX_STEPS times
            self._episode_ended = False
            return ts.transition(np.array([self._state], dtype=np.float32), reward = award, discount=1.0)
        else:
            # print(f'in terminate block') # will also reset the environment
            self._episode_ended = True
            # time_step.is_last() = True
            return ts.termination(np.array([self._state], dtype=np.float32), reward=award)