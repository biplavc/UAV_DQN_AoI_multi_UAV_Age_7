from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

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

import random
from itertools import combinations

from create_graph_1 import *
from path_loss_probability import *
from age_calculation import *
from itertools import product  
from drl import *
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import copy

verbose = True # False True
verbose = False

MAX_STEPS = 10

class UAV_network(py_environment.PyEnvironment):
    
    
    def __init__(self, n_users, coverage_capacity):
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=8, name='sample_update') # actually nCk*nCk
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,6), dtype=np.int32, minimum=0, maximum=MAX_STEPS+5, name='age_status')
        
        self._state = [1, 1, 1, 1, 1, 1]
        self._episode_ended = False

        
        self.n_users        = n_users
        self.list_of_users  = [0,1,2]
        self.L              = 21
        self.B              = 21
        self.UAV_capacity   = 1
        self.r              = 20  
        self.R              = round(self.r*(2**0.5), 3)
        self.user_locs      = []
        self.grid           = []
        self.UAV_loc        = []
        self.cover          = []
        self.UAV            = 0 # initialized to 0 but updated in create_topology()
        # self.current_state  = np.array([])
        # self.reward         = 0
        # self.actions_space  = [] # initialized once the coverage is calculated
        self.coverage_cap   = coverage_capacity
        self.action_size    = 1 #int(2*scipy.special.comb(coverage_capacity, self.UAV_capacity)) # coverage_capacity in create_graph_1
        self.episode_step   = 0
        # self.agent          = DLModel(self.max_coverage(), self.action_size)
        self.tx_attempts_BS = {}
        self.tx_attempts_UAV= {}
        self.preference     = {}
        self.current_step   = 1
        self.episodes_run   = 0
        self.UAV_age        = {}
        self.BS_age         = {}
        self.BS_age_prev    = {}
        self.start_network()
        

        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
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
    
    def start_network(self):
        if verbose:
            print('network started')
        self.create_action_space()
        # self.create_user_locations()
        # self.create_grid_locations()
        self._reset()

    
    def initialize_age(self):
        for i in self.list_of_users:
            # initial age put 1 and not 0 as if 0, in first time step whethere sampled or not, all users age at UAV becomes 1 but for 1, it is different - 2 for not sampled and 1 for sampled
            self.UAV_age[i] = 1
            self.BS_age[i]  = 1
            self.BS_age_prev[i] = 1 # special case for first step ??
    
    def _reset(self):
        # state at the start of the game
        self._state = [1, 1, 1, 1, 1, 1]
        self._episode_ended = False
        self.current_step   = 1
        if verbose:
            print(f'after reset, self._state = {self._state}') # debug
        self.initialize_age()

        return ts.restart(np.array([self._state], dtype=np.int32))
    
    # def _if_episode_ended(self, step):
    #     if step == MAX_STEPS:
    #         self._episode_ended = True  
    #     else:
    #         self._episode_ended = False  
        # resetting is done separately in _step, this is just to see is MAX_STEPS over in current episode
        
    def map_actions(self, action):  
        '''
        convert the single integer action to specific sampling and updating tasks
        '''
        # print(f'action={action},self.actions_space={self.actions_space}')
        actual_action = self.actions_space[action]
        # print(f'action space is {self.actions_space, selected action is {action} which maps to {actual_action}')
        return actual_action
    
    def get_current_state(self): # 
        # doesn't change anything, just returns the current state. Ages have been updated in the take_RL_action, here the new state is returned
        state_UAV = np.array(list(self.UAV_age.values()))
        state_BS  = np.array(list(self.BS_age.values()))
        self._state = np.concatenate((state_UAV, state_BS), axis=None) 
        # print(f'UAV_age = {self.UAV_age}, BS_age = {self.BS_age}') # biplav
        # print(f'inside getstate state = {self.state}') # biplav
        if verbose:
            print(f'self._state from get_current_state() = {self._state}') # debug
        return (self._state)
    
    def create_action_space(self):
        '''
        for 1 UAV once the coverage has been decided, create the action space
        '''
        # sampled_user = random.sample(self.list_of_users, self.coverage_cap)
        # sampled_user = random.sample(self.list_of_users, self.coverage_cap)

        # has to be changed for multiple users ?? drones user or overall user. 
        sampled_user_possibilities = list(combinations(self.list_of_users, self.UAV_capacity))
        updated_user_possibilities = list(combinations(self.list_of_users, self.UAV_capacity))
        self.actions_space = list(product(sampled_user_possibilities, updated_user_possibilities))
        self.action_size = len(self.actions_space)
        # print(f'action space is {self.actions_space}, number of actions are {(self.action_size)}')

    
    def _step(self, action):
        # each step returns TimeStep(step_type, reward, discount, observation
        
        if verbose:
            time.sleep(3)
            print(f'\n\ncurrent_step = {self.current_step}') # debug
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            self.episodes_run +=1
            if verbose:
                print(f'episode imptt = {self.episodes_run}')
            return self.reset()
        
        actual_action = self.map_actions(action)
            
        sampled_users = actual_action[0]
        updated_users = actual_action[1]
        if verbose:
            print(f'actual_action={actual_action}') # debug
            print(f'sampled_users={sampled_users}, updated_users={updated_users}') # debug
        
        if self.current_step==1: 
        # step 1 so BS has nothing to get from UAV
            for i in self.list_of_users:
                self.BS_age[i] = self.BS_age[i]+1

        else:
            for i in self.list_of_users:
                if i in updated_users:
                    # print("user ", i, " was updated")
                    self.BS_age[i] = self.UAV_age[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                    # self.tx_attempts_BS[i][episode-1] = self.tx_attempts_BS[i][episode-1] + 1
                else:
                    # print("user ", i, " was not updated")
                    self.BS_age[i] = self.BS_age[i] + 1

        for i in self.list_of_users:
            if i in sampled_users:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
                # self.tx_attempts_UAV[i][episode-1] = self.tx_attempts_UAV[i][episode-1] + 1
            else:
                # print("user ", i, " was sampled")
                self.UAV_age[i] = self.UAV_age[i] + 1
                
        self._state = self.get_current_state()
                    
        BS_sum_age = np.sum(list(self.BS_age.values()))
        
        self.current_step += 1
        # print(f'new current_step = {self.current_step}')
        award = -BS_sum_age
        if verbose:
            print(f'award is {award}') # debug
        
        if self.current_step!=MAX_STEPS:        
            self._episode_ended = False
            return ts.transition(np.array([self._state], dtype=np.int32), reward = award, discount=1.0)
        else:
            # print(f'in terminate block') # will also reset the environment
            self._episode_ended = True
            # time_step.is_last() = True
            return ts.termination(np.array([self._state], dtype=np.int32), reward=award)
        
if __name__ == '__main__':
    
    net_1 = UAV_network(5, 3)
    
    print('action_spec:', net_1.action_spec())
    print('time_step_spec.observation:', net_1.time_step_spec().observation)
    print('time_step_spec.step_type:', net_1.time_step_spec().step_type)
    print('time_step_spec.discount:', net_1.time_step_spec().discount)
    print('time_step_spec.reward:', net_1.time_step_spec().reward)
    
    '''

    
    train_py_env = UAV_network(5,3)
    eval_py_env = UAV_network(5,3)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    print('\n')
    print(isinstance(train_env, tf_environment.TFEnvironment))
    print(isinstance(eval_env, tf_environment.TFEnvironment))
    print('\n')
    
    time_step_1 = train_env.reset()
    time_step_2 = eval_env.reset()
    print('Time step:')
    print(time_step_1, '\n', time_step_2)
    
    # following arrays will be saved for plotting
    
    dqn_returns = []
    reinforcement_returns = []
    c51_returns = []

    
    #### HyperParameters
    # if learning=="REINFORCE agent":
    num_iterations = 15_000 # @param {type:"integer"} # number of times collect data is called, log_interval and eval_interval are used here
    
    collect_episodes_per_iteration = 2 # @param {type:"integer"} # collect_episode runs this number of episodes per iteration
    
    replay_buffer_capacity = 5000 # @param {type:"integer"} value of max_length, same for both

    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 25 # @param {type:"integer"} # how frequently to print out
    num_eval_episodes = 100 # @param {type:"integer"}
    eval_interval = 500 # @param {type:"integer"} # # compute_avg_return called every eval_interval, i.e avg_return calculated filled every eval_interval. this is what is shown in plot 
    
    
    #### Agent
    fc_layer_params = (32,16)
    
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
    
    #### Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) # same

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    
    tf_agent.initialize()
    
    
    #### Policies
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    
    
    #### Metrics and Evaluation
    def compute_avg_return(environment, policy, num_episodes=100):
    
        total_return = 0.0
        for i in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            if verbose:
                print(f'episode={i}, step reward = {time_step.reward}, episode_return={episode_return}, total_return={total_return}')

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    
    #### Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity) # batch size not explicitly mentioned
    
    
    #### Data Collection
    

    def collect_episode(environment, policy, num_episodes):
        episode_counter = 0
        environment.reset()

        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

            if traj.is_boundary():
                episode_counter += 1
                
    #### Training the agent

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            
    reinforcement_returns = returns
    pickle.dump(reinforcement_returns, open("reinforcement_returns.pickle", "wb"))



    
    # if learning=="DQN":
        
    #### Hyperparameters
        
    num_iterations = 15_000 # @param {type:"integer"} # number of times collect_data is called, log_interval and eval_interval are used here
    
    initial_collect_steps = 100  # @param {type:"integer"}  # collect_data runs this number of steps for the first time, not in REINFORCE agent
    
    collect_steps_per_iteration = 1  # @param {type:"integer"} # collect_data runs this number of steps after its first run per iteration, not in REINFORCE agent
    
    replay_buffer_max_length = 5000  # @param {type:"integer"} # value of max_length, same for both

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 25  # @param {type:"integer"} # how frequently to print out
    num_eval_episodes = 100  # @param {type:"integer"}
    eval_interval = 500  # @param {type:"integer"} # compute_avg_return called every eval_interval, i.e avg_return calculated filled every eval_interval. this is what is shown in plot 
    
    #### Agent 
    fc_layer_params = (32,16)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) # same

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    
    #### Policies
    
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()) # used once in collect_data first time with initial_collect_steps
    
    time_step = train_env.reset()
    random_policy.action(time_step)


    #### Metrics and Evaluation
    def compute_avg_return(environment, policy, num_episodes=100):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            if verbose:
                print(f'episode={i}, step reward = {time_step.reward}, episode_return={episode_return}, total_return={total_return}')

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    #### Replay Buffer
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
    
    agent.collect_data_spec
    agent.collect_data_spec._fields


    
    #### Data Collection
    
    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)
        
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(3)
    
    iterator = iter(dataset)
    
    #### Training the agent
    
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            
    dqn_returns = returns
    pickle.dump(dqn_returns, open("dqn_returns.pickle", "wb"))


            
     # if learning=="C51":
    #### Hyperparameters
    
    num_iterations = 15_000 # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"} 
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 5000  # @param {type:"integer"}


    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99
    log_interval = 25  # @param {type:"integer"}

    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -20  # @param {type:"integer"}
    max_q_value = 20  # @param {type:"integer"}
    n_step_update = 2  # @param {type:"integer"}

    num_eval_episodes = 100  # @param {type:"integer"}
    eval_interval = 500  # @
    
    fc_layer_params = (32,16)
    
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)
    
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=train_step_counter)
    
    agent.initialize()
    
    #### Metrics and Evaluation
    
    def compute_avg_return(environment, policy, num_episodes=10):
    
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    #### Data Collection
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    for _ in range(initial_collect_steps):
        collect_step(train_env, random_policy)

    # This loop is so common in RL, that we provide standard implementations of
    # these. For more details see the drivers module.

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)

    iterator = iter(dataset)
    
    
    #### Training the agent

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                returns.append(avg_return)
                
    c51_returns = returns
    
    pickle.dump(c51_returns, open("c51_returns.pickle", "wb"))

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, c51_returns, label = "C51")
    plt.plot(iterations, dqn_returns, label = "DQN")
    plt.plot(iterations, reinforcement_returns, label = "RL")
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    # plt.ylim(top=250)
    plt.legend()
    plt.show()

    '''
        

        
        