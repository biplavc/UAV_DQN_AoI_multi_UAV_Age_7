from tf_environment import *
from create_graph_1 import *
# if comet:
#     from main_tf import experiment
random.seed(42)

# source = https://github.com/tensorflow/agents/blob/master/docs/tutorials/9_c51_tutorial.ipynb
# if learning=="C51":
def tf_c51(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss):
    all_actions = []    ## save all actions over all steps of all episodes
    print(f"\nc51 started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss} and {deployment} deployment")
    print(f"\nc51 started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss} and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    train_py_env = UAV_network(I, drones_coverage, "train net", folder_name, packet_update_loss, packet_sample_loss)
    eval_py_env = UAV_network(I, drones_coverage, "eval net", folder_name, packet_update_loss, packet_sample_loss)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)    
        
    train_env.reset()
    eval_env.reset()
    
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space,  file=open(folder_name + "/results.txt", "a"), flush = True)    
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space, "\n")

    
    final_step_rewards = []
    
    c51_returns = []
    
    initial_collect_episodes = 1000  # @param {type:"integer"}  # collect_data runs this number of steps for the first time, not in REINFORCE agent

    batch_size = 16  # @param {type:"integer"}
    gamma = 0.99

    num_atoms = 51  # @param {type:"integer"}
    min_q_value = - MAX_STEPS*train_py_env.n_users-2 # +2 is just for buffer worst possible age sum at BS  # @param {type:"integer"}
    max_q_value =  -train_py_env.n_users ## best possible case, happens at the beginning of the simulation  # @param {type:"integer"}
    n_step_update = 2  # @param {type:"integer"}
    
    
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)
    
    target_categorical_q_network = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)
    
    
    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)  ## AdamOptimizer RMSPropOptimizer AdagradOptimizer

    train_step_counter = tf.compat.v2.Variable(0)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn= common.element_wise_squared_loss,
        gamma=set_gamma,
        # gradient_clipping = 1,
        target_categorical_q_network = target_categorical_q_network,
        target_update_tau = 1.0,
        target_update_period = 10,
        train_step_counter=train_step_counter)
    
    agent.initialize()
    
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
            
            final_step_rewards.append(time_step.reward.numpy()[0])
            # if comet==True:
            #     experiment.log_metric("final_return_c51", time_step.reward.numpy()[0], step = i) # "loss",loss_val,step=i

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    
    if agent.train_step_counter.numpy()==0:
            # time.sleep(10)
        # pass
        print(f"\nc51 scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, BS_capacity is {train_py_env.BS_capacity}, UAV_capacity = {train_py_env.UAV_capacity}, action space size is {train_py_env.action_size} and they are {train_py_env.actions_space} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
        print(f"\nc51 scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, BS_capacity is {train_py_env.BS_capacity}, UAV_capacity = {train_py_env.UAV_capacity}, action space size is {train_py_env.action_size} \n\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    # decay epsilon parameters
    
    start_epsilon = 0.2
    end_learning_rate=0.01
    decay_steps = 20_000
    
    epsilon = tf.compat.v1.train.polynomial_decay(
    learning_rate = start_epsilon,
    global_step = agent.train_step_counter.numpy(), # current_step
    decay_steps = decay_steps,
    end_learning_rate=end_learning_rate)
    
    ## tf.compat.v1.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0,cycle=False, name=None)
    
    # decay epsilon ends
    
    agent.collect_policy._epsilon = epsilon

    if verbose:
        print(f"\nc51 eval_policy = {random_policy}, collect_policy = {agent.collect_policy} with epsilon = {agent.collect_policy._epsilon}")
        # c51 eval_policy = <tf_agents.policies.random_tf_policy.RandomTFPolicy object at 0x7f0e804ecc70>, collect_policy = <tf_agents.policies.epsilon_greedy_policy.EpsilonGreedyPolicy object at 0x7f0e80451fa0>
        time.sleep(5)

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    #### Data Collection
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_episode(environment, policy):
        # collect_episode doesn't have collect_episodes_per_iteration as additional argument as from the place of calling, it is directly put in a for loop
        environment.reset() # biplav, was not there in the default code
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        all_actions.append(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    for _ in range(initial_collect_episodes):
        collect_episode(train_env, random_policy)

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
        for _ in range(collect_episodes_per_iteration):
            collect_episode(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss), flush=True)
            print(f"Average Age = {np.mean(final_step_rewards[-5:])}\n")
            # pass

        if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                # print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return), flush=True)
                returns.append(avg_return)
                

    c51_returns = returns
    pickle.dump(c51_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_returns.pickle", "wb"))
    
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_final_step_rewards.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_tx_attempt_BS.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_tx_attempt_UAV.pickle", "wb"))
    pickle.dump(eval_py_env.preference, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_preference.pickle", "wb")) 
        

    pickle.dump(eval_py_env.BS_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_BS_age.pickle", "wb")) 
    pickle.dump(eval_py_env.UAV_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_UAV_age.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_age_dist_BS.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_c51_age_dist_UAV.pickle", "wb"))
    # for variable users per drone, all these above codes have to be taken inside the scheduling loop like the tx_attempt pickle
    pickle.dump(all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_all_actions_c51.pickle", "wb"))

    print("\nc51 scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " and avg of overall_ep_reward = ", np.mean(c51_returns[-5:])) 
    
    print("\nc51 scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " and avg of overall_ep_reward = ", np.mean(c51_returns[-5:]), file = open(folder_name + "/results.txt", "a"), flush = True)

    print(f"c51 ended for {I} users and {deployment} deployment")
    return c51_returns, final_step_rewards, all_actions