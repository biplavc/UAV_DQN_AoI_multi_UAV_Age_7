from tf_environment import *
from create_graph_1 import *
# if comet:
#     from main_tf import experiment

random.seed(42)
np.random.seed(42)
# tf.random.set_seed(42)

### collect_steps_per_iteration -> collect_episodes_per_iteration

# source - https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

def tf_dqn(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity):  
    all_actions = []    ## save all actions over all steps of all episodes  
    print(f"\nDQN started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity} and {deployment} deployment")
    print(f"\nDQN started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity} and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)

    train_py_env = UAV_network(I, drones_coverage, "train net", folder_name, packet_update_loss, packet_sample_loss, periodicity)
    eval_py_env = UAV_network(I, drones_coverage, "eval net", folder_name, packet_update_loss, packet_sample_loss, periodicity)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
        
    train_env.reset()
    eval_env.reset()
    
    final_step_rewards = []
    
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space,  file=open(folder_name + "/results.txt", "a"), flush = True)
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space,"\n")
    
    dqn_returns = []
    
    initial_collect_episodes = 1000  # @param {type:"integer"}  # collect_data runs this number of steps for the first time, not in REINFORCE agent
    
    batch_size = 16  # @param {type:"integer"}
    
    #### Agent 

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
    target_q_network = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) # same

    train_step_counter = tf.Variable(0)


## https://github.com/tensorflow/agents/blob/755b43c78bb50e36b1331acc9492be599997a47f/tf_agents/agents/dqn/dqn_agent.py#L113

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        gamma = set_gamma,
        target_q_network = target_q_network,
        target_update_tau = 1.0,
        target_update_period = 10,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    
    # decay epsilon parameters, relates to exploration value though the param names are in terms of learning rate

    start_epsilon = 0.2
    end_learning_rate = 0.01
    decay_steps = 40_000
    
    epsilon = tf.compat.v1.train.polynomial_decay(
                                                learning_rate = start_epsilon,
                                                global_step = agent.train_step_counter.numpy(), # current_step
                                                decay_steps = decay_steps,
                                                power = 1.0,
                                                #cycle = True,
                                                end_learning_rate=end_learning_rate)
    
    ## tf.compat.v1.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0,cycle=False, name=None)
    
    # decay epsilon ends

    agent.initialize()
    
    #### Policies are properties of agents, sometimes optimizers are also properties of agents
    
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    collect_policy._epsilon = epsilon
    
    if agent.train_step_counter.numpy()==0:
        # time.sleep(10)
        # pass
        print(f"\nDQN scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, BS_capacity is {train_py_env.BS_capacity}, UAV_capacity = {train_py_env.UAV_capacity}, action space size is {train_py_env.action_size} and they are {train_py_env.actions_space} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
        print(f"\nDQN scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, BS_capacity is {train_py_env.BS_capacity}, UAV_capacity = {train_py_env.UAV_capacity}, action space size is {train_py_env.action_size} \n\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    if verbose:
        print(f"DQN reward discount rate = {agent._gamma}")
        print(f"\nDQN eval_policy = {eval_policy}, collect_policy = {collect_policy} with epsilon = {collect_policy._epsilon}")
        # DQN eval_policy = <tf_agents.policies.greedy_policy.GreedyPolicy object at 0x7fa02c4a2700>, collect_policy = <tf_agents.policies.epsilon_greedy_policy.EpsilonGreedyPolicy object at 0x7fa02c51a8e0> with epsilon 0.1        time.sleep(5)
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()) # used once in collect_episode first time with initial_collect_episodes
    
    time_step = train_env.reset()
    random_policy.action(time_step)


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
            
            # if comet==True: 
            #     experiment.log_metric("final_return_DQN", time_step.reward.numpy()[0], step = i) # "loss",loss_val,step=i
            final_step_rewards.append(time_step.reward.numpy()[0])

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    compute_avg_return(eval_env, random_policy, num_eval_episodes) # to see a baseline with random policy

    #### Replay Buffer
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)
    
    agent.collect_data_spec
    agent.collect_data_spec._fields


    
    #### Data Collection
    
    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        all_actions.append(action_step.action.numpy()[0])
        # print(f"all_actions = {all_actions}, type = {type(all_actions)}")
        # time.sleep(2)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_episode(env, policy, buffer, num_episodes): # DRL's collect_episode = collect_episode+collect_steps
        for _ in range(num_episodes):
            collect_step(env, policy, buffer)

    collect_episode(train_env, random_policy, replay_buffer, initial_collect_episodes)
        
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

    # Evaluate the agent's policy once before training, same as DRL
    
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    UAV_returns = [sum(eval_py_env.UAV_age.values())]
    # print(vars(eval_py_env))
    
    # print(f"UAV_returns = {UAV_returns} and with {eval_py_env.UAV_Age}")
    # time.sleep(15)
    
    for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
        collect_episode(train_env, agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss), flush=True)
            print(f"Average Age = {np.mean(final_step_rewards[-5:])}\n")
            # pass

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            # print('step = {0}: Average Return = {1}'.format(step, avg_return), flush =True)
            returns.append(avg_return)
            UAV_returns.append(sum(eval_py_env.UAV_age.values()))
            # print(f"UAV_returns = {UAV_returns} and with {eval_py_env.UAV_age}")

            
    dqn_returns = returns
    
    pickle.dump(dqn_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_returns.pickle", "wb"))
    pickle.dump(UAV_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_UAV_returns.pickle", "wb"))
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_final_step_rewards.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_tx_attempt_BS.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_tx_attempt_UAV.pickle", "wb"))
    pickle.dump(eval_py_env.preference, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_preference.pickle", "wb")) 


    pickle.dump(eval_py_env.BS_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_BS_age.pickle", "wb")) 
    pickle.dump(eval_py_env.UAV_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_UAV_age.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_age_dist_BS.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_age_dist_UAV.pickle", "wb"))
    
    pickle.dump(eval_py_env.attempt_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_attempt_sample.pickle", "wb"))
    pickle.dump(eval_py_env.success_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_success_sample.pickle", "wb"))
    pickle.dump(eval_py_env.attempt_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_attempt_update.pickle", "wb"))
    pickle.dump(eval_py_env.success_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_success_update.pickle", "wb"))
    
    # for variable users per drone, all these above codes have to be taken inside the scheduling loop like the tx_attempt pickle
    # pickle.dump(all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_all_actions_dqn.pickle", "wb"))
    
    
    print("\nDQN scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(dqn_returns[-5:]), " : end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state))
    
    print("\nDQN scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(dqn_returns[-5:]), " : end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)

    
    print(f"greedy ended for {I} users and {deployment} deployment")
    return dqn_returns, final_step_rewards, all_actions
    