from tf_environment import *
from create_graph_1 import *
# if comet:
#     from main_tf import experiment

random.seed(42)

#### learning=="REINFORCE agent":

# source - https://www.tensorflow.org/agents/tutorials/6_reinforce_tutorial

def tf_reinforce(I, drones_coverage, folder_name, deployment):
    all_actions = []    ## save all actions over all steps of all episodes
    print("\nreinforce started ", file=open(folder_name + "/results.txt", "a"))
    print(f"\nreinforce started for {I} users and {deployment} deployment")
    
    print(f"\nI = {I}, drones_coverage = {drones_coverage}, folder_name = {folder_name}\n")
    # time.sleep(5)
    
    train_py_env = UAV_network(I, drones_coverage, "train net", folder_name)
    eval_py_env = UAV_network(I, drones_coverage, "eval net", folder_name)
    

    train_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    print('\n')
    print(isinstance(train_env, tf_environment.TFEnvironment))
    print(isinstance(eval_env, tf_environment.TFEnvironment))
    print('\n')
    
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space,  file=open(folder_name + "/results.txt", "a"), flush = True) 
    # print("action space size is ", train_py_env.action_size, " and they are ", train_py_env.actions_space, "\n")
    
    
    final_step_rewards = [] # current rewards is the sum of BS age at each step of an episode, but we need the age only for the final step. so final step's reward is added here for each episode run.

    train_env.reset()
    eval_env.reset()
    # print('\n\nTime step:')
    # print(time_step_1, '\n', time_step_2)
    #print(f"\nreinforce ended for {I} users and {deployment} deployment") following arrays will be saved for plotting
    
    reinforcement_returns = []

    #### Agent

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)


    #### Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) # same

    train_step_counter = tf.compat.v2.Variable(0)
            
    # print('batch size is ', train_env.batch_size, eval_env.batch_size)

    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        gamma = set_gamma,
        gradient_clipping=1,
        normalize_returns=True,
        train_step_counter=train_step_counter)

    tf_agent.initialize()
    
    # decay epsilon parameters

    # here the collect policy is not greedy epsilon so no decay epsilon here
    
    # decay epsilon ends


    #### Policies
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    if verbose:
        print(f"\nRL eval_policy = {eval_policy}, RL collect_policy = {collect_policy}")
        time.sleep(5)
        # RL eval_policy = <tf_agents.policies.greedy_policy.GreedyPolicy object at 0x7fbedc1d8df0>, RL collect_policy = <tf_agents.policies.actor_policy.ActorPolicy object at 0x7fbedc1d8dc0>


    #### Metrics and Evaluation
    def compute_avg_return(environment, policy, num_episodes=100):
    # runs on eval_environment
        if verbose:
            print(f"\nInside compute_avg_return with env={environment}, policy={policy}, num_episodes={num_episodes}\n")


        total_return = 0.0
        for i in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                all_actions.append(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            
            # store the age at UAV and BS at the end of every episode
            
            # if comet==True:
            #     experiment.log_metric("final_return_REINFORCE", time_step.reward.numpy()[0], step = i) # "loss",loss_val,step=i
            final_step_rewards.append(time_step.reward.numpy()[0])
            
            if verbose:
                print(f'episode={i}, step reward = {time_step.reward}, episode_return={episode_return}, total_return={total_return}, final_step_reward ={ final_step_rewards}')            
            
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    #### Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity) # batch size not explicitly mentioned


    #### Data Collection


    def collect_episode(environment, policy, num_episodes):
    # collect episode fills the buffer
        episode_counter = 0
        environment.reset()
        if verbose:
            print(f"inside collect_episode with env={environment}, num_episodes={num_episodes}")


        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            if verbose:
                print(f"action_step = {action_step}")
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj) # even though it is add_batch, we are adding one transitions at a time and not a batch of transitions

            if traj.is_boundary():
                episode_counter += 1
                
    #### Training the agent

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    if verbose:
        print(f"\nabout to run compute_avg_return for avg_return\n")
    # first to run and this line runs only once
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        if verbose:
            print(f"\ninside for loop of num_iterations = {num_iterations}\n")

    # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all() # Returns all the items in buffer
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()
        
        if verbose:
            ## gets incremented by 1 for each time the collect_episode runs
            print(f"\nval of step inside num_iterations is {step}")

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss), flush=True)
            
        if verbose:
            pass
            # print(f'about to evaluate with compute_avg_return using eval_env')
            
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            # print('step = {0}: Average Return = {1}'.format(step, avg_return), flush=True)
            returns.append(avg_return)
            
    reinforcement_returns = returns
    pickle.dump(reinforcement_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_returns.pickle", "wb"))
    
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_final_step_rewards.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_tx_attempt_BS.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_tx_attempt_UAV.pickle", "wb"))
    pickle.dump(eval_py_env.preference, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_preference.pickle", "wb")) 
        

    pickle.dump(eval_py_env.BS_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_BS_age.pickle", "wb")) # will only print the final ages
    pickle.dump(eval_py_env.UAV_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_UAV_age.pickle", "wb")) # # will only print the final ages
    pickle.dump(eval_py_env.age_dist_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_age_dist_BS.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_RL_age_dist_UAV.pickle", "wb"))
    
    pickle.dump(all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_all_actions_reinforce.pickle", "wb"))
    
    # for variable users per drone, all these above codes have to be taken inside the scheduling loop like the tx_attempt pickle

    # print("reinforce scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards), " and avg of reinforcement_returns = ", np.mean(reinforcement_returns))
    print("reinforce scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " and avg of reinforcement_returns = ", np.mean(reinforcement_returns[-5:]), file = open(folder_name + "/results.txt", "a"), flush = True)
    
    print(f"reinforce ended for {I} users and {deployment} deployment")
    return reinforcement_returns, final_step_rewards, all_actions