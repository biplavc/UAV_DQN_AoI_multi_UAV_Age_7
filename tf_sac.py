from tf_environment import *


## source = https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb

## the final_returns is not implemented here as this code doesn't have asy is_step.is_last() step. But we oly plot user's age so we can still plot sum age at the BS using BS_age_dist

def tf_sac(folder_name):
    
    # num_iterations = 100000 # @param {type:"integer"}

    initial_collect_steps = 100 # 100 based on dqn, 10000 # @param {type:"integer"}
    collect_steps_per_iteration = collect_episodes_per_iteration # 1 # @param {type:"integer"}
    # replay_buffer_capacity = 10000 # @param {type:"integer"}

    batch_size = 64 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (32, 16)
    critic_joint_fc_layer_params = (32, 16)

    # log_interval = 500 # @param {type:"integer"}

    # num_eval_episodes = 20 # @param {type:"integer"}
    # eval_interval = 10000 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}
    
    print(f'sac started', flush=True)
    
    ## ENVIRONMENT
    
    train_py_env = UAV_network(3,3, "train net")
    eval_py_env = UAV_network(3,3, "eval net")

    collect_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)    
        
    collect_env.reset()
    eval_env.reset()
      
    final_step_rewards = []
    
    sac_returns = []
    
    # GPU AND STRATEGY
    use_gpu = False #@param {type:"boolean"}

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    
    ## AGENTS
    
    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))


    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')
        
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(  observation_spec, action_spec,       fc_layer_params=actor_fc_layer_params,           continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork))
        
    
    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=train_step)

        tf_agent.initialize()
        
    ## REPLAY BUFFER
    
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])
    
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)
    
    dataset = reverb_replay.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset
    
    
    # POLICIES
    
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
    
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)
    
    random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())
    
    ## ACTORS
    
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(reverb_replay.py_client,table_name,sequence_length=2,stride_length=1)
    
    initial_collect_actor = actor.Actor(collect_env, random_policy,  train_step, steps_per_run=initial_collect_steps, observers=[rb_observer])
    
    initial_collect_actor.run()
    
    env_step_metric = py_metrics.EnvironmentSteps()
    
    collect_actor = actor.Actor(collect_env, collect_policy, train_step, steps_per_run=1, metrics=actor.collect_metrics(10), summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric])
    
    eval_actor = actor.Actor(eval_env, eval_policy, train_step, episodes_per_run=num_eval_episodes, metrics=actor.eval_metrics(num_eval_episodes), summary_dir=os.path.join(tempdir, 'eval'),)
    
    ## LEARNERS
    
    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger( saved_model_dir,           tf_agent, train_step, interval=policy_save_interval),         triggers.StepPerSecondLogTrigger(train_step, interval=1000),]

    agent_learner = learner.Learner( tempdir, train_step, tf_agent,     experience_dataset_fn, triggers=learning_triggers)
    
    ##  METRICS AND EVALUATION
    
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()
    
    def log_eval_metrics(step, metrics):
        eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)
    
    ## TRAINING THE AGENT
    
    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
    # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()
    
    sac_returns = returns
    
    pickle.dump(sac_returns, open(folder_name + "/SAC_returns.pickle", "wb"))
    
    # pickle.dump(final_step_rewards, open(folder_name + "/SAC_final_step_rewards.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_BS, open(folder_name + "/SAC_tx_attempt_BS.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/SAC_tx_attempt_UAV.pickle", "wb"))
    pickle.dump(eval_py_env.preference, open(folder_name + "/SAC_preference.pickle", "wb")) 
        
    # print(f'final episode\'s age = ', final_age, file=open(folder_name + "/results.txt", "a"), flush = True)

    # print("deployment =", deployment, ",scheduling = ", scheduling, ", overall average sum BS age = ", np.mean(age_BS), file=open(folder_name + "/results.txt", "a"), flush = True)

    pickle.dump(eval_py_env.BS_age, open(folder_name + "/SAC_BS_age.pickle", "wb")) 
    pickle.dump(eval_py_env.UAV_age, open(folder_name + "/SAC_UAV_age.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_BS, open(folder_name + "/SAC_age_dist_BS.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/SAC_age_dist_UAV.pickle", "wb"))
    # for variable users per drone, all these above codes have to be taken inside the scheduling loop like the tx_attempt pickle
    
    return sac_returns
    
    
    
    
    
    
    
    