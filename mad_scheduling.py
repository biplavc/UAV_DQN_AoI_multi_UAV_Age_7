from tf_environment import *
from create_graph_1 import *
import collections
from collections import defaultdict
import itertools
import operator
from datetime import datetime
import sys

# random.seed(42)

def find_mad_action(eval_env): ## maximal age difference
    ## https://pynative.com/python-random-seed/
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)

    
    mad_age_BS = {x:(eval_env.BS_age[x] - eval_env.UAV_age[x]) for x in eval_env.user_list}
    if verbose:
        print(f"age difference of users at BS is {mad_age_BS} where age at BS is {eval_env.BS_age}, age at UAV is {eval_env.UAV_age}")
        # time.sleep(5)
   
        print(f"mad_age_BS = {mad_age_BS}")
   
    ## UAV selection begins

    sampleDict_copy = copy.deepcopy(mad_age_BS)

    updated_users = []
    while len(updated_users) < eval_env.BS_capacity:

        itemMaxValue = max(sampleDict_copy.items(), key=lambda x: x[1])
        # print('Max Age Difference : ', itemMaxValue[1])
        listOfKeys = list() ## covered UAVs that will be removed once added to the selected_UAVs
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in sampleDict_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)
        # print(f"sorted_BS_users_age_diff_dict = {sorted_BS_users_age_diff_dict}, BS_age = {eval_env.BS_age}, UAV_age = {eval_env.UAV_age}")
        # print('UAVs with maximum Age Diff : ', listOfKeys)


        remaining_capacity = eval_env.BS_capacity - len(updated_users)
        if remaining_capacity > 0:
            # print(f"remaining_capacity = {remaining_capacity}")
            if len(listOfKeys) > remaining_capacity: ## listOfKeys can be filled with some combination
                updated_users.extend(random.sample(listOfKeys, remaining_capacity))
            else: ## entire listOfKeys can be entered
                updated_users.extend(listOfKeys)
        
        
        for items in listOfKeys:
            del sampleDict_copy[items] ## covered users will be deleted

        # print(f"sampleDict_copy for next loop is {sampleDict_copy} with BS_age = {eval_env.BS_age} and selected_UAVs = {selected_UAVs}")
    
    updated_users = tuple(updated_users)
    
    if verbose: 
        print(f"updated_users = {updated_users}")
        
    ## users for updating completed, then select user for sampling
    
    sampling_possibilities = [] ## for each UAV, combination of the capacity number of users
    for m in eval_env.UAV_list:
        users_UAV = eval_env.act_coverage[m] # users under the UAV m
        if len(users_UAV) < eval_env.UAV_capacity:
            sample = list(combinations(users_UAV, len(users_UAV)))
            sampling_possibilities.extend(sample)
        else:
            sample = list(combinations(users_UAV, eval_env.UAV_capacity))
            sampling_possibilities.extend(sample)


    ## all sampling actions collected
    
    ## select users to sample
    
    selected_users_overall = []
    
    for UAV in eval_env.UAV_list: # make a dict for each UAV storing the age of the users under it
        
        selected_users_UAV = [] ## users selected under this UAV
        
        users_covered = eval_env.act_coverage[UAV]
        users_age = {user:eval_env.UAV_age[user] for user in users_covered}  ## dict
        
        if verbose:
        
            print(f"under UAV {UAV}, users covered are {users_covered} and their ages are {users_age} when the general UAV age was {eval_env.UAV_age}")
              
        
        while len(selected_users_UAV) < eval_env.UAV_capacity and len(users_age) > 0: ## users selected under this UAV should be less than capacity but also in cases when UAV has less users, second condition here invoked
        
            itemMaxValue = max(users_age.items(), key=lambda x: x[1])
            # print('Max Age  : ', itemMaxValue[1])
            listOfKeys = list() ## covered UAVs that will be removed once added to the selected_users_UAV
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in users_age.items():
                if value == itemMaxValue[1]:
                    listOfKeys.append(key)
            # print(f"sorted_BS_users_age_diff_dict = {sorted_BS_users_age_diff_dict}, BS_age = {eval_env.BS_age}, UAV_age = {eval_env.UAV_age}")
            # print('UAVs with maximum Age Diff : ', listOfKeys)


            remaining_capacity = eval_env.UAV_capacity - len(selected_users_UAV)
            if remaining_capacity > 0:
                # print(f"remaining_capacity = {remaining_capacity}")
                if len(listOfKeys) > remaining_capacity: ## listOfKeys can be filled with some combination
                    selected_users_UAV.extend(random.sample(listOfKeys, remaining_capacity))
                else: ## entire listOfKeys can be entered
                    selected_users_UAV.extend(listOfKeys)
            
            
            for items in listOfKeys:
                del users_age[items] ## covered users will be deleted

        if verbose:
            print(f"users selected under UAV {UAV} are {selected_users_UAV}")
        
        selected_users_overall.extend(selected_users_UAV)
        
    if verbose:
        print(f"selected_users_overall = {selected_users_overall}")
    
############################################

    mad_action = list(itertools.product([updated_users], [selected_users_overall]))
        
    ## finally collect the result and compare with the action space of the environment
    updated_users = sorted(mad_action[0][0])
    sampled_users= sorted(mad_action[0][1])
    
    actual_mad_action = None # this has to change
    
    for action in range(eval_env.action_size):
        eval_action = eval_env.map_actions(action)
        # if verbose:
        #     print(f"\nm = {action}, eval_action={eval_action}, eval_action[0]={list(eval_action[0])}, eval_action[1]={list(eval_action[1])}, sampled_users = {sampled_users}, updated_UAVs = {updated_UAVs}")
        
        if sorted(list(eval_action[0]))==sorted(updated_users) and sorted(list(eval_action[1]))==sorted(sampled_users):
            actual_mad_action = action
            break

    if verbose:
        print(f"updated_users are {updated_users}, sampled_users = {sampled_users}, mad_action = {mad_action}, actual_mad_action = {actual_mad_action}")
        
    assert actual_mad_action!=None
        
    return actual_mad_action    


def mad_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity):  ## maximal age difference
    print(f"\nmad started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity} and {deployment} deployment")
    print(f"\nmad started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity} and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)
    # do scheduling for MAX_STEPS random_episodes times and take the average
    final_step_rewards = []
    overall_ep_reward = []
    
    all_actions = [] # just saving all actions to see the distribution of the actions
    
    age_dist_UAV =  {} ## dummpy vars to store age dist for each episode. 
    age_dist_BS  =  {}


    dd_age_dist_UAV = defaultdict(list) ## will be the final age
    dd_age_dist_BS = defaultdict(list) ## will be the final age   
    
    attempt_sample = []
    success_sample = []
    attempt_update = []
    success_update = []
    
    
    for ep in range(random_episodes): # how many times the random policy will be run, similar to episode
        ep_reward = 0
        # print(f"I = {I}, drones_coverage = {drones_coverage}")
        eval_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_update_loss, packet_sample_loss, periodicity) ## will have just the eval env here

        eval_env.reset() # initializes age

        episode_wise_attempt_sample = 0
        episode_wise_success_sample = 0
        episode_wise_attempt_update = 0
        episode_wise_success_update = 0    

        
        if ep==0:
            # time.sleep(10)
            # pass
            print(f"\nmad scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, BS_capacity is {eval_env.BS_capacity}, UAV_capacity = {eval_env.UAV_capacity}, action space size is {eval_env.action_size} and they are {eval_env.actions_space} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
            print(f"\nmad scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, BS_capacity is {eval_env.BS_capacity}, UAV_capacity = {eval_env.UAV_capacity}, action space size is {eval_env.action_size} ", file = open(folder_name + "/results.txt", "a"), flush = True)
        # print("eval_env.current_step = ", eval_env.current_step, ", eval_env.current_step = ", eval_env.current_step)
        eval_env.current_step  = 1
        action_space = eval_env.actions_space
            
        for i in eval_env.user_list:
                
            eval_env.age_dist_UAV[i].append(eval_env.UAV_age[i])
            eval_env.age_dist_BS[i].append(eval_env.BS_age[i])
            
            eval_env.tx_attempt_BS[i].append(0) # 0 will be changed in _step for every attempt
            eval_env.tx_attempt_UAV[i].append(0)
                
        for i in range(eval_env.action_size):
            eval_env.preference[i].append(0) 
            
        for x in range(MAX_STEPS):
            # print(x)
            # print("inside greedy - ", x, " step started")      ## runs MAX_STEPS times       
            selected_action = find_mad_action(eval_env)
            all_actions.append(selected_action)
            # print("all_actions", type(all_actions))
            eval_env.preference[selected_action][-1] = eval_env.preference[selected_action][-1] + 1
            action = eval_env.map_actions(selected_action)
            updated_users = list(action[0])
            sampled_users = list(action[1])

                
            if verbose:
                # pass
                print(f" step = {eval_env.current_step}, mad selection is {selected_action}, actual action is {action}, updated_users = {updated_users}, sampled_users = {sampled_users}")
                
            if eval_env.current_step==1: ## updating
            # step 1 so BS has nothing to get from UAV
                for k in eval_env.user_list:
                    eval_env.BS_age[k] = eval_env.BS_age[k]+1
                    
            else: # not time step = 1
                for i in eval_env.user_list: ## updating
                    if i in updated_users:
                        ## find associated UAV
                        for kk in eval_env.act_coverage:
                            if i in eval_env.act_coverage[kk]:
                                associated_UAV = kk
                        ##
                        episode_wise_attempt_update = episode_wise_attempt_update + 1
                        eval_env.tx_attempt_BS[i][-1] = eval_env.tx_attempt_BS[i][-1] + 1
                        chance_update_loss = round(random.random(), 2)
                        if verbose:
                            # print(f"user {i}'s associated UAV is {associated_UAV}")
                            print(f"for user {i}, chance_update_loss = {chance_update_loss} and eval_env.update_loss = {eval_env.update_loss[i]} ")
                        if chance_update_loss > eval_env.update_loss[i]:
                            if verbose:
                                print("user ", i, " was selected to be updated")
                            eval_env.BS_age[i] = eval_env.UAV_age[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                            episode_wise_success_update = episode_wise_success_update + 1
                        
                        else:
                            eval_env.BS_age[i] = eval_env.BS_age[i] + 1
                            if verbose:
                                print(f'user {i} was updated but had update failure')
                                
                    else:
                        if verbose:
                            print("user ", i, " was not updated")
                        eval_env.BS_age[i] = eval_env.BS_age[i] + 1
                            
            ## updating process done
            
            for i in eval_env.user_list: ## sampling
                if i in sampled_users:
                    chance_sample_loss = round(random.random(),2)
                    eval_env.tx_attempt_UAV[i][-1] = eval_env.tx_attempt_UAV[i][-1] + 1
                    episode_wise_attempt_sample = episode_wise_attempt_sample + 1
                    if verbose:
                        print(f" for user {i}, chance_sample_loss = {chance_sample_loss} and eval_env.sample_loss = {eval_env.sample_loss[i]} ")
                    if chance_sample_loss > eval_env.sample_loss[i]:
                        if x%periodicity[i]==0:
                            if verbose:
                                print("slot = ", x, " - user ", i, " period = ", periodicity[i], " was selected to sample and sampled")
                            eval_env.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
                            episode_wise_success_sample = episode_wise_success_sample + 1 
                        else:
                            if verbose:
                                print("slot = ", x, " - user ", i, " period = ", periodicity[i], " was selected to sample but not sampled")                    
                    else:
                        eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                        if verbose:
                            print(f'user {i} was sampled but had sample failure')
    
                else:
                    if verbose:
                        print("user ", i, " was not sampled")
                    eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                    
                    
            ## sampling process done
            if verbose:
                print(f"tx_attempt_UAV has become {eval_env.tx_attempt_UAV} and tx_attempt_BS has become {eval_env.tx_attempt_BS}")
           
            if verbose:
                print(f"\n step = {eval_env.current_step} ended, UAV_age = {eval_env.UAV_age}, BS_age = {eval_env.BS_age}") #
                # , tx_attempt_UAV = {eval_env.tx_attempt_UAV}, tx_attempt_BS = {eval_env.tx_attempt_BS}, preference = {eval_env.preference}")
                
            if verbose:
                    print(f"episode_wise_attempt_sample = {episode_wise_attempt_sample}")
                    print(f"episode_wise_success_sample = {episode_wise_success_sample}")
                    print(f"episode_wise_attempt_update = {episode_wise_attempt_update}")
                    print(f"episode_wise_success_update = {episode_wise_success_update}")
                    time.sleep(1)
                
            if eval_env.current_step==MAX_STEPS:
                final_reward = np.sum(list(eval_env.BS_age.values()))
                # print("sum age at BS = ", final_reward)
                
            eval_env.current_step += 1
            ep_reward = ep_reward + np.sum(list(eval_env.BS_age.values()))
          
        
        attempt_sample.append(episode_wise_attempt_sample)
        success_sample.append(episode_wise_success_sample)
        attempt_update.append(episode_wise_attempt_update)
        success_update.append(episode_wise_success_update)
        
        # if verbose:
        #     print(f"attempt_sample = {attempt_sample}")
        #     print(f"success_sample = {success_sample}")
        #     print(f"attempt_update = {attempt_update}")
        #     print(f"success_update = {success_update}")
        #     time.sleep(5)
        
        final_step_rewards.append(final_reward)
        overall_ep_reward.append(ep_reward)
        
        if ep==0: ## set up the empty dict with the appropriate keys in the first run
            age_dist_UAV.update(eval_env.UAV_age)
            age_dist_BS.update(eval_env.BS_age)
            
        else:
            for x in (age_dist_UAV, eval_env.UAV_age): # you can list as many input dicts as you want here
                for key, value in x.items():
                    dd_age_dist_UAV[key].append(value)
            

            for x in (age_dist_BS, eval_env.BS_age): # you can list as many input dicts as you want here
                for key, value in x.items():
                    dd_age_dist_BS[key].append(value)
        if verbose:
            time.sleep(20)
            print(f"\n*****************************************************\n")

        # print("final - age_dist_UAV = ", dd_age_dist_UAV, ", age_dist_BS = ", dd_age_dist_BS)
        
        
    pickle.dump(dd_age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_age_dist_UAV.pickle", "wb"))
    pickle.dump(dd_age_dist_BS, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_age_dist_BS.pickle", "wb"))
    
    pickle.dump(attempt_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_attempt_sample.pickle", "wb"))
    pickle.dump(success_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_success_sample.pickle", "wb"))
    pickle.dump(attempt_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_attempt_update.pickle", "wb"))
    pickle.dump(success_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_success_update.pickle", "wb"))
    
    
    print("\nMAD scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), " : end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state))
    
    print("\nMAD scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), " : end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)

    # pickle.dump(all_actions, open(folder_name + "/" + str(I) + "U_all_actions_greedy.pickle", "wb")) 
    
    assert(len(final_step_rewards)==len(final_step_rewards))
    return overall_ep_reward, final_step_rewards, all_actions
    