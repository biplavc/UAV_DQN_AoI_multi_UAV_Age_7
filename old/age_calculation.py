import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
from parameters import *
import collections
import itertools

random.seed(3)
age_penalty = 10

def age_calculation(drones, coverage):

    '''
    drones = number of drones deployed, int
    coverage = list of users under each drone, list of list. list at index j will be the list of users under drone jth drone selected, not necessarily the jth drone in sequence

    need two ways to calculate age - random sampling and updating, Maximum Age sampling and updating

    maintain list containing Age of every user at UAV and BS. Initializing to 1
    '''

    # print("coverage=", coverage, "drones = ", drones)

    list_of_users = [x for y in coverage for x in y] # len(np.sum(coverage)) # sum(coverage) will contain all users
    users =  len(list_of_users)
    print("users = ", users)

    # initialization for random sampling and updating
    # each user has an initial age of 1
    UAV_age_1 = {} # age of users at UAV
    for i in list_of_users:
        UAV_age_1[i] = 1
    BS_age_1  = {} # age of user at BS
    for i in list_of_users:
        BS_age_1[i] = 1

    # initialization for greedy sampling and updating
    UAV_age_2 = {} # age of user at UAV
    for i in list_of_users:
        UAV_age_2[i] = 1
    BS_age_2  = {} # age of user at BS
    for i in list_of_users:
        BS_age_2[i] = 1


    drone_1 = copy.deepcopy(drones) # for random sch and updating
    coverage_1 = copy.deepcopy(coverage)

    drone_2 = copy.deepcopy(drones) # for greedy sch and updating
    coverage_2 = copy.deepcopy(coverage)

    """
    passing arguments:
    drone_1             - number of drones, int
    coverage_1          - users covered by each drone, list of list
    users               - number of users, int
    list_of_users       - all users as a list, list
    UAV_age_1           - age of all users at UAV, list and initialized to all 1s
    BS_age_1            - age of all users at BS , list and initialized to all 1s

    returned arguments:
    res_1               - final age of all users at UAV
    res_2               - final age of all users at BS
    """

    res_1, res_2 = age_random(drone_1, coverage_1, users, list_of_users, UAV_age_1, BS_age_1)
    res_3, res_4 = age_greedy(drone_2, coverage_2, users, list_of_users, UAV_age_2, BS_age_2)

    return(res_1, res_2, res_3, res_4)


def age_random(drone_1, coverage_1, users, list_of_users, UAV_age_1, BS_age_1):

    print("Random scheduling")
    
    for time in range(1, run_time): # for every time slot
        # print("\n")
        sampled_users = [] # list of users sampled by the UAV
        updated_users = [] # list of users updated by the UAV

        # sampling by UAV part starts

        for j in range(drone_1): # for every drone
            users_covered = coverage_1[j] # list containing all users covered
            if len(users_covered) < UAV_capacity: # all users under drone j is sampled
                # print('for drone ', j,", covered users = ", len(users_covered), " which is less than UAV capacity of ", UAV_capacity)
                for user in users_covered:
                    sampled_users.append(user)
                    # print("sampled_users = ", sampled_users)
            
            else: # all users can't be served only UAV_capacity users can be served
                # print('for drone ', j,", covered users = ", len(users_covered), " which is more than UAV capacity of ", UAV_capacity)
                to_sample = random.sample(users_covered, UAV_capacity) # selected users to sample
                for user in to_sample:
                    sampled_users.append(user)
                    # print("sampled_users = ", sampled_users)

        # sampling by UAV part ends

        # print("sampling process for each drone done, sampled users are ", sampled_users)
        # if time==run_time-1:
        #     print(len(sampled_users)," users sampled out of ", users, " in the random sampling and updating policy")
        # every drone has added which users to sample in sampled_users, next is to select which user will get updated

        # updating to BS starts

        if time==1:
            for i in list_of_users:
                BS_age_1[i] = BS_age_1[i]+1 #  # at the first slot no user is updated and age becomes 2 at the BS for all of the users, see table 1 of paper - Optimal Scheduling Policy for Minimizing Age of Information with a Relay


        else: 
            for j in range(drone_1): # for every drone
                users_covered = coverage_1[j] # list containing all users covered
                if len(users_covered) < UAV_capacity: # all users under drone j is updated
                    for user in users_covered:
                        updated_users.append(user)
            
                else: # all users can't be served only UAV_capacity users can be served
                    to_update = random.sample(users_covered, UAV_capacity) # selected users to sample
                    for user in to_update:
                        updated_users.append(user)

            # age at BS is updated before age at UAV as age at BS of the next slot depends on the age at UAV of current slot, and updating age at UAV before will overwrite the original value
            for i in list_of_users:
                if i in updated_users:
                    # print("user ", i, " was updated")
                    BS_age_1[i] = UAV_age_1[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                else:
                    # print("user ", i, " was not updated")
                    BS_age_1[i] = BS_age_1[i] + 1

        # updating to BS ends

        # age at BS is updated before age at UAV as age at BS of the next slot depends on the age at UAV of current slot, and updating age at UAV before will overwrite age at UAV

        # print("updating process for each drone done, updated users are ", updated_users)
        # if time==run_time-1:
        #     print(len(updated_users)," users updated] out of ", users, " in the random sampling and updating policy")
        # which of these two will happen first ? seems like updating has to change first ??

        for i in list_of_users:
            if i in sampled_users:
                # print("user ", i, " was sampled")
                UAV_age_1[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
            else:
                # print("user ", i, " was sampled")
                UAV_age_1[i] = UAV_age_1[i] + 1

        # fix a user and show his age at each stage
        if time==1:
            xx = np.random.choice(list(UAV_age_1.keys()))

        ## to see age progression, comment out the following lines
        # print("time = ", time)
        # print("sampled_users = ", sampled_users)
        # print("Age at UAV ", UAV_age_1)
        # print("updated_users = ", updated_users)
        # print("Age at BS ", BS_age_1)


        # print("results for user ", xx, " will be shown ")
        # # check age evolution of user at each time step
        # if xx in sampled_users:
        #     print("t = ", time, ", user ", xx, " was sampled and age at UAV is ", UAV_age_1[xx])
        # else:
        #     print("t = ", time, ", user ", xx, " NOT sampled and age at UAV is ", UAV_age_1[xx])

        # if user in updated_users:
        #     print("t = ", time, ", user ", xx, " was updated and age at  BS is ", BS_age_1[xx])
        # else:
        #     print("t = ", time, ", user ", xx, " NOT updated and age at  BS is ", BS_age_1[xx])
        


    UAV_age_1 = collections.OrderedDict(sorted(UAV_age_1.items())) # sort as per user IDs
    BS_age_1 = collections.OrderedDict(sorted(BS_age_1.items())) # sort as per user IDs

    # print("UAV_age_1 = ", UAV_age_1)
    # print("BS_age_1 = ", BS_age_1)
    return UAV_age_1, BS_age_1



def age_greedy(drone_2, coverage_2, users, list_of_users, UAV_age_2, BS_age_2):

    print("Greedy scheduling")

    for time in range(1, run_time): # for every time slot
        # print("\n")
        sampled_users = [] # list of users sampled by the UAV
        updated_users = [] # list of users updated by the UAV

        # sampling by UAV part starts

        for j in range(drone_2): # for every drone
            users_covered = coverage_2[j] # list containing all users covered
            if len(users_covered) < UAV_capacity: # all users under drone j is sampled
                # print('for drone ', j,", covered users = ", len(users_covered), " which is less than UAV capacity of ", UAV_capacity)
                for user in users_covered:
                    sampled_users.append(user)
                # print("sampled_users = ", sampled_users)
            
            else: # all users can't be served only UAV_capacity users can be served. Do it greedily
                # print('for drone ', j,", covered users = ", len(users_covered), " which is more than UAV capacity of ", UAV_capacity)
                covered_users_age = {}
                for k in users_covered:
                    covered_users_age[k] = UAV_age_2[k] ## covered user age will be age at UAV
                covered_users_age = {k: v for k, v in sorted(covered_users_age.items(), key=lambda item: item[1], reverse = True)} ## order age at UAV for greedy selection
                out = dict(itertools.islice(covered_users_age.items(), UAV_capacity))  # select the users with highest age
                # print("covered_users_age = ", covered_users_age)
                # print("greedily selected users = ", out)
                to_sample = list(out.keys())


                for user in to_sample:
                    sampled_users.append(user)
                # print("sampled_users = ", sampled_users)
        
        # sampling by UAV part ends

        # print("sampling process for each drone done, sampled users are ", sampled_users)
        # if time==run_time-1:
        #     print(len(sampled_users)," users sampled out of ", users, " in the random sampling and updating policy")
        # every drone has added which users to sample in sampled_users, next is to select which user will get updated


        # updating to BS starts

        if time==1: #  # at the first slot no user is updated and age becomes 2 at the BS for all of the users, see table 1 of paper - Optimal Scheduling Policy for Minimizing Age of Information with a Relay
            for i in list_of_users:
                BS_age_2[i] = BS_age_2[i]+1 


        else:
            for j in range(drone_2): # for every drone
                users_covered = coverage_2[j] # list containing all users covered
                if len(users_covered) < UAV_capacity: # all users under drone j is updated
                    for user in users_covered:
                        updated_users.append(user)
                    # print("updated_users = ", updated_users)
            
                else: # all users can't be served only UAV_capacity users can be served
                    covered_users_age = {}
                    for k in users_covered:
                        covered_users_age[k] = BS_age_2[k] ## covered user age will be age at BS
                    covered_users_age = {k: v for k, v in sorted(covered_users_age.items(), key=lambda item: item[1], reverse = True)} ## order age at UAV for greedy selection
                    out = dict(itertools.islice(covered_users_age.items(), UAV_capacity))  # select the users with highest age
                    # print("covered_users_age = ", covered_users_age)
                    # print("greedily selected users = ", out)
                    to_sample = list(out.keys())

                    for user in to_sample:
                        updated_users.append(user)
                    # print("updated_users = ", updated_users)

            # updating to BS ends

            # age at BS is updated before age at UAV as age at BS of the next slot depends on the age at UAV of current slot, and updating age at UAV before will overwrite age at UAV
            for i in list_of_users:
                if i in updated_users:
                    # print("user ", i, " was updated")
                    BS_age_2[i] = UAV_age_2[i] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                else:
                    # print("user ", i, " was not updated")
                    BS_age_2[i] = BS_age_2[i] + 1

        # print("updating process for each drone done, updated users are ", updated_users)
        # if time==run_time-1:
        #     print(len(updated_users)," users updated] out of ", users, " in the random sampling and updating policy")
        # which of these two will happen first ? seems like updating has to change first ??

        for i in list_of_users:
            if i in sampled_users:
                # print("user ", i, " was sampled")
                UAV_age_2[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
            else:
                # print("user ", i, " was sampled")
                UAV_age_2[i] = UAV_age_2[i] + 1

        # fix a user and show his age at each stage
        if time==1:
            xx = np.random.choice(list(UAV_age_2.keys()))

        ## to see age progression, comment out the following lines
        print("time = ", time)
        print("sampled_users = ", sampled_users)
        print("Age at UAV ", UAV_age_2)
        print("updated_users = ", updated_users)
        print("Age at BS ", BS_age_2)


        print("results for user ", xx, " will be shown ")
        # check age evolution of user at each time step
        if xx in sampled_users:
            print("t = ", time, ", user ", xx, " was sampled and age at UAV is ", UAV_age_2[xx])
        else:
            print("t = ", time, ", user ", xx, " NOT sampled and age at UAV is ", UAV_age_2[xx])

        if user in updated_users:
            print("t = ", time, ", user ", xx, " was updated and age at  BS is ", BS_age_2[xx])
        else:
            print("t = ", time, ", user ", xx, " NOT updated and age at  BS is ", BS_age_2[xx])


    UAV_age_2 = collections.OrderedDict(sorted(UAV_age_2.items()))
    BS_age_2 = collections.OrderedDict(sorted(BS_age_2.items()))

    # print("Final UAV_age_2 = ", UAV_age_2)
    
    # age evolution check block finished
        
    return UAV_age_2, BS_age_2
