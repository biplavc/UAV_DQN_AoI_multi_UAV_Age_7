import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from create_graph import *
from create_graph_1 import *

from path_loss_probability import *
from age_calculation import *
from itertools import product  
import pickle


from parameters import *

random.seed(3)

def start_simulation():
    for I in users:
        # I is number of users, L length and B breadth
        x_vals = random.sample(range(1, L-1), I) # x-coordinates for users
        y_vals = random.sample(range(1, B-1), I) # y-coordinates for users
        z_vals = [0]*I

        user_coordinates = list(zip(x_vals,y_vals))

        x_grid_nos =(L/r) + 1 # number of different values the grid takes for x axis
        y_grid_nos = (B/r) + 1 # number of different values the grid takes for y axis

        grid_x = np.linspace(0, L, num = x_grid_nos) # generate evenly spaced x positions for grid
        grid_y = np.linspace(0, B, num = y_grid_nos) # generate evenly spaced y positions for grid
 
        grid_coordinates = list(product(grid_x , grid_y)) 

        x_points = [x[0] for x in grid_coordinates]
        y_points = [x[1] for x in grid_coordinates]

        # fig, ax1 = plt.subplots()

        # plt.scatter(x_vals, y_vals, label = 'user positions')
        # plt.scatter(x_points, y_points, label = 'drone deployment positions')
        # plt.xlabel("X-axis (m)")
        # plt.ylabel("Y-axis (m)")
        # plt.legend(loc='best', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=4)

        drones_needed_random, drone_coverage_random = create_graph_1(user_coordinates, grid_coordinates, "random")
        drones_needed_MDS, drone_coverage_MDS = create_graph_1(user_coordinates, grid_coordinates, "MDS") 
        # print("drones_needed_random = ", drones_needed_random, "drones_needed_MDS = ", drones_needed_MDS)
        

        drones_Random.append(drones_needed_random) # for plotting the drones needed result
        drones_CustomMDS.append(drones_needed_MDS)


        ## IMPTT :
        ## for the next part, drones are first selected from above and then the selected drones are re-indexed sequentially. So a drone that had id 4 before could have id 2 now
        ## but the user's IDs are kept same as here. So drones can be accessed using range(no of drones) but users under drones are kept in a dictionary
        '''
        drones = number of drones deployed, int
        coverage = list of users under each drone, list of list. list at index j will be the list of users under drone jth drone selected, not necessarily the jth drone in sequence
        
        # results for random placement
        # RP means random placement, GS means greedy sampling updating, RS means random placement

        UAV_age_RP_RS, BS_age_RP_RS, UAV_age_RP_GS, BS_age_RP_GS = age_calculation(drones_needed_random, drone_coverage_random)
        avg_UAV_age_RP_RS = sum(UAV_age_RP_RS.values())*1.00/len(UAV_age_RP_RS)
        avg_BS_age_RP_RS = sum(BS_age_RP_RS.values())*1.00/len(BS_age_RP_RS)
        avg_UAV_age_RP_GS = sum(UAV_age_RP_GS.values())*1.00/len(UAV_age_RP_GS)
        avg_BS_age_RP_GS = sum(BS_age_RP_GS.values())*1.00/len(BS_age_RP_GS)

        # assert len(UAV_age_RP_RS) == 50
        # assert len(BS_age_RP_RS)  == 50
        # assert len(UAV_age_RP_GS) == 50
        # assert len(BS_age_RP_GS)  == 50

        # results for MDS placement
        # MP means MDS placement, GS means greedy sampling updating, RS means random placement
        UAV_age_MP_RS, BS_age_MP_RS, UAV_age_MP_GS, BS_age_MP_GS = age_calculation(drones_needed_MDS, drone_coverage_MDS)
        avg_UAV_age_MP_RS = sum(UAV_age_MP_RS.values())*1.00/len(UAV_age_MP_RS)
        avg_BS_age_MP_RS = sum(BS_age_MP_RS.values())*1.00/len(BS_age_MP_RS)
        avg_UAV_age_MP_GS = sum(UAV_age_MP_GS.values())*1.00/len(UAV_age_MP_GS)
        avg_BS_age_MP_GS = sum(BS_age_MP_GS.values())*1.00/len(BS_age_MP_GS)
        # assert len(UAV_age_MP_RS) == 50
        # assert len(BS_age_MP_RS)  == 50
        # assert len(UAV_age_MP_GS) == 50
        # assert len(BS_age_MP_GS)  == 50

        UAV_AOI_RP_RS.append(avg_UAV_age_RP_RS)
        BS_AOI_RP_RS.append(avg_BS_age_RP_RS)

        ## results for random sampling and MDS placement

        UAV_AOI_MP_RS.append(avg_UAV_age_MP_RS)
        BS_AOI_MP_RS.append(avg_BS_age_MP_RS)

        ## results for greedy sampling and random placement

        UAV_AOI_RP_GS.append(avg_UAV_age_RP_GS)
        BS_AOI_RP_GS.append(avg_BS_age_RP_GS)

        ## results for greedy sampling and MDS placement

        UAV_AOI_MP_GS.append(avg_UAV_age_MP_GS)
        BS_AOI_MP_GS.append(avg_BS_age_MP_GS)

        '''


    ### plot results

    # plot the number of drones needed in the two approaches
    # fig, ax1 = plt.subplots()

    # ax1.plot(users, drones_Random, 'k', marker='+', label = 'Random with removal')
    # ax1.plot(users, drones_CustomMDS, 'g', marker='^', label = 'Custom MDS')

    # legend = ax1.legend(loc='best', shadow=False, fontsize='large')
    # # plt.xlabel('Simulation Time')
    # plt.xlabel('Number of ground users')
    # plt.ylabel('Number of drones needed')
    # # ax1.set_xticks(T)
    # legend.get_frame().set_facecolor('C0')



    # plt.show()

if __name__ == '__main__':

    users = [10, 50, 100, 150, 250, 350, 450] #, 550, 650, 750, 850]
    print("users = ", users)
    drones_Random = [] # contain number of drones for every different number of users
    drones_CustomMDS = [] # ''

    ## results for random sampling and random placement

    UAV_AOI_RP_RS = []
    BS_AOI_RP_RS = []

    ## results for random sampling and MDS placement

    UAV_AOI_MP_RS = []
    BS_AOI_MP_RS = []

    ## results for greedy sampling and random placement

    UAV_AOI_RP_GS = []
    BS_AOI_RP_GS = []

    ## results for greedy sampling and MDS placement

    UAV_AOI_MP_GS = []
    BS_AOI_MP_GS = []

    start_simulation() 
    # print(path_loss_probability_LOS(100,300)) # format h/r
    # print(SNR_th)

    print("drones_random = ", drones_Random)
    print("drones_MSD = ", drones_CustomMDS)

    '''
    pickle.dump(drones_Random, open("drones_Random.pickle", "wb"))
    pickle.dump(drones_CustomMDS, open("drones_CustomMDS.pickle", "wb"))


    pickle.dump(UAV_AOI_RP_RS, open("UAV_AOI_RP_RS.pickle", "wb"))
    pickle.dump(BS_AOI_RP_RS, open("BS_AOI_RP_RS.pickle", "wb"))
    pickle.dump(UAV_AOI_MP_RS, open("UAV_AOI_MP_RS.pickle", "wb"))
    pickle.dump(BS_AOI_MP_RS, open("BS_AOI_MP_RS.pickle", "wb"))
    
    pickle.dump(UAV_AOI_RP_GS, open("UAV_AOI_RP_GS.pickle", "wb"))
    pickle.dump(BS_AOI_RP_GS, open("BS_AOI_RP_GS.pickle", "wb"))
    pickle.dump(UAV_AOI_MP_GS, open("UAV_AOI_MP_GS.pickle", "wb"))
    pickle.dump(BS_AOI_MP_GS, open("BS_AOI_MP_GS.pickle", "wb"))



    drones_Random = pickle.load(open("drones_Random.pickle", "rb"))
    drones_CustomMDS = pickle.load(open("drones_CustomMDS.pickle", "rb"))

    UAV_AOI_RP_RS = pickle.load(open("UAV_AOI_RP_RS.pickle", "rb"))
    BS_AOI_RP_RS = pickle.load(open("BS_AOI_RP_RS.pickle", "rb"))
    UAV_AOI_MP_RS = pickle.load(open("UAV_AOI_MP_RS.pickle", "rb"))
    BS_AOI_MP_RS = pickle.load(open("BS_AOI_MP_RS.pickle", "rb"))

    UAV_AOI_RP_GS = pickle.load(open("UAV_AOI_RP_GS.pickle", "rb"))
    BS_AOI_RP_GS = pickle.load(open("BS_AOI_RP_GS.pickle", "rb"))
    UAV_AOI_MP_GS = pickle.load(open("UAV_AOI_MP_GS.pickle", "rb"))
    BS_AOI_MP_GS = pickle.load(open("BS_AOI_MP_GS.pickle", "rb"))

    fig, ax1 = plt.subplots()

    ax1.plot(users, drones_Random, 'k', marker='+', label = 'Random with removal')
    ax1.plot(users, drones_CustomMDS, 'b', marker='^', label = 'Custom MDS')

    legend = ax1.legend(loc='best', shadow=False, fontsize='large')
    # plt.xlabel('Simulation Time')
    plt.xlabel('Number of ground users')
    plt.ylabel('Number of drones needed')
    # ax1.set_xticks(T)
    legend.get_frame().set_facecolor('C0')

    ########## Age at UAV

    # fig, ax1 = plt.subplots()

    # ax1.plot(users, UAV_AOI_RP_RS, 'k', marker='+', label = 'random plc random sch')
    # ax1.plot(users, UAV_AOI_RP_GS, 'm', marker='_', label = 'Random plc greedy sch')
    # ax1.plot(users, UAV_AOI_MP_RS, 'g', marker='^', label = 'MDS plc random sch')
    # ax1.plot(users, UAV_AOI_MP_GS, 'b', marker='D', label = 'MDS plc greedy sch')

    # legend = ax1.legend(loc='best', shadow=False, fontsize='large')
    # # plt.xlabel('Simulation Time')
    # plt.xlabel('Number of ground users')
    # plt.ylabel('Age at UAV relay')
    # # ax1.set_xticks(T)
    # legend.get_frame().set_facecolor('C0')

    ########## Age at BS

    fig, ax1 = plt.subplots()

    ax1.plot(users, BS_AOI_RP_RS, 'k', marker='+', label = 'Random placement + random scheduling')
    ax1.plot(users, BS_AOI_RP_GS, 'm', marker='_', label = 'Random placement + greedy scheduling')
    ax1.plot(users, BS_AOI_MP_RS, 'g', marker='^', label = 'MDS placement + random scheduling')
    ax1.plot(users, BS_AOI_MP_GS, 'b', marker='D', label = 'MDS placement + greedy scheduling')

    legend = ax1.legend(loc='best', shadow=False, fontsize='large')
    # plt.xlabel('Simulation Time')
    plt.xlabel('Number of ground users', fontsize='large')
    plt.ylabel('Age at final BS', fontsize='large')
    # ax1.set_xticks(T)

    legend.get_frame().set_facecolor('C0')
    plt.show()


    '''
    print("simulation finished")


