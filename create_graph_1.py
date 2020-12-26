## new graph where drones can be deployed over the grid points.

from scipy.spatial.distance import pdist, squareform
from networkx import *
from networkx.algorithms.approximation import min_weighted_dominating_set
from random import choice
import collections
import copy
# from networkx.algorithms.approximation import dominating_set

import matplotlib.pyplot as plt
import numpy as np
from parameters import *
import random

from tf_environment import *

random.seed(42)


def create_graph_1(user_coordinates, grid_coordinates, str):
      
    A_1 = np.array(grid_coordinates)
    A_2 = np.array(user_coordinates)
    A = np.append(A_1, A_2, axis = 0)
    # print(type(A), np.shape(A_1), np.shape(A_2))
    # print("A_1 = ", A_1)
    # print("A_2 = ", A_2)
    # print("A_3 = ", A)
    B = squareform(pdist(A, metric='euclidean'))
    # print("B=", B)
    G = nx.from_numpy_matrix(B)
    position_dict = {} # dict containing all positions of users and grid points
    for i in range(len(A)):
        position_dict[i] = [A[i][0], A[i][1]]

    position_dict_1 = copy.deepcopy(position_dict) 
    position_dict_2 = copy.deepcopy(position_dict) 

    grid_start_index = 0 
    grid_end_index = len(grid_coordinates) - 1
    user_start_index = grid_end_index + 1
    user_end_index = len(user_coordinates) + len(grid_coordinates) - 1

    print("indexes follow ", grid_start_index, grid_end_index, user_start_index, user_end_index)


    color_map = []
    for node in G:
        if node < user_start_index:
            color_map.append('yellow')
        else: 
            color_map.append('green')   


    # fig, ax1 = plt.subplots()
    # nx.draw_networkx(G, pos = position_dict, node_color=color_map, with_labels=True)
    # plt.title("Original Graph")
    # plt.show()

    G.remove_edges_from(list(G.edges())) # remove all edges
    # fig, ax1 = plt.subplots()
    # nx.draw_networkx(G, pos = position_dict, node_color=color_map, with_labels=True)
    # plt.title("Edge removed Graph")
    # plt.show()

    drone_ids = list(np.linspace(grid_start_index, grid_end_index, num = len(grid_coordinates))) 
    user_ids = list(np.linspace(user_start_index, user_end_index, num = len(user_coordinates))) # for MDS

    drone_ids_1 = copy.deepcopy(drone_ids)
    user_ids_1 = copy.deepcopy(user_ids) # for random

    drone_ids_2 = copy.deepcopy(drone_ids)
    user_ids_2 = copy.deepcopy(user_ids) # for MCDS

    # print('drone ids ', drone_ids )
    # print('user_ids ', user_ids )



    # for i in drone_ids:
    #     for j in user_ids:
    #         print(G.get_edge_data(i,j)) # all node as no edges
    

    for i in range(len(grid_coordinates)): # add edges between drone loc and user based on distance
        for j in range(len(user_coordinates)):
            # print(i, j, grid_coordinates[i], user_coordinates[j])
            distance = ((grid_coordinates[i][0] -  user_coordinates[j][0])**2 + (grid_coordinates[i][1] -  user_coordinates[j][1])**2)**0.5
            # print("grid = ", i, "user = ", j + user_start_index, grid_coordinates[i], user_coordinates[j], distance)
            if distance < R:
                G.add_edge(i, j+user_start_index, weight=distance)
                # print("grid location ", user_coordinates[i], "user location = ", user_coordinates[j])
                # print("distance between ", i," and ", j + user_start_index, "  is ", distance, " so edge added")
            # else:
            #     G.add_edge(i, j+user_start_index, weight=None)


    labels = nx.get_edge_attributes(G,'weight')

    # print("labels = ", labels )

    # fig, ax1 = plt.subplots()

    # plt.xlabel("X-axis (m)")
    # plt.ylabel("Y-axis (m)")
    # # nx.draw_networkx_edge_labels(G,position_dict) #,edge_labels=labels)

    # nx.draw_networkx(G, pos = position_dict, node_color=color_map) #, with_labels=True)
    # plt.title("Coverage Plot for '{0}' users with R='{1}'".format(len(user_coordinates), R))    
    # plt.grid()


    ## plot the needed drones with all users
    # print("position dict = ", position_dict)
    # print("drones_needed_MDS = ", drones_needed_MDS)
    # MDS_drone_coordinates = []
    # for drone in drones_needed_MDS:
    #     MDS_drone_coordinates.append(position_dict[drone])
    # print("MDS_drone_locations = ", MDS_drone_coordinates)

    # construct_graph(user_coordinates, MDS_drone_coordinates) uncomment to see the resulting drone coverage with the selected drones under MDS

    if str=="RP":
        H = G.__class__() # deep copy
        H.add_nodes_from(G)
        H.add_edges_from(G.edges)
        drones_needed_random, m, drone_coverage_random_overall = random_selection(H, drone_ids_1, user_ids_1, position_dict_2) # drones_needed_random is a list
        assert (m == 0) # all users have to be removed as removed means under coverage
        return (len(drones_needed_random), drone_coverage_random_overall)
    
    if str=="MDS":
        J = G.__class__() # deep copy
        J.add_nodes_from(G)
        J.add_edges_from(G.edges)
        drones_needed_MDS, n, drone_coverage_MDS_overall = custom_mds(J, drone_ids, user_ids, position_dict_1) # drones_needed_MDS is a list
        assert (n == 0) # all users have to be removed as removed means under coverage
        return (len(drones_needed_MDS), drone_coverage_MDS_overall)

    # return (len(drones_needed_random), drone_coverage_random_overall, len(drones_needed_MDS), drone_coverage_MDS_overall)

    
def custom_mds(G, drone_ids, user_ids, position_dict):
    # print("position dict at beginning is ", position_dict)
    # print("custom user ids are ", user_ids)
    selected_nodes_list = [] # will contain the drone deployment locations
    selected_nodes_coverage_overall = [] # will contain the list of users covered by the drone at same index in selected_nodes_list, a list of lists
    while (len(user_ids) > 0): # number of users, and it has to be 0 if all users covered
    # print(G.number_of_nodes() - len(drone_ids))
        degree_list = G.degree(drone_ids) # only drone's degrees are used
        # print(degree_list) # working correctly
        degree_list = sorted(degree_list, key=lambda x: x[1], reverse = True)
        # print("sorted degree list = ", degree_list) # working correctly
        selected_node = degree_list[0][0] # 1st node in the tuple
        # print("selected node is ", selected_node)
        neigh = list(G.neighbors(selected_node))
        if len(neigh) == 0:
            # print("node ", selected_node, " has no neighbor")
            # selected_nodes_list.append(selected_node)
            # print("remaining users = ", len(user_ids))
            G.remove_node(selected_node) # here this node is removed so that this node is not selected again
            drone_ids.remove(selected_node)
            del position_dict[selected_node]
            # print("node ", selected_node, " removed for no neighbor")
            # print("position dict becomes ", position_dict)

        else:
            # print("node ", selected_node, " has neighbors neighbors = ", neigh)
            selected_nodes_list.append(selected_node)
            selected_nodes_coverage = [] # list corresponding to every drone
            # print("node ", selected_node, " selected" )
            for i in neigh:
                G.remove_node(i)
                user_ids.remove(i)
                del position_dict[i]
                selected_nodes_coverage.append(i)
                if len(selected_nodes_coverage) > coverage_capacity-1:
                    break # no more neighbors will be added
                # print("neighbor ", i, " removed")
                # print("remaining users = ", (user_ids))
                # print("position dict becomes ", position_dict)
            G.remove_node(selected_node)
            drone_ids.remove(selected_node)
            del position_dict[selected_node]
            selected_nodes_coverage_overall.append(selected_nodes_coverage)
            # print("position dict becomes ", position_dict)

    # fig, ax1 = plt.subplots()

    # plt.xlabel("X-axis (m)")
    # plt.ylabel("Y-axis (m)")
    # nx.draw_networkx_edge_labels(G, position_dict) #,edge_labels=labels)

    # nx.draw_networkx(G, pos = position_dict, node_color=color_map) #, with_labels=True)
    # nx.draw_networkx(G, pos = position_dict)
    # plt.title("Coverage Plot for '{0}' users with R='{1}'".format(len(user_coordinates), R))    
    # plt.grid()
    
    # print("selected drones for custom MDS are ", selected_nodes_list, " whose length is  ", len(selected_nodes_list))
    return ((selected_nodes_list), len(user_ids), selected_nodes_coverage_overall) # list of selected drones, user ids to check no user is left, selected_nodes_coverage to see number of users covered by each node

def random_selection(G, drone_ids, user_ids, position_dict):
    selected_nodes_list = []
    selected_nodes_coverage_overall = []
    # print("random user ids are ", user_ids)
    while (len(user_ids) > 0): # number of users, and it has to be 0 if all users covered
        selected_node = random.choice(drone_ids) 
        # print("selected node is ", selected_node)
        neigh = list(G.neighbors(selected_node))
        if len(neigh) == 0:
            # this drone doesn't cover any user
            # print("node ", selected_node, " has no neighbor")
            # selected_nodes_list.append(selected_node) # not covering anyone so not selected
            # print("remaining users = ", len(user_ids))
            G.remove_node(selected_node) # keep in graph but remove from list so that not selected again
            drone_ids.remove(selected_node) 
            # print("node ", selected_node, " removed")

        else:
            # print("node ", selected_node, " has neighbors neighbors = ", neigh)
            selected_nodes_list.append(selected_node)
            selected_nodes_coverage = [] # list corresponding to every drone
            # print("node ", selected_node, " selected" )
            for i in neigh:
                G.remove_node(i)
                user_ids.remove(i)
                selected_nodes_coverage.append(i)
                if len(selected_nodes_coverage) > coverage_capacity-1:
                    break # no more neighbors will be added

                # print("node ", i, " removed")
                # print("remaining users = ", len(user_ids))
            G.remove_node(selected_node)
            drone_ids.remove(selected_node)
            selected_nodes_coverage_overall.append(selected_nodes_coverage)
            # print("node ", selected_node, " removed")

    # print("selected drones for random are ", selected_nodes_list, " whose length is  ", len(selected_nodes_list))

    return ((selected_nodes_list), len(user_ids), selected_nodes_coverage_overall) #, G.number_of_nodes())

    
def construct_graph(user_coordinates, drone_coordinates): # no manipulation of lists, only construct the graph between drone-users based on drone's ground coverage for plotting
    A_1_new = np.array(drone_coordinates)
    A_2_new = np.array(user_coordinates)
    A_new = np.append(A_1_new, A_2_new, axis = 0)
    B_new = squareform(pdist(A_new, metric='euclidean'))
    G_new = nx.from_numpy_matrix(B_new)
    position_dict_new = {} # dict containing all positions of users and grid points
    for i in range(len(A_new)):
        position_dict_new[i] = [A_new[i][0], A_new[i][1]]

    grid_start_index = 0 
    grid_end_index = len(drone_coordinates) - 1
    user_start_index = grid_end_index + 1
    user_end_index = len(user_coordinates) + len(drone_coordinates) - 1

    # print("indexes follow = ",grid_start_index, grid_end_index, user_start_index, user_end_index )


    color_map = []
    for node in G_new:
        if node < user_start_index:
            color_map.append('yellow')
        else: 
            color_map.append('green') 

    G_new.remove_edges_from(list(G_new.edges()))

    for i in range(len(drone_coordinates)): # add edges between drone loc and user based on distance
        for j in range(len(user_coordinates)):
            distance = ((drone_coordinates[i][0] -  user_coordinates[j][0])**2 + (drone_coordinates[i][1] -  user_coordinates[j][1])**2)**0.5
            if distance < R:
                G_new.add_edge(i, j+user_start_index, weight=distance)


    # fig, ax1 = plt.subplots()

    # plt.xlabel("X-axis (m)")
    # plt.ylabel("Y-axis (m)")
    # # nx.draw_networkx_edge_labels(G,position_dict) #,edge_labels=labels)

    # nx.draw_networkx(G_new, pos = position_dict_new, node_color=color_map) #, with_labels=True)
    # plt.title("Plot for drone coverage")
    # plt.grid()