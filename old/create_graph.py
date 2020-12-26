# create_graph(coordinates)
## based on the old idea where drones can only be deployed over users

from scipy.spatial.distance import pdist, squareform
from networkx import *
from networkx.algorithms.approximation import min_weighted_dominating_set
from random import choice
import collections

# from networkx.algorithms.approximation import dominating_set

import matplotlib.pyplot as plt
import numpy as np
from parameters import *
import random

coverage_capacity = 5

def create_graph(coordinates, x_vals, y_vals, I):
    A = np.array(coordinates)
    B = squareform(pdist(A, metric='euclidean'))
    # print("B=", B)
    G = nx.from_numpy_matrix(B)
    position_dict = {}
    for i in range(I):
        position_dict[i] = [x_vals[i], y_vals[i]]

    # remove edges longer than R

    # nx.draw_networkx(G, with_labels = True, pos = position_dict) # fully connected graph
    # plt.title("Original Graph")
    # plt.show()
    # print("weight = ", G.get_edge_data(2,60))
    for i in range(I):
        for j in range(I):
            if G.get_edge_data(i,j) !=None:
                if (G.get_edge_data(i,j)['weight']>R):
                    # print("i=",i, "j=",j)
                    G.remove_edge(i,j)

    # find_degree(G, I)
    ds_val = analyze_graph(G, position_dict, x_vals, y_vals) # function below

    # create a fresh copy of the graph for the random selection
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    random_val = random_selection_removal(H)

    J = G.__class__()
    J.add_nodes_from(G)
    J.add_edges_from(G.edges)
    custom_val = custom_mds(J)
    return custom_val, random_val, ds_val


    
def analyze_graph(G, position_dict, x_vals, y_vals):
    # keys = [i for i in range(I)]
    # nx.draw_networkx(G, with_labels = True, pos = position_dict) # graph with edge's existence conditioned on node's distance
    # plt.title("Final Graph")
    # plt.show()
    # print("nodes=", G.number_of_nodes(), "edges=", G.number_of_edges())
    
    # for i in range(I):
    #     for j in range(I):
            # print( "i=",i,"j=",j,G.get_edge_data(i,j))
    ds_val = get_dominating_set(G, x_vals, y_vals)
    return ds_val

def get_dominating_set(G, x_vals, y_vals):
    vertices_1 = min_weighted_dominating_set(G)
    vertices_2 = dominating_set(G)
    # print("no of chosen vertices with min_weighted_dominating_set are ", len(vertices_1)) # "and they are ", vertices_1)
    # print("no of chosen vertices with dominating_set are ", len(vertices_2)) # "and they are ", vertices_2)
    # print("weight = ", G.get_edge_data(2,60))
    # compare_graph(vertices_1, vertices_2 , x_vals, y_vals)
    return len(vertices_2)

'''
def compare_graph(vertices_1, vertices_2 , x_vals, y_vals):

    x_new_1 = [x_vals[i] for i in vertices_1]
    y_new_1 = [y_vals[i] for i in vertices_1]

    x_new_2 = [x_vals[i] for i in vertices_2]
    y_new_2 = [y_vals[i] for i in vertices_2]


    a = plt.scatter( x_vals, y_vals, label = 'All users')
    b = plt.scatter( x_new_1, y_new_1, label = 'Selected Users')
    # plt.title("Result of min_weighted_dominating_set")
    plt.legend(loc='best', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=4)
    # plt.show()

    a = plt.scatter( x_vals, y_vals, label = 'All users')
    b = plt.scatter( x_new_2, y_new_2, label = 'Selected Users')
    plt.xlabel('X Coordinates', fontsize = 20)
    plt.ylabel('Y Coordinates', fontsize = 20)
    plt.title("Result of dominating_set", fontsize = 20)
    plt.legend(loc='best', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4, fontsize = 20)
    # plt.show()
'''

def random_selection_removal(G): # receiving H but using G here

    selected_nodes_list = []
    while (G.number_of_nodes() > 0):
        all_nodes = list(G.nodes)
        n_old = len(all_nodes)
        # print("no of nodes = ", n_old, " and they are ", all_nodes)
        selected_node = random.choice(all_nodes) 
        # print("selected_node = ", selected_node)
        neigh = list(G.neighbors(selected_node))
        if len(neigh) == 0:
            # print("node ", selected_node, " has no neighbor")
            selected_nodes_list.append(selected_node)
            G.remove_node(selected_node)
        
        else:
            # print("neighbors = ", neigh)
            selected_nodes_list.append(selected_node)
            G.remove_node(selected_node)
            for i in neigh:
                G.remove_node(i)
                # print("node ", i, " removed")
                # neigh = G.neighbors(selected_node)
                # print("new n = ", G.number_of_nodes())

    # print("random selection with removal has ", len(selected_nodes_list), " nodes") # and they are ", selected_nodes_list)
    return len(selected_nodes_list)

def custom_mds(G):
    selected_nodes_list = []
    while (G.number_of_nodes() > 0):
        degree_list = G.degree()
        degree_list = sorted(degree_list, key=lambda x: x[1], reverse = True)
        # print(degree_list)
        selected_node = degree_list[0][0] # 1st node in the tuple
        neigh = list(G.neighbors(selected_node))
        if len(neigh) == 0:
            # print("node ", selected_node, " has no neighbor")
            selected_nodes_list.append(selected_node)
            G.remove_node(selected_node)

        else:
            # print("neighbors = ", neigh)
            selected_nodes_list.append(selected_node)
            G.remove_node(selected_node)
            for i in neigh:
                G.remove_node(i)
                # print("node ", i, " removed")
                # print("new n = ", G.number_of_nodes())

    # print('custom MDS has ', len(selected_nodes_list), ' nodes') # and they are ', selected_nodes_list)
    return len(selected_nodes_list)

'''
def find_degree(G, I):
    # print('G.degree() = ',G.degree())
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence

    # print ("Degree sequence = ", degree_sequence)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram for %i users" %I)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)

    # # draw graph in inset
    # plt.axes([0.4, 0.4, 0.5, 0.5])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # pos = nx.spring_layout(G)
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)

    plt.show()

'''