'''
Initialize constants.
All values in SI unless specified
Unverified values have been commented with 'REMAINING'


# only create_graph_1.py file uses this file
'''


# imptt: L,B should be greater than the maximum number of users otherwise x/y locations possible becomes less than x/y locations available

L = 50 # length of whole region in meters
B = 50 # breadth of whole region in meters


fc = 2 * 10**9       # carrier frequency. 2 GHz, https://arxiv.org/pdf/1704.04813.pdf page 21
c = 3 * 10**8         # speed of light

n_L = 3 # LOS System loss in dB, https://arxiv.org/pdf/1704.04813.pdf page 21
n_N = 23 # NLOS system loss in dB  " do "


a = 9.61 # environmental parameters for path loss, from https://arxiv.org/pdf/1702.08395.pdf
b = 0.16 
# LOS link probability , will be calculated from path_loss_probability.py

Pt = 0.5 # Drone transmit power in Watt, https://arxiv.org/pdf/1704.04813.pdf  page 21

BW = 10**6 # 1 MHz,  https://arxiv.org/pdf/1704.04813.pdf  page 21

# Pn = from_dBm(-170) * BW # Watt, https://arxiv.org/pdf/1704.04813.pdf  page 21

# SNR_th = from_dB(10) # https://arxiv.org/pdf/1801.05000.pdf

r = 10 # grid length and breadth

# R = round(r*(2**0.5), 3) # REMAINING, edge from a vertex a to b exist if distance between i and j < R
R = 2*r # ((2)**0.5)*r ## R > r for graph creation

