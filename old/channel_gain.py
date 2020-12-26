'''
calculates channel gain based on pathloss
'''

from parameters import *

def channel_gain(path_loss):
    return (10**(path_loss/10))
