'''
2 functions to calculate the LOS and NLOS loss. 
    loss_LOS(d):
    loss_NLOS(d):

returns in linear scale
'''


from parameters import *
from dB_linear import *
import math

pi = math.pi

def loss_LOS(d):
    loss_val = from_linear( (4*pi*fc*d) / (c) + n_L )
    return from_dB(loss_val)

def loss_NLOS(d):
    loss_val = from_linear( (4*pi*fc*d) / (c) + n_N )
    return from_dB(loss_val)