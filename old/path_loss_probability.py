# h is height of UAV above UE and r is horizontal distance between UE and UAV

from parameters import * 
import math

def path_loss_probability_LOS(h, r):
    theta_rad = math.atan(h/r) # radians
    theta_deg = 180*theta_rad/(math.pi) # degrees
    p_LOS = 1/(1+a*(math.exp(-b*(theta_deg - a))))
    return p_LOS