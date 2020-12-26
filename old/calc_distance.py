'''
taken in 2 3d coordinates and returns the distance in meters

'''
import math

def distance(p1, p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i])**2
    
    return math.sqrt(dist)