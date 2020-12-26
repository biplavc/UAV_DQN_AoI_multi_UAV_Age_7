'''
calculate SNR in linear scale
'''

def SNR(p_drone, gain, p_noise):
    val = (p_drone * gain) / p_noise**2
    return val