'''
3 functions, one for dB to linear (watts) and the other for dBm to linear

def from_dB(val) - dB to linear
def from_dBm(val) - dBm to linear
def from_linear(val) - linear to dB.

db = 10 * log(A)

''' 
import control
# import parameters

def from_dB(db_val):
    return (10**(db_val*0.1))


def from_dBm(dbm_val):
    db_val = dbm_val - 30
    return from_dB(db_val)

def from_linear(linear_val):
    db_val = 10*log(linear_val)
    return db_val

# print(from_dB(20))
# print(from_dBm(20))
# print(from_linear(20))
