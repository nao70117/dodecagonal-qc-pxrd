import sys
import numpy as np
import tqdm

PI = np.pi
SIN = np.sqrt(3)/2.0
CONST1 = 2/np.sqrt(6)
CONST2 = 1/np.sqrt(2)

def mattrix_dodeca_sym():
    """
    dodecagonal symmetry operations
    """
    #c12
    m1=np.array([[ 0, 1, 0, 0, 0],\
                [ 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 1, 0],\
                [-1, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1]])
    # mirror
    m2=np.array([[ 0, 0, 1, 0, 0],\
                [ 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0],\
                [ 0, 1, 0,-1, 0],\
                [ 0, 0, 0, 0, 1]])
    # mirror`
    m3=np.array([[0, 0, 0, 1, 0],\
                [ 0, 0, 1, 0, 0],\
                [ 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 1]])
    # inversion
    m4=np.array([[-1, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0],\
                [ 0, 0,-1, 0, 0],\
                [ 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0,-1]])
    return m1,m2,m3,m4