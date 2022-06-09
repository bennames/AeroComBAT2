import numpy as np
from numba import jit
import numba as nb


def stress_strain_rot(axis = 'x', angle_deg = 0.0):
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    # Active definition of rotations
    Rx = np.array([[1,  0,  0],
                   [0,  c, -s],
                   [0,  s,  c]])

    Ry = np.array([[ c,  0,  s],
                   [ 0,  1,  0],
                   [-s,  0,  c]])

    Rz = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]])


    rot_matrix = {'x': Rx, 'y': Ry, 'z': Rz}

    R = rot_matrix[axis]


    # Definition is here: https://en.wikipedia.org/wiki/Orthotropic_material#Condition_for_material_symmetry_2
    rot_sig = np.array([[R[0,0]**2, R[0,1]**2, R[0,2]**2, 2*R[0,1]*R[0,2], 2*R[0,0]*R[0,2], 2*R[0,0]*R[0,1]],
                        [R[1,0]**2, R[1,1]**2, R[1,2]**2, 2*R[1,1]*R[1,2], 2*R[1,0]*R[1,2], 2*R[1,0]*R[1,1]],
                        [R[2,0]**2, R[2,1]**2, R[2,2]**2, 2*R[2,1]*R[2,2], 2*R[2,0]*R[2,2], 2*R[2,0]*R[2,1]],
                        [R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2], R[1,1]*R[2,2] + R[1,2]*R[2,1],  R[1,0]*R[2,2] + R[1,2]*R[2,0],  R[1,0]*R[2,1] + R[1,1]*R[2,0]],
                        [R[0,0]*R[2,0], R[0,1]*R[2,1], R[0,2]*R[2,2], R[0,1]*R[2,2] + R[0,2]*R[2,1],  R[0,0]*R[2,2] + R[0,2]*R[2,0],  R[0,0]*R[2,1] + R[0,1]*R[2,0]],
                        [R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2], R[0,1]*R[1,2] + R[0,2]*R[1,1],  R[0,0]*R[1,2] + R[0,2]*R[1,0],  R[0,0]*R[1,1] + R[0,1]*R[1,0]],])

    rot_eps = np.array([[R[0,0]**2, R[0,1]**2, R[0,2]**2, R[0,1]*R[0,2], R[0,0]*R[0,2], R[0,0]*R[0,1]],
                        [R[1,0]**2, R[1,1]**2, R[1,2]**2, R[1,1]*R[1,2], R[1,0]*R[1,2], R[1,0]*R[1,1]],
                        [R[2,0]**2, R[2,1]**2, R[2,2]**2, R[2,1]*R[2,2], R[2,0]*R[2,2], R[2,0]*R[2,1]],
                        [2*R[1,0]*R[2,0], 2*R[1,1]*R[2,1], 2*R[1,2]*R[2,2], R[1,1]*R[2,2] + R[1,2]*R[2,1],  R[1,0]*R[2,2] + R[1,2]*R[2,0],  R[1,0]*R[2,1] + R[1,1]*R[2,0]],
                        [2*R[0,0]*R[2,0], 2*R[0,1]*R[2,1], 2*R[0,2]*R[2,2], R[0,1]*R[2,2] + R[0,2]*R[2,1],  R[0,0]*R[2,2] + R[0,2]*R[2,0],  R[0,0]*R[2,1] + R[0,1]*R[2,0]],
                        [2*R[0,0]*R[1,0], 2*R[0,1]*R[1,1], 2*R[0,2]*R[1,2], R[0,1]*R[1,2] + R[0,2]*R[1,1],  R[0,0]*R[1,2] + R[0,2]*R[1,0],  R[0,0]*R[1,1] + R[0,1]*R[1,0]],])

    return rot_sig, rot_eps


def transformCompl(S,th,**kwargs):
    xsect = kwargs.pop('xsect',True)
    if xsect:
        # Note: This operation is done because during material property input,
        # the convention is that the fibers of a composite run in the
        # x direction, however for cross-sectional analysis, the nominal fiber
        # angle is parallel to the z-axis, and so before any in plane rotations
        # occur, the fiber must first be rotated about the y axis.

        # Rotate material about y:
        Rysig, Ryeps = stress_strain_rot(axis = 'y', angle_deg = -90.0)
        Rysig_inv = np.linalg.inv(Rysig)
        S = np.dot(Ryeps,np.dot(S,Rysig_inv))

    # Rotate material about x:
    Rxsig, Rxeps = stress_strain_rot(axis = 'x', angle_deg = th[0])
    Rxsig_inv = np.linalg.inv(Rxsig)
    S = np.dot(Rxeps,np.dot(S,Rxsig_inv))

    # Rotate material about y:
    Rysig, Ryeps = stress_strain_rot(axis = 'y', angle_deg = th[1])
    Rysig_inv = np.linalg.inv(Rysig)
    S = np.dot(Ryeps,np.dot(S,Rysig_inv))

    # Rotate material about z:
    Rzsig, Rzeps = stress_strain_rot(axis = 'z', angle_deg = th[2])
    Rzsig_inv = np.linalg.inv(Rzsig)
    S = np.dot(Rzeps,np.dot(S,Rzsig_inv))


    return S



def add_scalar_tuple(tuple1,scalar):
    newTuples = ()
    for i in range(0,len(tuple1)):
        newTuples += (tuple1[i] + scalar,)
    return newTuples

@jit()
def add_scalar_nested_tuple(tuples,scalar):
    newTuples = ()
    for i in range(0,len(tuples)):
        tempTuples = ()
        for j in range(0,len(tuples[0])):
            tempTuples += (tuples[i][j] + scalar,)
        newTuples += (tempTuples,)
    return newTuples

def multiply_scalar_nested_tuple(tuples,scalar):
    newTuples = ()
    for i in range(0,len(tuples)):
        tempTuples = ()
        for j in range(0,len(tuples[0])):
            tempTuples += (tuples[i][j]*scalar,)
        newTuples += (tempTuples,)
    return newTuples

def add_2d_tuples(tuples1,tuples2):
    newTuples = ()
    for i in range(0,len(tuples1)):
        tempTuples = ()
        for j in range(0,len(tuples1[0])):
            tempTuples += (tuples1[i][j] + tuples2[i][j],)
        newTuples += (tempTuples,)
    return newTuples

@jit(nopython=True)
def flatten(np_array,rows,cols):
    data = [0.]
    for i in range(0,rows):
        for j in range(0,cols):
            data += [np_array[i,j]]
    return data[1:]

def decomposeRotation(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
#    x = np.degrees(np.arctan2(R[2,1],R[2,2]))
#    y = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2])))
#    z = np.degrees(np.arctan2(R[2,1], R[1,1]))
    return np.degrees(x),np.degrees(y),np.degrees(z)
