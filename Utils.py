import numpy as np
from numba import jit
import numba as nb

def transformCompl(S,th,**kwargs):
    xsect = kwargs.pop('xsect',True)
    if xsect:
        # Note: This operation is done because during material property input,
        # the convention is that the fibers of a composite run in the
        # x direction, however for cross-sectional analysis, the nominal fiber
        # angle is parallel to the z-axis, and so before any in plane rotations
        # occur, the fiber must first be rotated about the y axis.
        Rysig, Ryeps = genCompRy(-90)
        S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))
    # Rotate material about x:
    Rxsig, Rxeps = genCompRx(th[0])
    S = np.dot(Rxeps,np.dot(S,np.linalg.inv(Rxsig)))
    # Rotate material about y:
    Rysig, Ryeps = genCompRy(th[1])
    S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))
    # Rotate material about z:
    Rzsig, Rzeps = genCompRz(th[2])
    S = np.dot(Rzeps,np.dot(S,np.linalg.inv(Rzsig)))
    return S

def genCompRx(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rxsig = np.array([[1.,0.,0.,0.,0.,0.],\
                      [0.,c**2,s**2,2*c*s,0.,0.],\
                      [0.,s**2,c**2,-2*c*s,0.,0.],\
                      [0.,-c*s,c*s,c**2-s**2,0.,0.],\
                      [0.,0.,0.,0.,c,-s],\
                      [0.,0.,0.,0.,s,c]])
    Rxeps = np.array([[1.,0.,0.,0.,0.,0.],\
                      [0.,c**2,s**2,c*s,0.,0.],\
                      [0.,s**2,c**2,-c*s,0.,0.],\
                      [0.,-2*c*s,2*c*s,c**2-s**2,0.,0.],\
                      [0.,0.,0.,0.,c,-s],\
                      [0.,0.,0.,0.,s,c]])
    return Rxsig, Rxeps
    
def genCompRy(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rysig = np.array([[c**2,0.,s**2,0.,2*c*s,0.],\
                      [0.,1.,0.,0.,0.,0.],\
                      [s**2,0.,c**2,0.,-2*c*s,0.],\
                      [0.,0.,0.,c,0.,-s],\
                      [-c*s,0.,c*s,0.,c**2-s**2,0.],\
                      [0.,0.,0.,s,0.,c]])
    Ryeps = np.array([[c**2,0.,s**2,0.,c*s,0.],\
                      [0.,1.,0.,0.,0.,0.],\
                      [s**2,0.,c**2,0.,-c*s,0.],\
                      [0.,0.,0.,c,0.,-s],\
                      [-2*c*s,0.,2*c*s,0.,c**2-s**2,0.],\
                      [0.,0.,0.,s,0.,c]])
    return Rysig, Ryeps
    
def genCompRz(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rzsig = np.array([[c**2,s**2,0.,0.,0.,2*c*s],\
                      [s**2,c**2,0.,0.,0.,-2*c*s],\
                      [0.,0.,1.,0.,0.,0.],\
                      [0.,0.,0.,c,s,0.],\
                      [0.,0.,0.,-s,c,0.],\
                      [-c*s,c*s,0.,0.,0.,c**2-s**2]])
    Rzeps = np.array([[c**2,s**2,0.,0.,0.,c*s],\
                      [s**2,c**2,0.,0.,0.,-c*s],\
                      [0.,0.,1.,0.,0.,0.],\
                      [0.,0.,0.,c,s,0.],\
                      [0.,0.,0.,-s,c,0.],\
                      [-2*c*s,2*c*s,0.,0.,0.,c**2-s**2]])
    return Rzsig, Rzeps
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