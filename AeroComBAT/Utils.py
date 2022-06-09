import numpy as np
from numba import jit
import numba as nb
import mechkit
from scipy.spatial.transform import Rotation as R
from sympy import symbols, Array, Matrix, solve
from sympy.utilities import lambdify

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


def stress_strain_rot(axis = 'x', angle_deg = 0.0):
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)


    # Active
    Rx = np.array([[1,  0, 0],
                   [0,  c, -s],
                   [0,  s, c]])

    Ry = np.array([[ c,  0,  s],
                   [ 0,  1,  0],
                   [-s,  0,  c]])

    Rz = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]])
    """

    # Passive
    Rx = np.array([[1,  0, 0],
                   [0,  c, s],
                   [0, -s, c]])

    Ry = np.array([[ c,  0, -s],
                   [ 0,  1,  0],
                   [ s,  0,  c]])

    Rz = np.array([[ c,  s, 0],
                   [-s,  c, 0],
                   [ 0,  0, 1]])
    """


    rot_matrix = {'x': Rx, 'y': Ry, 'z': Rz}

    R = rot_matrix[axis]

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


def transformCompl_2(S,th,**kwargs):
    xsect = kwargs.pop('xsect',True)

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



def rotT(T,g):
    p1 = np.tensordot(T,g,axes=((0),(0)))
    p2 = np.tensordot(p1,g,axes=((0),(0)))
    p3 = np.tensordot(p2,g,axes=((0),(0)))
    p4 = np.tensordot(p3,g,axes=((0),(0)))
    return p4


def transformCompl_mech_kit(S,th,**kwargs):
    xsect = kwargs.pop('xsect',True)
    # convert S to tensor
    expl_converter = mechkit.notation.ExplicitConverter()
    S_as_tensor = expl_converter.convert(inp=S, source="voigt", target="tensor", quantity="stiffness",)
    if xsect:
        # Note: This operation is done because during material property input,
        # the convention is that the fibers of a composite run in the
        # x direction, however for cross-sectional analysis, the nominal fiber
        # angle is parallel to the z-axis, and so before any in plane rotations
        # occur, the fiber must first be rotated about the y axis.
        theta = [0.0, -90, 0.0]
        rotation = R.from_euler('xyz', theta, degrees=True)
        G = rotation.as_matrix()
        S_as_tensor = rotT(S_as_tensor, G)
    theta = th
    rotation = R.from_euler('xyz', theta, degrees=True)
    G = rotation.as_matrix()
    S_as_tensor = rotT(S_as_tensor, G)
    # convert back to voigt
    S = expl_converter.convert(inp=S_as_tensor, source="tensor", target="voigt", quantity="stiffness",)
    return S


def transformCompl_sympy(S, th, **kwargs):

    xsect = kwargs.pop('xsect',True)
    if xsect:
        # Note: This operation is done because during material property input,
        # the convention is that the fibers of a composite run in the
        # x direction, however for cross-sectional analysis, the nominal fiber
        # angle is parallel to the z-axis, and so before any in plane rotations
        # occur, the fiber must first be rotated about the y axis.
        Rysig, Ryeps = genSimRy(-90)
        S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))
    # Rotate material about x:
    Rxsig, Rxeps = genSimRx(th[0])
    S = np.dot(Rxeps,np.dot(S,np.linalg.inv(Rxsig)))
    # Rotate material about y:
    Rysig, Ryeps = genSimRy(th[1])
    S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))
    # Rotate material about z:
    Rzsig, Rzeps = genSimRz(th[2])
    S = np.dot(Rzeps,np.dot(S,np.linalg.inv(Rzsig)))
    return S

def genSimRx(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rxsig = np.array([[1.,0.,0.,0.,0.,0.],\
                      [0.,c**2,s**2,-2*c*s,0.,0.],\
                      [0.,s**2,c**2,2*c*s,0.,0.],\
                      [0.,c*s,-c*s,c**2-s**2,0.,0.],\
                      [0.,0.,0.,0.,c,s],\
                      [0.,0.,0.,0.,-s,c]])

    Rxeps = np.array([[1.,0.,0.,0.,0.,0.],\
                      [0.,c**2,s**2,-c*s,0.,0.],\
                      [0.,s**2,c**2,c*s,0.,0.],\
                      [0.,2*c*s,-2*c*s,c**2-s**2,0.,0.],\
                      [0.,0.,0.,0.,c,s],\
                      [0.,0.,0.,0.,-s,c]])
    return Rxsig, Rxeps


def genSimRy(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rxsig = np.array([[c**2,0.,s**2,0.,2*c*s,0.],\
                      [0.,1.,0.,0.,0.,0.],\
                      [s**2,0.,c**2,0.,-2*c*s,0.],\
                      [0.,0.,0., c, 0., -s],\
                      [-c*s,0.,c*s,0.,c**2-s**2,0.],\
                      [0.,0.,0.,s,0.,c]])

    Rxeps = np.array([[c**2,0.,s**2,0.,c*s,0.],\
                      [0.,1.,0.,0.,0.,0.],\
                      [s**2,0.,c**2,0.,-c*s,0.],\
                      [0.,0.,0., c, 0., -s],\
                      [-2*c*s,0.,2*c*s,0.,c**2-s**2,0.],\
                      [0.,0.,0.,s,0.,c]])
    return Rxsig, Rxeps


def genSimRz(th):
    th = np.deg2rad(th)
    s = np.sin(th)
    c = np.cos(th)
    Rxsig = np.array([[c**2,s**2,0.,0.,0.,-2*c*s],\
                      [s**2,c**2,0.,0.,0.,2*c*s],\
                      [0.,0.,1.,0.,0.,0.],\
                      [0.,0.,0., c, s, 0.],\
                      [0.,0.,0.,-s,c,0.],\
                      [c*s,-c*s,0.,0.,0.,c**2-s**2]])


    Rxeps = np.array([[c**2,s**2,0.,0.,0.,-c*s],\
                      [s**2,c**2,0.,0.,0.,c*s],\
                      [0.,0.,1.,0.,0.,0.],\
                      [0.,0.,0.,c,s,0.],\
                      [0.,0.,0.,-s,c,0.],\
                      [2*c*s,-2*c*s,0.,0.,0.,c**2-s**2]])
    return Rxsig, Rxeps



def voigt_to_tensor(vmatrix, quantity):
    if vmatrix.shape[1] > 1:
        return 'TODO'
    else:
        v = np.array(vmatrix)[:,0]
    expl_converter = mechkit.notation.ExplicitConverter(dtype=v.dtype)
    mandel = v.copy()
    for position, factor in expl_converter.factors_mandel_to_voigt[quantity]:
        mandel[position] = v[position] * 1.0 / factor
    base = expl_converter.BASE6
    out = np.einsum("ajk, ...a->...jk", base, mandel, dtype=mandel.dtype, optimize=True)
    return Array(out)


def tensor_to_mandel(tensor, quantity):
    inp = np.array(tensor)
    expl_converter = mechkit.notation.ExplicitConverter(dtype=inp.dtype)
    base = expl_converter.BASE6
    if len(inp.shape) == 2:
        m6 = np.einsum("aij, ...ij ->...a", base, tensor, dtype=inp.dtype, optimize=True)
    else:
        m6 = np.einsum("aij, ...ijkl, bkl ->...ab", base, inp, base, dtype=inp.dtype, optimize=True)
    return Array(m6)

def tensor_to_voigt(tensor, quantity):
    """Convert from tensor notation in a sympy array. It convers first ot mandel6"""
    inp = np.array(tensor)
    expl_converter = mechkit.notation.ExplicitConverter(dtype=inp.dtype)
    m6 = np.array(tensor_to_mandel(inp, quantity))
    return Array(expl_converter._mandel6_to_voigt(m6, quantity=quantity))

def solve_T_prop(prop_rot, reference, original):
    equations = []
    for i, j in zip(prop_rot, reference):
        equations.append((i - j).as_poly(list(set(original.vec()[:]))))
    results = dict()
    for p in equations:
        for coe in p.coeffs():
            results.update(solve(coe, dict=True)[0])
    return results

c_alpha, c_beta, c_gamma = symbols('c_alpha, c_beta, c_gamma')
s_alpha, s_beta, s_gamma = symbols('s_alpha, s_beta, s_gamma')
g_x = Array([[1, 0, 0], [0, c_alpha, -s_alpha], [0, s_alpha, c_alpha]])
g_y = Array([[c_beta, 0, s_beta], [0, 1, 0], [-s_beta, 0, c_beta]])
g_z = Array([[c_gamma, -s_gamma, 0], [s_gamma, c_gamma, 0], [0, 0, 1]])
g = Array(g_z.tomatrix() * g_y.tomatrix() * g_x.tomatrix())

sigma = Matrix(3, 3, symbols('sigma_1:4(1:4)'))
# symmetry
sigma[1, 0] = sigma[0, 1]
sigma[2, 0] = sigma[0, 2]
sigma[2, 1] = sigma[1, 2]

eps = Matrix(3, 3, symbols('eps_1:4(1:4)'))
# symmetry
eps[1, 0] = eps[0, 1]
eps[2, 0] = eps[0, 2]
eps[2, 1] = eps[1, 2]

# sigma and epsilon rotated as matrix
sigma_prime = g.tomatrix()*sigma*g.transpose().tomatrix()
eps_prime = g.tomatrix()*eps*g.transpose().tomatrix()
# Symbolic transformation matrices for sigma and epsilon
T_sigma = Matrix(6, 6, symbols('T_sigma1:7(1:7)'))
T_eps = Matrix(6, 6, symbols('T_eps1:7(1:7)'))

# Convert original stress (sigma unrotated) to vector notation
# sigma as vector
prop = 'stress'
sigma_vec = Matrix(tensor_to_voigt(sigma, quantity=prop))
# Rotate stress as vector usign transformation matrix T_sigma
sigma_prime_t = (T_sigma * sigma_vec)
# Transforming sigma rotated with T_sigma as vector to matrix
# Then it is compared with rotation as matrix to create system of equations
check_sigma = Matrix(voigt_to_tensor(sigma_prime_t, quantity=prop))
# create symbolic rotation matrix
m_sigma = T_sigma.subs(solve_T_prop(check_sigma, sigma_prime, sigma))

# Convert original stress (sigma unrotated) to vector notation
# epsilon as vector
prop = 'strain'
eps_vec = Matrix(tensor_to_voigt(eps, quantity=prop))
# Rotate strain as vector usign transformation matrix T_epsilon
eps_prime_t = (T_eps * eps_vec)
# Transforming epsilon rotated with T_epsilon as vector to matrix
# Then it is compared with rotation as matrix to create system of equations
check_eps = Matrix(voigt_to_tensor(eps_prime_t, quantity=prop))
# create symbolic rotation matrix
m_eps = T_eps.subs(solve_T_prop(check_eps, eps_prime, eps))

rot_sigma = lambdify((s_alpha, c_alpha, s_beta, c_beta, s_gamma, c_gamma), m_sigma)
rot_eps = lambdify((s_alpha, c_alpha, s_beta, c_beta, s_gamma, c_gamma), m_eps)

def rotCMatrix(S, theta):
    sin_cos = []
    for _ in theta:
        sin_cos.append(np.sin(np.deg2rad(_)))
        sin_cos.append(np.cos(np.deg2rad(_)))
    R_eps = rot_eps(*sin_cos)
    R_sigma = rot_sigma(*sin_cos)
    return np.dot(R_eps, np.dot(S, np.linalg.inv(R_sigma)))


def transformCompliance(S, theta, **kwargs):
    xsect = kwargs.pop('xsect',True)
    if xsect:
        # Note: This operation is done because during material property input,
        # the convention is that the fibers of a composite run in the
        # x direction, however for cross-sectional analysis, the nominal fiber
        # angle is parallel to the z-axis, and so before any in plane rotations
        # occur, the fiber must first be rotated about the y axis.
        S_rot = rotCMatrix(S, [0,-90,0])
    else:
        S_rot = S
    # complete rotation
    return rotCMatrix(S_rot, theta)


def trans_other_rot(S, theta):

    """
    Rysig, Ryeps = genSimRy(-90)
    S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))
    """

    # Rotate material about y:
    Rysig, Ryeps = genSimRy(theta[1]-90)
    S = np.dot(Ryeps,np.dot(S,np.linalg.inv(Rysig)))

    # Rotate material about z:
    Rzsig, Rzeps = genSimRz(theta[2])
    S = np.dot(Rzeps,np.dot(S,np.linalg.inv(Rzsig)))

    return S


