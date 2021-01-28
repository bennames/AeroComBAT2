from Model import Model
import time
import numpy as np
from numba import jit, njit, prange
from numba import vectorize, float64


import cProfile
import pstats

model = Model()
model.loadDat(r'C:\Users\benna\Desktop\Work Temp\SNC\FAST\SIMPLE_SECTIONS\CQUAD4_AeroComBAT.dat')
model.loadDat(r'C:\Users\benna\Desktop\Work Temp\SNC\FAST\SIMPLE_SECTIONS\CTRIA3_AeroComBAT.dat')
model.loadDat(r'C:\Users\benna\Desktop\Work Temp\SNC\FAST\SIMPLE_SECTIONS\CQUAD8_AeroComBAT.dat')

xsect1 = model.sections.get(1)
xsect2 = model.sections.get(2)
xsect3 = model.sections.get(3)

elem1 = model.xelements.get(1)
elem2 = model.xelements.get(401)
elem3 = model.xelements.get(801)

etas = np.random.rand(10000)
xis = np.random.rand(10000)

@njit
def N_master(EID,eta,xi,xs,ys):
    # DN/Dxi
    dNdxi = np.zeros(8)
    dNdxi[0] = (-eta + 1)*(0.25*xi - 0.25) + 0.25*(-eta + 1)*(eta + xi + 1)
    dNdxi[1] = -1.0*xi*(-eta + 1)
    dNdxi[2] = -(-eta + 1)*(-0.25*xi - 0.25) - 0.25*(-eta + 1)*(eta - xi + 1)
    dNdxi[3] = -0.5*eta**2 + 0.5
    dNdxi[4] = -(eta + 1)*(-0.25*xi - 0.25) - 0.25*(eta + 1)*(-eta - xi + 1)
    dNdxi[5] = -1.0*xi*(eta + 1)
    dNdxi[6] = (eta + 1)*(0.25*xi - 0.25) + 0.25*(eta + 1)*(-eta + xi + 1)
    dNdxi[7] = 0.5*eta**2 - 0.5
    # DN/Deta
    dNdeta = np.zeros(8)
    dNdeta[0] = (-eta + 1)*(0.25*xi - 0.25) - (0.25*xi - 0.25)*(eta + xi + 1)
    dNdeta[1] = 0.5*xi**2 - 0.5
    dNdeta[2] = (-eta + 1)*(-0.25*xi - 0.25) - (-0.25*xi - 0.25)*(eta - xi + 1)
    dNdeta[3] = -2*eta*(0.5*xi + 0.5)
    dNdeta[4] = -(eta + 1)*(-0.25*xi - 0.25) + (-0.25*xi - 0.25)*(-eta - xi + 1)
    dNdeta[5] = -0.5*xi**2 + 0.5
    dNdeta[6] = -(eta + 1)*(0.25*xi - 0.25) + (0.25*xi - 0.25)*(-eta + xi + 1)
    dNdeta[7] = -2*eta*(-0.5*xi + 0.5)
    
    J11 = np.dot(dNdxi,xs)
    J12 = np.dot(dNdxi,ys)
    J21 = np.dot(dNdeta,xs)
    J22 = np.dot(dNdeta,ys)
    det = J11*J22-J12*J21
    # if det==0:
    #     print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n',EID)
    # Jinvmat = np.zeros((3,3))
    # Jinvmat[0,0] = J22/det
    # Jinvmat[0,1] = -J12/det
    # Jinvmat[1,0] = -J12/det
    # Jinvmat[1,1] = J11/det
    # Jinvmat[2,2] = 1/det
    Jinvmat11 = J22/det
    Jinvmat12 = -J12/det
    Jinvmat22 = J11/det
    Jinvmat33 = 1/det
    #Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
    det = np.abs(det)
    return det, Jinvmat11, Jinvmat12, Jinvmat22, Jinvmat33

# @vectorize([float64(float64, float64)])
# def dot(x,y):
#     return x*y

t0 = time.time()

for i in range(0,len(etas)):
    J1, Jinvmat1 = elem.Jdet_inv(etas[i],xis[i])
    
t1 = time.time()

print('Time taken: {}'.format(t1-t0))

t0

for i in range(0,len(etas)):
    J2, Jinvmat11, Jinvmat12, Jinvmat22, Jinvmat33 = N_master(elem.EID,etas[i],xis[i],elem.xs,elem.ys)
    
t1 = time.time()
print('Time taken: {}'.format(t1-t0))

print(J2-J1)

import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
for i in range(0,len(etas)):
    x2 = N_master(elem.EID,etas[i],xis[i],elem.xs,elem.ys)
pr.disable()
pr.print_stats()


pr = cProfile.Profile()
pr.enable()
for i in range(0,len(etas)):
    x1 = elem.x(etas[i],xis[i])
pr.disable()
pr.print_stats()


# xsect.printSummary()
# F = np.array([[0.],[0.],[0.],[0.],[0.],[10000000.]])
# xsect.calcWarpEffects(F)
# xsect.plotWarped(contour='Eps_13')

# # PLOT
# md = gl.MeshData(vertexes=xsect.vertices, faces=xsect.surfaces,\
#                  vertexColors=xsect.colors)
# mesh = gl.GLMeshItem(meshdata=md)
# openglWindow.addItem(mesh)

#def test():
#    model = Model()
#    
#    model.load_dat('transverse_shear_stiffness.dat')
#    
#    xsect = model.sections.getCrossSection(3)
#    xsect.xSectionAnalysis()
#    #xsect.printSummary()
#    
#    xsect.calcWarpEffects(np.array([1.,0.,0.,0.,0.,0.]))
#    xsect.plotWarped(contour='Hoff')
#    
#cProfile.run('test()','profile_results.txt', sort='tottime')
#p = pstats.Stats('profile_results.txt')
#p.sort_stats('tottime').print_stats(25)

# from pyamg.krylov import bicgstab
# from scipy.sparse.linalg import minres
# from pyamg.multilevel import multilevel_solver
# from pyamg.strength import classical_strength_of_connection
# from pyamg.classical.split import RS
# from pyamg.classical import direct_interpolation

# A = xsect.EquiA
# C = classical_strength_of_connection(A)
# splitting = RS(A)
# P = direct_interpolation(A, C, splitting)
# R = P.T

# #M = ml.aspreconditioner(cycle='V')
# #ml = ruge_stuben_solver(A)
# b1 = xsect.Equib1[:,0]
# levels = []
# levels.append(multilevel_solver.level())
# levels.append(multilevel_solver.level())
# levels[0].A = A
# levels[0].C = C
# levels[0].P = P
# levels[0].R = R
# levels[1].A = R*A*P 
# levels[0].splitting = splitting
# ml = multilevel_solver(levels, coarse_solver='splu')
# residuals = []
# M = ml.aspreconditioner()
# x1_1, info = minres(A, b1, tol=1e-10)
# x1_1, info = minres(A, b1, tol=1e-10,M=M)


# (x,flag) = bicgstab(A,b1, tol=1e-8)
# b2 = xsect.Equib1[:,1]
# b3 = xsect.Equib1[:,2]
# b4 = xsect.Equib1[:,3]
# b5 = xsect.Equib1[:,4]
# b6 = xsect.Equib1[:,5]
# t0 = time.time()
# x1_1, info = minres(A, b1, tol=1e-10)
# t1 = time.time()
# x1_2, info = minres(A, b1, tol=1e-10, M=M)
# t2 = time.time()
# x2, info = minres(A, b1, tol=1e-8, maxiter=30, M=ml)
# x3, info = minres(A, b1, tol=1e-8, maxiter=30, M=ml)
# x4, info = minres(A, b1, tol=1e-8, maxiter=30, M=ml)
# x5, info = minres(A, b1, tol=1e-8, maxiter=30, M=ml)
# x6, info = minres(A, b1, tol=1e-8, maxiter=30, M=ml)
#t0 = time.time()
#residuals1 = []
#residuals2 = []
#residuals3 = []
#residuals4 = []
#residuals5 = []
#residuals6 = []
#x1 = ml.solve(b1, tol=1e-10,residuals=residuals1,accel='gmres')
#x2 = ml.solve(b2, tol=1e-10,residuals=residuals2,accel='minres')
#x3 = ml.solve(b3, tol=1e-10,residuals=residuals3,accel='minres')
#x4 = ml.solve(b4, tol=1e-10,residuals=residuals4,accel='minres')
#x5 = ml.solve(b5, tol=1e-10,residuals=residuals5,accel='minres')
#x6 = ml.solve(b6, tol=1e-10,residuals=residuals6,accel='minres')
#t1 = time.time()
#dt1 = t1-t0
#
##from pylab import *
#plt1 = semilogy(residuals1[1:],label='x-shear')
#plt2 = semilogy(residuals2[1:],label='y-shear')
#plt3 = semilogy(residuals3[1:],label='axial')
#plt4 = semilogy(residuals4[1:],label='x-bending')
#plt5 = semilogy(residuals5[1:],label='y-bending')
#plt6 = semilogy(residuals6[1:],label='torsion')
#xlabel('iteration')
#ylabel('residual norm')
#title('Residual History')
#legend()
#show()


#model.nodes.printSummary()
#model.xnodes.printSummary()
#model.materials.printSummary()
#model.xelements.printSummary()
#model.laminates.printSummary()

#model.save_dat('test1.dat')

#xsect = model.sections.getCrossSection(4)
#xsect.xSectionAnalysis()

#model.sections.printSummary()

#model.belements.printSummary()

#test = xsect.EquiA.todense()
#from numpy.linalg import cond
#cond(test)