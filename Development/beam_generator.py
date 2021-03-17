from pyNastran.bdf.bdf import BDF
import numpy as np

model = BDF()
model.is_nx = True

### Read In the beam property csvs
filename = r"filepath"
data = np.genfromtxt(filename,delimiter=',',skip_header=True)

### Add Quasi Iso Mat

#Create an initial grid point
model.add_grid(1,data[0,0:3])

#Add Beams
for i in range(0,np.size(data,0)-1):
    x0 = data[i,0]
    y0 = data[i,1]
    z0 = data[i,2]
    EA0 = data[i,3]
    EI0_yy = data[i,4]
    EI0_zz = data[i,5]
    EI0_yz = data[i,6]
    GJ0 = data[i,7]
    GAKy0 = data[i,8]
    GAKz0 = data[i,9]
    A0 = data[i,10]
    
    E0 = EA0/A0
    nu0 = 0.3
    G0 = E0/(2*(1+nu0))
    
    xf = data[i+1,0]
    yf = data[i+1,1]
    zf = data[i+1,2]
    EAf = data[i+1,3]
    EIf_yy = data[i+1,4]
    EIf_zz = data[i+1,5]
    EIf_yz = data[i+1,6]
    GJf = data[i+1,7]
    GAKyf = data[i+1,8]
    GAKzf = data[i+1,9]
    Af = data[i+1,10]
    
    MID = i+1
    E_sect = (EA0/A0+EAf/Af)/2
    nu_sect = 0.3
    G_sect = E0/(2*(1+nu_sect))
    model.add_mat1(MID,E_sect,G_sect,nu_sect)
    
    so = ['NO','NO']
    xxb = [0.,1.]
    area = [A0,Af]
    i1 = [EI0_yy/E_sect,EIf_yy/E_sect]
    i2 = [EI0_zz/E_sect,EIf_zz/E_sect]
    i12 = [EI0_yz/E_sect,EIf_yz/E_sect]
    j = [GJ0/G_sect,GJf/G_sect]
    k1 = (GAKy0/(G_sect*A0)+GAKyf/(G_sect*A0))/2
    k2 = (GAKz0/(G_sect*A0)+GAKzf/(G_sect*A0))/2
    
    #Create a new grid point
    GID1 = i+1
    GID2 = i+2
    model.add_grid(GID2,data[i+1,0:3])
    
    #Create the beam property
    PID = i+1
    model.add_pbeam(PID,MID,xxb,so,area,i1,i2,i12,j,k1=k1,k2=k2)
    
    #Create the beam elements
    EID = i+1
    nids = [GID1,GID2]
    model.add_cbeam(EID,PID,nids,[x0,y0+1.,z0],None)
    
model.write_bdf(r"filepath")