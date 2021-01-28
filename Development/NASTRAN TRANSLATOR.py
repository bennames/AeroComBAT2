from pyNastran.bdf.bdf import BDF

model = BDF()
model.is_nx = True
section = 5
#filename = r'D:\SNC IAS\01 - FAST Program\05 - Modified Sections\01 - AeroComBAT Files\section_{}.dat'.format(section)
filename = r'C:\Users\benna\Desktop\Work Temp\SNC\FAST\SIMPLE_SECTIONS\CTRIA6_1_100.dat'
model.read_bdf(filename, xref=True)

#Create Export File
f = open(filename[:-4]+'_AeroComBAT.dat','w')
f.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
f.write('$$$$$$$$$$$$$           AEROCOMBAT INPUT FILE           $$$$$$$$$$$$\n')
f.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')


for NID, node in model.nodes.items():
    node_pos = node.get_position()
    #Write node line
    f.write('XNODE,{},{},{}\n'.format(NID,node_pos[1],node_pos[2]))
    
# TEMP LINE FOR MAT PROPERTY
f.write('MAT_ISO,1,Lower Upper Aeroshell,8781151.,0.232915,0.0004144,0.0083\n')
f.write('MAT_ISO,2,Ring Frame Flange,7627582.,0.201668,0.0004144,0.0083\n')
f.write('MAT_ISO,3,Cabin Skin,8473671.,0.259765,0.0004144,0.0083\n')
f.write('MAT_ISO,4,Hat Stiffeners,9283126.,0.206558,0.0004144,0.0083\n')
f.write('MAT_ISO,5,Lower Outer Aeroshell,6544552.,0.428299,0.0004144,0.0083\n')
f.write('MAT_ISO,6,Upper Cabin,8196235.,0.284012,0.0004144,0.0083\n')
f.write('MAT_ISO,7,Titanium,16000000.,0.31,0.0004144,0.0083\n')
f.write('MAT_ISO,8,Quasi Iso,7944519.,0.306626,0.000144,0.0083\n')
f.write('MAT_ISO,9,Outer Aeroshell,7505270,0.344368,0.000144,0.0083\n')
f.write('MAT_ISO,10,Aluminum,10300000.,0.33,0.0002615,0.0083\n')

EIDs = []

for EID, elem in model.elements.items():
    if elem.pid==7000003:
        tmp_MID=1
    elif elem.pid == 7000004:
        tmp_MID=2
    elif elem.pid == 7000005:
        tmp_MID=3
    elif elem.pid == 7000006:
        tmp_MID=4
    elif elem.pid == 7000007:
        tmp_MID=5
    elif elem.pid == 7000008:
        tmp_MID=6
    elif elem.pid == 7000000:
        tmp_MID=7
    elif elem.pid == 7000001:
        tmp_MID=8
    elif elem.pid == 7000002:
        tmp_MID=9
    elif elem.pid == 7000009:
        tmp_MID=10
    else:
        raise ValueError('Encountered an unexpected Material Prop {}',elem.pid)
    
    EIDs += [EID]
    node_ids = elem.node_ids
    if elem.type=='CQUAD8':
        n1 = node_ids[0]
        n2 = node_ids[4]
        n3 = node_ids[1]
        n4 = node_ids[5]
        n5 = node_ids[2]
        n6 = node_ids[6]
        n7 = node_ids[3]
        n8 = node_ids[7]
        f.write('XQUAD8,{},{},{},{},{},{},{},{},{},{}\n'.format(EID,\
            n1,n2,n3,n4,n5,n6,n7,n8,tmp_MID))
    elif elem.type=='CQUAD4':
        n1 = node_ids[0]
        n2 = node_ids[1]
        n3 = node_ids[2]
        n4 = node_ids[3]
        f.write('XQUAD4,{},{},{},{},{},{}\n'.format(EID,\
            n1,n2,n3,n4,tmp_MID))
    elif elem.type=='CTRIA3':
        n1 = node_ids[0]
        n2 = node_ids[1]
        n3 = node_ids[2]
        f.write('XTRIA3,{},{},{},{},{}\n'.format(EID,\
            n1,n2,n3,tmp_MID))
    elif elem.type=='CTRIA6':
        n1 = node_ids[0]
        n2 = node_ids[1]
        n3 = node_ids[2]
        n4 = node_ids[3]
        n5 = node_ids[4]
        n6 = node_ids[5]
        f.write('XTRIA6,{},{},{},{},{},{},{},{}\n'.format(EID,\
            n1,n2,n3,n4,n5,n6,tmp_MID))
            
f.write('SECTIONG,{},{}\n'.format(section,section))
#EIDs = list(model.elements.keys())
f.write('LIST,{},INT,'.format(section)+str(EIDs)[1:-1]+'\n')
f.close()