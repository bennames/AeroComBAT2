from Structures import XSect,NodeLibrary, XNodeLibrary, MaterialLibrary, XElementLibrary, LaminateLibrary, Mesh, CrossSectionLibrary, BeamElementLibrary, BeamLibrary
from Utils import decomposeRotation
import pyqtgraph.opengl as gl
import sys
import numpy as np
import copy
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import minres
import os.path


class Model:
    def __init__(self):
        # Initialize Containers for various objects
        self.nodes = NodeLibrary()
        self.xnodes = XNodeLibrary()
        self.materials = MaterialLibrary()
        self.laminates = LaminateLibrary()
        self.sections = CrossSectionLibrary()
        self.belements = BeamElementLibrary()
        self.beams = BeamLibrary()
        self.xelements = XElementLibrary()
        self.lists = {}
        self.GUI = None
        self.showCSYS = False
        self.showNormals = False
        self.showRefAxis = True
        self.showShearCenter = False
        self.showMassCenter = False
        self.showTensionCenter = False
        self.currentXSectMesh = None
        self.activeSection = ''
        self.xsectFx = 0.
        self.xsectFy = 0.
        self.xsectFz = 0.
        self.xsectMx = 0.
        self.xsectMy = 0.
        self.xsectMz = 0.
        self.xref = ''
        self.yref = ''
        self.minContLim = ''
        self.maxContLim = ''
        self.warpFactor = 1.
        self.xsectColorScale = [0.,1]
        self.globalModels = []
        self.loads = {}
        self.sectionLoads = {}
        self.sectionLSIDs = {}
        self.constraints = {}
        self.tol = 1e-12

        # Finite Element Attributes:
        # Global Stiffness Matrix
        self.Kg = None
        # Reduced Global Stiffness Matrix
        self.Kgr = None
        # Global Force Vector
        self.Fg = None
        # Global Reduced Force Matricies
        self.Fgr = None
        # Global Mass Matrix
        self.Mg = None
        # Global Reduced Mass Matrix
        self.Mgr = None
        # Force Boundary Conditions
        self.Qg = None
        self.critVecID = {'Total Translation':1,\
                          'T1 Translation':2,\
                          'T2 Translation':3,\
                          'T3 Translation':4,\
                          'Von Mises Stress':106,\
                          'Maximum Principle Stress':107,\
                          'Minimum Principle Stress':108,\
                          'Sigma_xx':100,\
                          'Sigma_yy':101,\
                          'Sigma_zz':102,\
                          'Sigma_yz':103,\
                          'Sigma_xz':104,\
                          'Sigma_xy':105,\
                          'Sigma_11':109,\
                          'Sigma_22':110,\
                          'Sigma_33':111,\
                          'Sigma_23':112,\
                          'Sigma_13':113,\
                          'Sigma_12':114,\
                          'Maximum Principle Strain':207,\
                          'Minimum Principle Strain':208,\
                          'Max Abs Principle Strain':209,\
                          'Eps_11':201,\
                          'Eps_22':202,\
                          'Eps_33':203,\
                          'Eps_23':204,\
                          'Eps_13':205,\
                          'Eps_12':206,\
                          'Hoff':300}
        
    def loadDat(self,fileName):
        """A method to load a series of AeroComBAT cards form a .dat file.
        
        This method reads a series of cards into memory, then creates the
        objects corresponding to those object in the correct order.
        """
        supported_cards = ['NODE','XNODE','MAT_ISO','MAT_TISO','MAT_ORTHO',
                           'XQUAD4','XQUAD6','XQUAD8','XQUAD9','XTRIA3','XTRIA6','LIST',
                           'LAMINATE','SECTIONSB','SECTIONLM','SECTIONCT',
                           'SECTIONCC','SECTIONG']
        print('Importing {}...'.format(fileName))
        read_list = []
        with open(fileName) as file:
            for line in file:
                if not line[0]=='$':
                    read_list.append(line.split(','))
        copy_list = copy.copy(read_list)
        for i in range(0,len(supported_cards)):
            read_list = copy.copy(copy_list)
            for card in read_list:
                if card[0]==supported_cards[i]:
                    if supported_cards[i]=='NODE':
                        self.nodes.add(int(card[1]),float(card[2]),\
                            float(card[3]),float(card[4]))
                        copy_list.remove(card)
                    elif supported_cards[i]=='XNODE':
                        self.xnodes.add(int(card[1]),float(card[2]),\
                            float(card[3]))
                        copy_list.remove(card)
                    elif supported_cards[i]=='MAT_ISO':
                        if len(card)==6:
                            self.materials.add(int(card[1]),card[2],\
                                'ISO',[float(card[3]),float(card[4]),float(card[5]),1.,1.,1.])
                        elif len(card)==7:
                            self.materials.add(int(card[1]),card[2],\
                                'ISO',[float(card[3]),float(card[4]),float(card[5]),1.,1.,1.],\
                                mat_t=card[6])
                        elif len(card)==10:
                            self.materials.add(int(card[1]),card[2],\
                                'ISO',[float(card[3]),float(card[4]),float(card[5]),\
                                 float(card[7]),float(card[8]),float(card[9])],mat_t=card[6])
                        copy_list.remove(card)
                    elif supported_cards[i]=='MAT_TISO':
                        if len(card)==9:
                            self.materials.add(int(card[1]),card[2],\
                                'TISO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),1.,-1.,1.,-1.,1.])
                        elif len(card)==10:
                            self.materials.add(int(card[1]),card[2],\
                                'TISO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),1.,-1.,1.,-1.,1.],mat_t=float(card[9]))
                        elif len(card)==15:
                            self.materials.add(int(card[1]),card[2],\
                                'TISO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),float(card[10]),float(card[11]),
                                float(card[12]),float(card[13]),float(card[14])],mat_t=float(card[9]))
                        copy_list.remove(card)
                    elif supported_cards[i]=='MAT_ORTHO':
                        if len(card)==13:
                            self.materials.add(int(card[1]),card[2],\
                                'ORTHO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),float(card[9]),float(card[10]),\
                                float(card[11]),float(card[12]),1.,-1.,1.,-1.,1.,-1.,1.,1.,1.])
                        elif len(card)==14:
                            self.materials.add(int(card[1]),card[2],\
                                'ORTHO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),float(card[9]),float(card[10]),\
                                float(card[11]),float(card[12]),1.,-1.,1.,-1.,1.,-1.,1.,1.,1.],\
                                mat_t=float(card[13]))
                        elif len(card)==23:
                            self.materials.add(int(card[1]),card[2],\
                                'ORTHO',[float(card[3]),float(card[4]),\
                                float(card[5]),float(card[6]),float(card[7]),\
                                float(card[8]),float(card[9]),float(card[10]),\
                                float(card[11]),float(card[12]),float(card[14]),\
                                float(card[15]),float(card[16]),float(card[17]),\
                                float(card[18]),float(card[19]),float(card[20]),\
                                float(card[21]),float(card[22])],\
                                mat_t=float(card[13]))
                        copy_list.remove(card)
                    elif supported_cards[i]=='XQUAD4':
                        MID = int(card[6])
                        material = self.materials.get(MID)
                        if len(card)==7:
                            nids = [int(card[2]),int(card[3]),int(card[4]),int(card[5])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD4')
                        elif len(card)==10:
                            nids = [int(card[2]),int(card[3]),int(card[4]),int(card[5])]
                            th = [float(card[7]),float(card[8]),float(card[9])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD4',th=th)
                        copy_list.remove(card)
                    elif supported_cards[i]=='XQUAD6':
                        MID = int(card[8])
                        material = self.materials.get(MID)
                        if len(card)==9:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD6')
                        elif len(card)==12:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7])]
                            th = [float(card[9]),float(card[10]),float(card[11])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD6',th=th)
                        copy_list.remove(card)
                    elif supported_cards[i]=='XQUAD8':
                        MID = int(card[10])
                        material = self.materials.get(MID)
                        if len(card)==11:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7]),\
                                    int(card[8]),int(card[9])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD8')
                        elif len(card)==14:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7]),\
                                    int(card[8]),int(card[9])]
                            th = [float(card[11]),float(card[12]),float(card[13])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD8',th=th)
                        copy_list.remove(card)
                    elif supported_cards[i]=='XQUAD9':
                        MID = int(card[11])
                        material = self.materials.get(MID)
                        if len(card)==12:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7]),\
                                    int(card[8]),int(card[9]),int(card[10])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD9')
                        elif len(card)==15:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7]),\
                                    int(card[8]),int(card[9]),int(card[10])]
                            th = [float(card[12]),float(card[13]),float(card[14])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XQUAD9',th=th)
                        copy_list.remove(card)
                    
                    elif supported_cards[i]=='XTRIA3':
                        MID = int(card[5])
                        material = self.materials.get(MID)
                        if len(card)==6:
                            nids = [int(card[2]),int(card[3]),int(card[4])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XTRIA3')
                        elif len(card)==9:
                            nids = [int(card[2]),int(card[3]),int(card[4])]
                            th = [float(card[6]),float(card[7]),float(card[8])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XTRIA3',th=th)
                        copy_list.remove(card)
                        
                    elif supported_cards[i]=='XTRIA6':
                        MID = int(card[8])
                        material = self.materials.get(MID)
                        if len(card)==9:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XTRIA6')
                        elif len(card)==12:
                            nids = [int(card[2]),int(card[3]),int(card[4]),\
                                    int(card[5]),int(card[6]),int(card[7])]
                            th = [float(card[9]),float(card[10]),float(card[11])]
                            nodes = []
                            for nid in nids:
                                nodes += [self.xnodes.get(nid)]
                            self.xelements.add(int(card[1]),nodes,material,'XTRIA6',th=th)
                        copy_list.remove(card)
                    
                    elif supported_cards[i]=='LIST':
                        LSID = int(card[1])
                        if LSID in self.lists.keys():
                            print('WARNING: Overwritting list {}'.format(LSID))
                        list_array = []
                        for j in range(0,len(card)-3):
                            if card[2]=='INT':
                                list_array += [int(card[j+3])]
                            elif card[2]=='FLOAT':
                                list_array += [float(card[j+3])]
                        self.lists[LSID] = list_array
                        copy_list.remove(card)
                    elif supported_cards[i]=='LAMINATE':
                        LAMID = int(card[1])
                        NiLSID=int(card[2])
                        MiLSID=int(card[3])
                        THiLSID = int(card[4])
                        if 'TRUE' in card[5]:
                            sym=True
                        else:
                            sym=False
                        proceed=True
                        if not NiLSID in self.lists.keys():
                            print('The list of the number of plies does not exist. Laminate not created.\n')
                            proceed=False
                        if not MiLSID in self.lists.keys():
                            print('The list of the used materials does not exist. Laminate not created.\n')
                            proceed=False
                        if not THiLSID in self.lists.keys():
                            print('The list of the ply orientations does not exist. Laminate not created.\n')
                            proceed=False
                        
                        for MID in self.lists[MiLSID]:
                            if self.materials.get(MID).t==0.:
                                print("You cannot make a laminate using a material ({})"
                                      " whose material thickness=0. Laminate not created.\n".format(MID))
                                proceed=False
                        if proceed:
                            n_i = self.lists[NiLSID]
                            m_i = self.lists[MiLSID]
                            th_i = self.lists[THiLSID]
                            self.laminates.add(LAMID,n_i,m_i,self.materials,th=th_i,sym=sym)
                            laminate = self.laminates.get(LAMID)
                            laminate.NiLSID = NiLSID
                            laminate.MiLSID = MiLSID
                            laminate.THiLSID = THiLSID
                        copy_list.remove(card)
                    elif supported_cards[i]=='SECTIONSB':
                        XID = int(card[1])
                        if XID in self.sections.xsectDict.keys():
                            print('WARNING: You are overwritting a cross-section'
                                  ' object with the same XID.')
                        MID = int(card[2])
                        L1 = float(card[3])
                        L2 = float(card[4])
                        elemType = card[5]
                        elemX = int(card[6])
                        elemY = int(card[7])
                        typeXSect = 'solidBox'
                        self.createXSect(XID,typeXSect=typeXSect,L1=L1, L2=L2,\
                                         elemX=elemX,elemY=elemY,MID=MID,\
                                         matLib=self.materials,elemType=elemType)
                        copy_list.remove(card)
                    elif supported_cards[i]=='SECTIONLM':
                        XID = int(card[1])
                        if XID in self.sections.xsectDict.keys():
                            print('WARNING: You are overwritting a cross-section'
                                  ' object with the same XID.')
                        LAMID = int(card[2])
                        laminate = self.laminates.get(LAMID)
                        L1 = float(card[3])
                        elemType = card[4]
                        elemAR = float(card[5])
                        typeXSect = 'laminate'
                        self.createXSect(XID,typeXSect=typeXSect,L1=L1,\
                                         elemAR=elemAR,laminate=laminate,\
                                         elemType=elemType,matLib=self.materials)
                        copy_list.remove(card)
                    elif supported_cards[i]=='SECTIONCT':
                        XID = int(card[1])
                        if XID in self.sections.xsectDict.keys():
                            print('WARNING: You are overwritting a cross-section'
                                  ' object with the same XID.')
                        LAM_LSID = int(card[2])
                        LAMIDs = self.lists[LAM_LSID]
                        laminates = []
                        for LAMID in LAMIDs:
                            laminates += [self.laminates.get(LAMID)]
                        R = float(card[3])
                        elemType = card[4]
                        elemAR = float(card[5])
                        typeXSect = 'compositeTube'
                        self.createXSect(XID,typeXSect=typeXSect,R=R,\
                                         elemAR=elemAR,laminates=laminates,elemType=elemType,
                                         matLib=self.materials)
                        copy_list.remove(card)
                    elif supported_cards[i]=='SECTIONCC':
                        XID = int(card[1])
                        if XID in self.sections.xsectDict.keys():
                            print('WARNING: You are overwritting a cross-section'
                                  ' object with the same XID.')
                        LAMID = int(card[2])
                        laminate = self.laminates.get(LAMID)
                        L1 = float(card[3])
                        L2 = float(card[4])
                        elemType = card[5]
                        elemAR = float(card[6])
                        typeXSect = 'cchannel'
                        self.createXSect(XID,typeXSect=typeXSect,L1=L1,L2=L2,\
                                         elemAR=elemAR,laminate=laminate,\
                                         elemType=elemType,matLib=self.materials)
                        copy_list.remove(card)
                    elif supported_cards[i]=='SECTIONG':
                        XID = int(card[1])
                        if XID in self.sections.xsectDict.keys():
                            print('WARNING: You are overwritting a cross-section'
                                  ' object with the same XID.')
                        elem_LSID = int(card[2])
                        elem_list = self.lists[elem_LSID]
                        #if XID==1: print(elem_list)
                        elemDict = {}
                        nodeDict = {}
                        for xEID in elem_list:
                            xelem = self.xelements.get(xEID)
                            elemDict[xEID] = xelem
                            xnids = xelem.NIDs
                            for xnid in xnids:
                                if not xnid in nodeDict.keys():
                                    nodeDict[xnid] = self.xnodes.get(xnid)
                        genMesh = Mesh('general',elemDict=elemDict,nodeDict=nodeDict)
                        self.createXSect(XID,mesh=genMesh)
                        copy_list.remove(card)
                        print('Created cross-section {}!\n'.format(XID))
        print('Finished reading {}!'.format(fileName))
    
    def translateNastranDat(self,filenames,xdir,ydir,XID=""):
        print('Translating NASTRAN to AeroComBAT input file...')
        try:
            from pyNastran.bdf.bdf import BDF
            for fileName in filenames:
                model = BDF()
                model.is_nx = True
                model.read_bdf(fileName, xref=True)
                f = open(fileName[:-4]+'_AeroComBAT.dat','w')
                f.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
                f.write('$$$$$$$$$$$$$           AEROCOMBAT INPUT FILE           $$$$$$$$$$$$\n')
                f.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
                # Write Node Objects
                if xdir<0:
                    xfactor=-1
                    xdir=abs(xdir)-1
                else:
                    xfactor=1
                    xdir -= 1
                if ydir<0:
                    yfactor=-1
                    ydir=abs(ydir)-1
                else:
                    yfactor=1
                    ydir -= 1
                for NID, node in model.nodes.items():
                    node_pos = node.get_position()
                    f.write('XNODE,{},{},{}\n'.format(NID,node_pos[xdir]*xfactor,node_pos[ydir]*yfactor))
                # Write Material Objects
                for MID, mat in model.materials.items():
                    if mat.type=='MAT1':
                        if MID in self.materials.getIDs():
                            aeroMID = max(self.materials.getIDs())+1
                        else:
                            aeroMID = MID
                        name = mat.comment[mat.comment.find('Material')+len('Material'):].replace('\n','')
                        aeroE = mat.E()
                        aeronu = mat.nu
                        aerorho = mat.rho
                        f.write('MAT_ISO,{},{},{},{},{}\n'.format(aeroMID,name,aeroE,aeronu,aerorho))
                    else:
                        print('Currently only MAT1 cards are supported. Skipping material {}'.format(MID))
                # Write Element Objects
                EIDs = []
    
                for EID, elem in model.elements.items():
                    EIDs += [EID]
                    node_ids = elem.node_ids
                    mid = elem.Mid()
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
                            n1,n2,n3,n4,n5,n6,n7,n8,mid))
                    elif elem.type=='CQUAD4':
                        n1 = node_ids[0]
                        n2 = node_ids[1]
                        n3 = node_ids[2]
                        n4 = node_ids[3]
                        f.write('XQUAD4,{},{},{},{},{},{}\n'.format(EID,\
                            n1,n2,n3,n4,mid))
                    elif elem.type=='CTRIA3':
                        n1 = node_ids[0]
                        n2 = node_ids[1]
                        n3 = node_ids[2]
                        f.write('XTRIA3,{},{},{},{},{}\n'.format(EID,\
                            n1,n2,n3,mid))
                    elif elem.type=='CTRIA6':
                        n1 = node_ids[0]
                        n2 = node_ids[1]
                        n3 = node_ids[2]
                        n4 = node_ids[3]
                        n5 = node_ids[4]
                        n6 = node_ids[5]
                        f.write('XTRIA6,{},{},{},{},{},{},{},{}\n'.format(EID,\
                            n1,n2,n3,n4,n5,n6,mid))
                if not XID=="":
                    section=XID
                elif len(self.sections.getIDs())==0:
                    section = 1
                else:
                    section = max(self.sections.getIDs())+1
                f.write('SECTIONG,{},{}\n'.format(section,section))
                f.write('LIST,{},INT,'.format(section)+str(EIDs)[1:-1]+'\n')
                f.close()
                print('Finished translating NASTRAN to AeroComBAT input file!')
                self.loadDat(fileName[:-4]+'_AeroComBAT.dat')
        except ModuleNotFoundError:
            print('To import a nastran file, the module pyNastran must'\
                  ' first be installed.')
            
                        
    def save_dat(self,fileName):
        """A method to save a series of AeroComBAT cards to a .dat file.
        
        This method writes all AeroComBAT objects to a .csv file.
        """
        file = open(fileName,'w')
        sLSID_xsect = max(self.lists.keys())+1
        cards = []
        # Node Library
        cards += self.nodes.writeToFile()
        # XNode Library
        cards += self.xnodes.writeToFile()
        # Material Library
        cards += self.materials.writeToFile()
        # Laminate Library
        cards += self.laminates.writeToFile()
        # XElement Library
        cards += self.xelements.writeToFile()
        # Cross-Section Library
        cards += self.sections.writeToFile(sLSID_xsect)
        for card in cards:
            file.write(card+'\n')
        file.close()
        
                    
    def exportSectionsNeutral(self,XIDs,filename,path=''):
        if os.path.isdir(path):
            export_path=path
        else:
            export_path=''
        for XID in XIDs:
            intro = '   -1\n   100\n<NULL>\n19.1,\n   -1\n'
            section = self.sections.get(XID)
            MIDs = []
            node_block = '   -1\n   403\n'
            for NID, node in section.nodeDict.items():
                node_block += node.writeToNeutral()
            node_block += '   -1\n'
            element_block = '   -1\n   404\n'
            for EID, elem in section.elemDict.items():
                if elem.MID not in MIDs:
                    MIDs += [elem.MID]
                element_block += elem.writeToNeutral()
            element_block += '   -1\n'
            mat_block = '   -1\n   601\n'
            prop_block = '   -1\n   402\n'
            for MID in MIDs:
                mat = self.materials.get(MID)
                tmpMatStr, tmpPropStr = mat.writeToNeutral()
                mat_block += tmpMatStr
                prop_block += tmpPropStr
            mat_block += '   -1\n'
            prop_block += '   -1\n'
            f = open(export_path+r'/'+filename+'_XID_{}.neu'.format(XID), 'w')
            f.write(intro)
            f.write(mat_block)
            f.write(prop_block)
            f.write(node_block)
            f.write(element_block)
            f.close()
    def exportSectionsContour(self,XIDs,filename,criteria,path=''):
        if os.path.isdir(path):
            export_path=path
        else:
            export_path=''
        intro = '   -1\n   100\n<NULL>\n19.1,\n   -1\n'
        for XID in XIDs:
            print('Exporting results for crosss-section {}...'.format(XID))
            section = self.sections.get(XID)
            LCIDs = []
            if not section.analyzed:
                print('Cross-section {} must be analyzed before loads can be applied.'.format(XID))
            else:
                # Load the section with all of the loads
                if self.GUI.load_source_drop.currentText()=='Custom':
                    LCIDs = [0]
                elif self.GUI.load_source_drop.currentText()=='Load ID':
                    LCID = int(self.GUI.load_source_vals.currentText())
                    LCIDs = [LCID]
                elif self.GUI.load_source_drop.currentText()=='All':
                    tmpSectionLoads = self.sectionLoads[XID]
                    LCIDs = list(tmpSectionLoads.keys())
                else:
                    #Get the intersection of LCIDs in the set and added to the section
                    LSID = int(self.GUI.load_source_vals.currentText())
                    setLCIDs = self.sectionLSIDs[LSID]
                    tmpSectionLoads = self.sectionLoads[XID]
                    sectionLCIDs = list(tmpSectionLoads.keys())
                    LCIDs = [value for value in setLCIDs if value in sectionLCIDs]
                    ommittedLCIDs = [value for value in setLCIDs if value not in LCIDs]
                    print('User tried to export loads from set {} on section {}.'\
                          ' This set includes LCIDs {} which have not been added to'\
                          ' section {}'.format(LSID,XID,ommittedLCIDs,XID))
            outputSetStrs = '   -1\n   450\n'
            outputVecStrs = '   -1\n  1051\n'
            for LCID in LCIDs:
                if LCID==0:
                    outLCID=1001
                else:
                    outLCID=LCID
                print('   -LCID: {}'.format(outLCID))
                outputSetStrs += '{},\n'.format(outLCID)
                outputSetStrs += 'AeroComBAT Case {},\n'.format(outLCID)
                outputSetStrs += '0,1,0,0,\n'
                outputSetStrs += '0.\n'
                outputSetStrs += '1,\n'
                outputSetStrs += 'AeroComBAT section loads analysis for cross-section {}\n'.format(XID)
                outputSetStrs += '1,3,1,\n'
                outputSetStrs += '-1,-1,0.,\n'
                outputSetStrs += '0,0,\n'
                outputSetStrs += '0,\n'
                outputSetStrs += '0,0,0,\n'
                outputSetStrs += '0,0,0,0,0,0,\n'
                outputSetStrs += '0,0,0,0,0,0,\n'
                for crit in criteria:
                    print('      -Criteria: {}'.format(crit))
                    vecID = self.critVecID[crit]
                    outputVecStrs += '{},{},1,\n'.format(outLCID,vecID)
                    outputVecStrs += '{}\n'.format(crit)
                    outputVecStrs += '9.9900002E+30,-9.9900002E+30,9.9900002E+30,\n'
                    outputVecStrs += '0,0,0,0,0,0,0,0,0,0,\n'
                    outputVecStrs += '0,0,0,0,0,0,0,0,0,0,\n'
                    outputVecStrs += '0,\n'
                    outputVecStrs += '0,0,0,8,0,\n'
                    outputVecStrs += '1,0,1,0,\n'
                    for EID, elem in section.elemDict.items():
                        data = elem.getContour([outLCID],crit=crit,centroid=True)
                        outputVecStrs += '{},{},\n'.format(EID,data[0])
                    outputVecStrs += '-1,0.,\n'
            outputSetStrs += '   -1\n'
            outputVecStrs += '   -1\n'
            
            f = open(export_path+r'/'+filename+'_XID_{}.neu'.format(XID), 'w')
            f.write(intro)
            f.write(outputSetStrs)
            f.write(outputVecStrs)
            f.close()
            print('Completed exporting results for crosss-section {}!'.format(XID))
        
    def createXSect(self,XID,**kwargs):
        if len(self.xnodes.xnodeDict.keys())==0:
            sxnid=1
        else:
            sxnid=max(self.xnodes.xnodeDict.keys())+1
        if len(self.xelements.xelemDict.keys())==0:
            sxeid=1
        else:
            sxeid=max(self.xelements.xelemDict.keys())+1
        self.sections.add(XID,sxnid=sxnid,sxeid=sxeid,**kwargs)
        section = self.sections.get(XID)
        self.xnodes.xnodeDict.update(section.nodeDict)
        self.xelements.xelemDict.update(section.elemDict)
        if self.GUI:
            self.GUI.updateXSectDrop()
    def plotRigidXSect(self):
        if self.GUI.xsectDropDown.currentText()=='' and self.activeSection == '':
            print('Please select a cross-section to view.')
        elif self.GUI.xsectDropDown.currentText()=='' and self.currentXSectMesh:
            self.GUI.XSectView.removeItem(self.currentXSectMesh)
            self.currentXSectMesh = ''
            self.activeSection = ''
        else:
            currentXID = int(self.GUI.xsectDropDown.currentText())
            self.GUI.sectionLabel.setText('Cross-section: {} '.format(currentXID))
            if self.GUI.load_source_drop.currentText()=='Load ID':
                self.GUI.updateLSIDsDrop()
            try:
                tmpXSect = self.sections.get(currentXID)
            except Exception as e: print(str(e))
            md = gl.MeshData(vertexes=tmpXSect.vertices, faces=tmpXSect.surfaces,\
                             vertexColors=tmpXSect.colors)
            mesh = gl.GLMeshItem(meshdata=md,drawEdges=True,\
                                 edgeColor=(0, 0, 0, 1))
            if self.showCSYS:
                CSYSs = tmpXSect.CSYSs
            else:
                CSYSs = []
            
            if self.showNormals:
                normals = tmpXSect.normals
            else:
                normals = []
            
            xsect_points = []
            sphere = gl.MeshData.sphere(rows=10, cols=10)
            if self.showRefAxis:
                refAx = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', glOptions='opaque', color=(1, 0, 0, 1))
                refAx.translate(tmpXSect.refAxis[0], tmpXSect.refAxis[1], 0)
                refAx.scale(tmpXSect.scale*1e-2, tmpXSect.scale*1e-2, tmpXSect.scale*1e-2)
                xsect_points += [refAx]
            if self.showShearCenter and tmpXSect.analyzed:
                shearCntr = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', glOptions='opaque', color=(0, 1, 0, 1))
                shearCntr.translate(tmpXSect.xs, tmpXSect.ys, 0)
                shearCntr.scale(tmpXSect.scale*1e-2, tmpXSect.scale*1e-2, tmpXSect.scale*1e-2)
                xsect_points += [shearCntr]
            if self.showTensionCenter and tmpXSect.analyzed:
                tensionCntr = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', glOptions='opaque', color=(0, 0, 1, 1))
                tensionCntr.translate(tmpXSect.xt, tmpXSect.yt, 0)
                tensionCntr.scale(tmpXSect.scale*1e-2, tmpXSect.scale*1e-2, tmpXSect.scale*1e-2)
                xsect_points += [tensionCntr]
            if self.showMassCenter and tmpXSect.analyzed:
                massCntr = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', glOptions='opaque', color=(1, 1, 0, 1))
                massCntr.translate(tmpXSect.x_m[0], tmpXSect.x_m[1], 0)
                massCntr.scale(tmpXSect.scale*1e-2, tmpXSect.scale*1e-2, tmpXSect.scale*1e-2)
                xsect_points += [massCntr]
            if (self.showShearCenter or self.showTensionCenter or self.showMassCenter) and not tmpXSect.analyzed:
                print('In order to view the shear, tension, or mass center of'
                      ' the cross-section, the section must first be analyzed.')
            try:
                if self.currentXSectMesh:
                    self.GUI.XSectView.removeItem(self.currentXSectMesh)
                    oldCSYSs = self.currentCSYSs
                    oldXSectPoints = self.currentXSectPoints
                    oldNormals = self.currentNormals
                    for CSYS in oldCSYSs:
                        self.GUI.XSectView.removeItem(CSYS)
                    for point in oldXSectPoints:
                        self.GUI.XSectView.removeItem(point)
                    for normal in oldNormals:
                        self.GUI.XSectView.removeItem(normal)
                    
                self.currentXSectMesh = mesh
                self.currentCSYSs = CSYSs
                self.GUI.XSectView.addItem(mesh)
                self.activeSection = tmpXSect
                self.currentXSectPoints = xsect_points
                self.currentNormals = normals
                
                for CSYS in CSYSs:
                    self.GUI.XSectView.addItem(CSYS)
                for point in xsect_points:
                    self.GUI.XSectView.addItem(point)
                for normal in normals:
                    self.GUI.XSectView.addItem(normal)
            except Exception as e: print(str(e))
    def analyzeActiveSection(self):
        if self.activeSection:
            ref_ax_str = self.GUI.ref_ax_Box.currentText()
            if ref_ax_str == 'Custom Axis':
                if self.xref=='' and self.yref=='':
                    ref_ax = [0,0]
                elif self.xref=='':
                    ref_ax = [0,self.yref]
                elif self.yref=='':
                    ref_ax = [self.xref,0]
                else:
                    ref_ax = [self.xref,self.yref]
            elif ref_ax_str == 'Tension Center':
                ref_ax = 'tensionCntr'
            elif ref_ax_str == 'Mass Center':
                ref_ax = 'massCntr'
            elif ref_ax_str == 'Shear Center':
                ref_ax = 'shearCntr'
            self.activeSection.xSectionAnalysis(tol=self.tol,ref_ax=ref_ax)
        else:
            print('No active cross-section.')
    def analyzeSections(self,XIDs):
        ref_ax_str = self.GUI.ref_ax_Box.currentText()
        if ref_ax_str == 'Custom Axis':
            if self.xref=='' and self.yref=='':
                ref_ax = [0,0]
            elif self.xref=='':
                ref_ax = [0,self.yref]
            elif self.yref=='':
                ref_ax = [self.xref,0]
            else:
                ref_ax = [self.xref,self.yref]
        elif ref_ax_str == 'Tension Center':
            ref_ax = 'tensionCntr'
        elif ref_ax_str == 'Mass Center':
            ref_ax = 'massCntr'
        elif ref_ax_str == 'Shear Center':
            ref_ax = 'shearCntr'
        for XID in XIDs:
            section = self.sections.get(XID)
            print('Starting analysis on section: {}...'.format(XID))
            section.xSectionAnalysis(tol=self.tol,ref_ax=ref_ax)
            print('Completed analysis on section: {}'.format(XID))
    def exportSections(self,XIDs,filename,path=''):
        print('Exporting cross-sections stiffnesses...')
        if os.path.isdir(path):
            export_path=path
        else:
            export_path=''
        f = open(export_path+r'/'+filename+'.csv', 'a')
        for XID in XIDs:
            f.write('Section (tol={}) {}\n'.format(self.tol,XID))
            section = self.sections.get(XID)
            if section.analyzed:
                K_section = section.K
                np.savetxt(f,K_section,delimiter=',')
            else:
                print('Section {} has not been analyzed and will not be exported. \n'.format(XID))
        f.close()
        print('Completed exporting cross-section:')
    def updateSectionRefAx(self):
        print('Updating the cross-section reference axis...')
        if self.activeSection:
            tmpXSect = self.activeSection
            if not tmpXSect.analyzed:
                print('This cross-section must be analyzed before the reference'
                      ' axis can be moved.')
            else:
                ref_ax_str = self.GUI.ref_ax_Box.currentText()
                if ref_ax_str == 'Custom Axis':
                    if self.xref=='':
                        ref_ax = [0,self.yref]
                    elif self.yref=='':
                        ref_ax = [self.xref,0]
                    elif self.xref=='' and self.yref=='':
                        ref_ax = [0,0]
                    else:
                        ref_ax = [self.xref,self.yref]
                elif ref_ax_str == 'Tension Center':
                    ref_ax = 'tensionCntr'
                elif ref_ax_str == 'Mass Center':
                    ref_ax = 'massCntr'
                elif ref_ax_str == 'Shear Center':
                    ref_ax = 'shearCntr'
                if not (ref_ax_str == 'Custom Axis' and (self.xref=='' or self.yref=='')):
                    tmpXSect.setReferenceAxis(ref_ax)
                print('Finished updating the cross-section reference axis!')
        else:
            print('No active cross-section.')
    def setFx(self,sb):
        if sb:
            try:
                self.xsectFx = float(sb)
            except:
                self.xsectFx = 0.
        else:
            self.xsectFx = 0.
    def setFy(self,sb):
        if sb:
            try:
                self.xsectFy = float(sb)
            except:
                self.xsectFy = 0.
        else:
            self.xsectFy = 0.
    def setFz(self,sb):
        if sb:
            try:
                self.xsectFz = float(sb)
            except:
                self.xsectFz = 0.
        else:
            self.xsectFz = 0.
    def setMx(self,sb):
        if sb:
            try:
                self.xsectMx = float(sb)
            except:
                self.xsectMx = 0.
        else:
            self.xsectMx = 0.
    def setMy(self,sb):
        if sb:
            try:
                self.xsectMy = float(sb)
            except:
                self.xsectMy = 0.
        else:
            self.xsectMy = 0.
    def setMz(self,sb):
        if sb:
            try:
                self.xsectMz = float(sb)
            except:
                self.xsectMz = 0.
        else:
            self.xsectMz = 0.
    def setXref(self,sb):
        if sb:
            try:
                self.xref = float(sb)
            except:
                self.xref = 0.
        else:
            self.xref = ''
    def setYref(self,sb):
        if sb:
            try:
                self.yref = float(sb)
            except:
                self.yref = 0.
        else:
            self.yref = ''
    def setWarpFactor(self,sb):
        if sb:
            try:
                self.warpFactor = float(sb)
            except:
                self.warpFactor = 0.
        else:
            self.warpFactor = 0.
    def setMinContLim(self,sb):
        if sb:
            try:
                self.minContLim = float(sb)
            except:
                self.minContLim = ''
        else:
            self.minContLim = ''
    def setMaxContLim(self,sb):
        if sb:
            try:
                self.maxContLim = float(sb)
            except:
                self.maxContLim = ''
        else:
            self.maxContLim = ''
    def setTol(self,sb):
        if sb:
            try:
                self.tol = float(sb)
            except:
                self.tol = 1e-12
        else:
            self.tol = 1e-12
    def importSectionLoads(self,filenames,xdir,ydir,LF=1):
        for filename in filenames:
            if len(self.sectionLSIDs.keys())==0:
                LSID = 1
            else:
                LSID = max(self.sectionLSIDs.keys())+1
            print('Importing loads from {} and associating LCIDs with LSID {}...'.format(filename,LSID))
            LCIDs = []
            # Get indices for the coordinates and set them to start at 0
            if xdir<0:
                xfactor=-1
                xdir=abs(xdir)-1
            else:
                xfactor=1
                xdir -= 1
            if ydir<0:
                yfactor=-1
                ydir=abs(ydir)-1
            else:
                yfactor=1
                ydir -= 1
            load_file = open(filename, 'r')
            for line in load_file:
                data = line.replace('\n','').split(',')
                if not len(data)==11:
                    print('When importing the loads csv {}, a line was found to have'\
                          ' more than 11 entried. Ensure that the load csv uses the'\
                          ' following syntax:\n XID,LCID,x,y,z,Fx,Fy,Fz,Mx,My,Mz.\n The'\
                          ' line in question to be skipped is:\n{}'.format(filename,line))
                else:
                    try:
                        XID = int(data[0])
                        LCID = int(data[1])
                        xtmp = float(data[2])
                        ytmp = float(data[3])
                        ztmp = float(data[4])
                        Fxtmp = float(data[5])*LF
                        Fytmp = float(data[6])*LF
                        Fztmp = float(data[7])*LF
                        Mxtmp = float(data[8])*LF
                        Mytmp = float(data[9])*LF
                        Mztmp = float(data[10])*LF
                        if XID in self.sectionLoads.keys():
                            section_load_dict = self.sectionLoads[XID]
                        else:
                            section_load_dict = {}
                        if LCID in section_load_dict.keys():
                            print('WARNING: User is trying to import LCID {} for '\
                                  ' cross-section {}, but that LC already exists.'\
                                  ' Overwritting previous load cases:\n{}'.format(LCID,XID,section_load_dict[LCID]))
                        
                        # Determine the CSYS x and y directions
                        if xdir==0:
                            x = xtmp
                            Fx = Fxtmp
                            Mx = Mxtmp
                            xvec = np.array([1.,0.,0.])*xfactor
                        elif xdir==1:
                            x = ytmp
                            Fx = Fytmp
                            Mx = Mytmp
                            xvec = np.array([0.,1.,0.])*xfactor
                        else:
                            x = ztmp
                            Fx = Fztmp
                            Mx = Mztmp
                            xvec = np.array([0.,0.,1.])*xfactor
                        if ydir==0:
                            y = xtmp
                            Fy = Fxtmp
                            My = Mxtmp
                            yvec = np.array([1.,0.,0.])*yfactor
                        elif ydir==1:
                            y = ytmp
                            Fy = Fytmp
                            My = Mytmp
                            yvec = np.array([0.,1.,0.])*yfactor
                        else:
                            y = ztmp
                            Fy = Fztmp
                            My = Mztmp
                            yvec = np.array([0.,0.,1.])*yfactor
                        #Determine the CSYS z direction
                        ivec = xvec[1]*yvec[2]-yvec[1]*xvec[2]
                        jvec = -xvec[0]*yvec[2]+yvec[0]*xvec[2]
                        kvec = xvec[0]*yvec[1]-yvec[0]*xvec[1]
                        if (ivec==0) and (jvec==0):
                            Fz = Fztmp*kvec
                            Mz = Mztmp*kvec
                        elif (ivec==0) and (kvec==0):
                            Fz = Fytmp*jvec
                            Mz = Mytmp*jvec
                        elif (jvec==0) and (kvec==0):
                            Fz = Fxtmp*ivec
                            Mz = Mxtmp*ivec
                        else:
                            raise ValueError('Problem')
                        section_load_dict[LCID] = [x,y,Fx,Fy,Fz,Mx,My,Mz]
                        self.sectionLoads[XID] = section_load_dict
                        if not LCID in LCIDs:
                            LCIDs += [LCID]
                    except ValueError:
                        print('Skipping the following line that could not be converted:\n{}'.format(line))
        self.sectionLSIDs[LSID]=LCIDs
        print('Finished importing loads!')
    def loadSection(self):
        if self.activeSection:
            tmpXSect = self.activeSection
            LCIDs = []
            flag=True
            if not tmpXSect.analyzed:
                print("This cross-section must be analyzed before loads can be"
                " applied.")
            else:
                # Load the section with all of the loads
                if self.GUI.load_source_drop.currentText()=='Custom':
                    F = [self.xsectFx,self.xsectFy,self.xsectFz,self.xsectMx,\
                         self.xsectMy,self.xsectMz]
                    tmpXSect.calcWarpEffects(0,F)
                    LCIDs = [0]
                elif self.GUI.load_source_drop.currentText()=='Load ID' and \
                    not self.GUI.load_source_vals.currentText()=='':
                    F = [self.xsectFx,self.xsectFy,self.xsectFz,self.xsectMx,\
                         self.xsectMy,self.xsectMz]
                    LCID = int(self.GUI.load_source_vals.currentText())
                    tmpXSect.calcWarpEffects(LCID,F)
                    LCIDs = [LCID]
                elif self.GUI.load_source_drop.currentText()=='All' and tmpXSect.XID in self.sectionLoads.keys():
                    tmpSectionLoads = self.sectionLoads[tmpXSect.XID]
                    LCIDs = list(tmpSectionLoads.keys())
                    for LCID, loads in tmpSectionLoads.items():
                        newLoads = tmpXSect.transformLoads(loads)
                        F = [newLoads[0,0],newLoads[1,0],newLoads[2,0],newLoads[3,0],\
                             newLoads[4,0],newLoads[5,0]]
                        tmpXSect.calcWarpEffects(LCID,F)
                elif self.GUI.load_source_drop.currentText()=='Set':
                    #Get the intersection of LCIDs in the set and added to the section
                    LSID = int(self.GUI.load_source_vals.currentText())
                    setLCIDs = self.sectionLSIDs[LSID]
                    tmpSectionLoads = self.sectionLoads[tmpXSect.XID]
                    sectionLCIDs = list(tmpSectionLoads.keys())
                    LCIDs = [value for value in setLCIDs if value in sectionLCIDs]
                    ommittedLCIDs = [value for value in setLCIDs if value not in LCIDs]
                    print('User tried to apply loads from set {} on section {}.'\
                          ' This set includes LCIDs {} which have not been added to'\
                          ' section {}'.format(LSID,tmpXSect.XID,ommittedLCIDs,tmpXSect.XID))
                    for LCID in LCIDs:
                        loads = tmpSectionLoads[LCID]
                        newLoads = tmpXSect.transformLoads(loads)
                        F = [newLoads[0,0],newLoads[1,0],newLoads[2,0],newLoads[3,0],\
                             newLoads[4,0],newLoads[5,0]]
                        tmpXSect.calcWarpEffects(LCID,F)
                else:
                    print('Cannot load the section with any loads since none'\
                              ' have been added to section {}.'.format(tmpXSect.XID))
                    flag=False
                if flag:
                    if self.GUI.contLimSetting.currentText()=='Max-Min':
                        tmpXSect.plotWarped(LCIDs,contour=self.GUI.contourBox.currentText(),
                                            warpScale=self.warpFactor,contLimMin=self.minContLim,\
                                            contLimMax=self.maxContLim)
                        if not (self.minContLim=='' and self.maxContLim==''):
                            if self.minContLim=='':
                                tmpXSect.plotWarped(LCIDs,contour=self.GUI.contourBox.currentText(),
                                                    warpScale=self.warpFactor,contLimMax=self.maxContLim)
                            elif self.maxContLim=='':
                                tmpXSect.plotWarped(LCIDs,contour=self.GUI.contourBox.currentText(),
                                                    warpScale=self.warpFactor,contLimMin=self.minContLim)
                    else:
                        tmpXSect.plotWarped(LCIDs,contour=self.GUI.contourBox.currentText(),
                                                warpScale=self.warpFactor)
                    #except Exception as e: print(str(e))
                    md = gl.MeshData(vertexes=tmpXSect.vertices, faces=tmpXSect.surfaces,\
                                     vertexColors=tmpXSect.colors)
                    mesh = gl.GLMeshItem(meshdata=md,drawEdges=True,edgeColor=(0, 0, 0, 1))
                    #try:
                    if self.currentXSectMesh:
                        self.GUI.XSectView.removeItem(self.currentXSectMesh)
                    self.currentXSectMesh = mesh
                    self.GUI.XSectView.addItem(mesh)
                    self.xsectColorScale = tmpXSect.contLim
                    self.GUI.updateColorBar()
        else:
            print('No active cross-section.')
    def createBeam(self,BID,x1,x2,XID,noe,btype,sNID,sEID,chordVec):
        xsect = self.sections.get(XID)
        if not xsect.analyzed:
            xsect.xSectionAnalysis()
        #try:
        self.beams.addBeam(BID,x1,x2,xsect,noe,btype,sNID,sEID,chordVec)
        #except Exception as e: print(str(e))
        newBeam = self.beams.get(BID)
        for key in newBeam.elems.keys():
            if key in self.elements.elemDict.keys():
                raise ValueError('Error: You cannot overwrite beam elements already '
                                 'existent in the model.')
        print('Begin iterating through elements')
        for EID, elem in newBeam.elems.items():
            self.elements.addBeamElement(elem)
            print('verifying no node overwrite')
            if not elem.n1.NID in self.nodes.keys():
                self.nodes[elem.n1.NID] = elem.n1
            else:
                tmpNode = self.nodes[elem.n1.NID]
                for i in range(0,3):
                    if elem.n1.x[i]-tmpNode.x[i]>1e-8:
                        raise ValueError('In creating new Beam %d, you made'
                                         ' beam element %d was created with'
                                         ' node %d. Node %d already exists'
                                         ' in the model and the coordinates'
                                         ' between the two do not match.'
                                         %(BID,elem.EID,elem.n1.NID,elem.n1.NID) )
            if not elem.n2.NID in self.nodes.keys():
                self.nodes[elem.n2.NID] = elem.n2
            else:
                tmpNode = self.nodes[elem.n2.NID]
                for i in range(0,3):
                    if elem.n2.x[i]-tmpNode.x[i]>1e-8:
                        raise ValueError('In creating new Beam %d, you made'
                                         ' beam element %d was created with'
                                         ' node %d. Node %d already exists'
                                         ' in the model and the coordinates'
                                         ' between the two do not match.'
                                         %(BID,elem.EID,elem.n2.NID,elem.n2.NID) )
            beamVis = gl.GLLinePlotItem(pos=np.array(elem.vertices),color=elem.colors,antialias=True)
            elem.beamVis = beamVis
            self.globalModels += [beamVis]
    def assembleGlobalModel(self,analysisType,CID,LID=-1):
        """Assembles the global model.
        
        Primarily intended as a private method, this method assembles the
        necessary matricies for the finite element model. For example, if the
        user is executing a linear static analysis, the model will generate the
        global and reduced stiffness matricies as well as the global and
        reduced force vector.
        
        The three currently suported assemblies are for (which correspond to
        the analysis type) are linear static (1) and normal mode analysis (3).
        
        :Args:
        
        - `analysisType (int)`: The analysis type to be executed by the model.
        - `LID (int)`: If a linear static analysis is executed, this LID
            corresponds to which load set should be applied to the model.
        - `static4BuckName (str)`: The analysis name of the static analysis
            should a corresponding linear buckling analysis be run.
            
        :Returns:
        
        - None
        
        .. Note:: When a flutter analysis is executed, the normal mode assebly
        is executed.
        """
        nids = self.nodes.keys()
        NID2ind = {}
        for i in range(0,len(nids)):
            NID2ind[nids[i]] = i
        self.NID2ind = NID2ind
        # For a Linear Static Analysis
        if analysisType==1:
            tmpLoad = None
            if LID in self.Loads.keys():
                tmpLoad = self.loads[LID]
            else:
                raise ValueError('You selected a load ID that does not exist.')
            # Determine the degrees of freedom in the model
            DOF = 6*len(self.nodes)
            # Initialize the global stiffness matrix
            Kg = np.zeros((DOF,DOF),dtype=float)
            # Initialize the global force vector
            Fg = np.zeros((DOF,1),dtype=float)
            # For all of the elements in the elems array
            for EID, elem in self.elements.elemDict.items():
                # Apply the distributed load to the element
                if EID in tmpLoad.distributedLoads.keys():
                    elem.applyDistributedLoad(tmpLoad.distributedLoads[EID])
                # Determine the node ID's associated with the elem
                nodes = [elem.n1.NID,elem.n2.NID]
                # For both NID's
                for i in range(0,len(nodes)):
                    # The row in the global matrix (an integer correspoinding to
                    # the NID)
                    row = NID2ind[nodes[i]]
                    # Add the elem force vector to the global matrix
                    Fg[6*row:6*row+6,:] = Fg[6*row:6*row+6,:] +\
                        elem.Fe[6*i:6*i+6,:]
                    for j in range(0,len(nodes)):
                        # Determine the column range for the NID
                        col = NID2ind[nodes[j]]
                        # Add the elem stiffness matrix portion to the global
                        # stiffness matrix
                        Kg[6*row:6*row+6,6*col:6*col+6] = Kg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Ke[6*i:6*i+6,6*j:6*j+6]
            # Apply the point loads to the model
            for NID in tmpLoad.pointLoads.keys():
                # The row in the global matrix (an integer correspoinding to
                # the NID)
                row = NID2ind[NID]
                Fg[6*row:6*row+6,:]=Fg[6*row:6*row+6,:]\
                    +np.reshape(tmpLoad.pointLoads[NID],(6,1))
            # Save the global stiffness matrix
            self.Kg = Kg
            # Save the global force vector
            self.Fg = Fg
            # Determine the list of NIDs to be contrained
            constraint = self.constraints[CID]
            cnds = sorted(list(constraint.cons.keys()))
            # Initialize the number of equations to be removed from the system
            deleqs = 0
            # For the number of constrained NIDs
            for i in range(0,len(cnds)):
                # The row range to be removed associated with the NID
                row = self.NID2ind[cnds[i]]
                # Determine which DOF are to be removed
                tmpcst = constraint.cons[cnds[i]]
                # For all of the degrees of freedom to be removed
                for j in range(0,len(tmpcst)):
                    # Remove the row associated with the jth DOF for the ith NID
                    Fg = np.delete(Fg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    # Incremend the number of deleted equations
                    deleqs += 1
            # Save the reduced global force vector
            self.Fgr = Fg
            # Save the reduced global stiffness matrix
            self.Kgr = Kg
            
        # For a Normal Modes Analysis
        if analysisType==3:
            # Determine the degrees of freedom in the model
            DOF = 6*len(self.nids)
            # Initialize the global stiffness matrix
            Kg = np.zeros((DOF,DOF),dtype=float)
            Mg = np.zeros((DOF,DOF),dtype=float)
            # For all of the elements in the elems array
            for elem in self.elems:
                # Determine the node ID's associated with the elem
                nodes = [elem.n1.NID,elem.n2.NID]
                # For both NID's
                for i in range(0,len(nodes)):
                    # The row in the global matrix (an integer correspoinding to
                    # the NID)
                    row = self.NID2ind[nodes[i]]
                    for j in range(0,len(nodes)):
                        # Determine the column range for the NID
                        col = self.NID2ind[nodes[j]]
                        # Add the elem stiffness matrix portion to the global
                        # stiffness matrix
                        Kg[6*row:6*row+6,6*col:6*col+6] = Kg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Ke[6*i:6*i+6,6*j:6*j+6]
                        # Add the element mass matrix portion to the global mass matrix
                        Mg[6*row:6*row+6,6*col:6*col+6] = Mg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Me[6*i:6*i+6,6*j:6*j+6]                        
            # Save the global stiffness matrix
            self.Kg = Kg
            # Save the global mass matrix
            self.Mg = Mg
            # Determine the list of NIDs to be contrained
            constraint = self.constraints[CID]
            cnds = sorted(list(constraint.cons.keys()))
            # Initialize the number of equations to be removed from the system
            deleqs = 0
            # For the number of constrained NIDs
            for i in range(0,len(cnds)):
                # The row range to be removed associated with the NID
                row = self.NID2ind[cnds[i]]
                # Determine which DOF are to be removed
                tmpcst = constraint.cons[cnds[i]]
                # For all of the degrees of freedom to be removed
                for j in range(0,len(tmpcst)):
                    # Remove the row associated with the jth DOF for the ith NID
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    # Incremend the number of deleted equations
                    deleqs += 1
            # Save the reduced global stiffness matrix
            self.Kgr = Kg
            # Save the reduced global mass matrix
            self.Mgr = Mg
    def staticAnalysis(self,LID,**kwargs):
        """Linear static analysis.
        
        This method conducts a linear static analysis on the model. This will
        calculate all of the unknown displacements in the model, and save not
        only dispalcements, but also internal forces and moments in all of the
        beam elements.
        
        :Args:
        
        - `LID (int)`: The ID corresponding to the load set to be applied to
            the model.
        - `analysis_name (str)`: The string name to be associated with this
            analysis. By default, this is chosen to be 'analysis_untitled'.
        
        :Returns:
        
        - None
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.assembleGlobalModel(1,LID=LID)
        # Prepare the reduced stiffness matrix for efficient LU decomposition solution
        lu,piv = linalg.lu_factor(self.Kgr)
        # Solve global displacements
        u = linalg.lu_solve((lu,piv),self.Fgr)
        self.ur = u
        # Generate list of constraint keys
        ckeys = sorted(list(self.const.keys()))
        # For each node constrained
        for i in range(0,len(ckeys)):
            # Establish temporary node constrined
            tmpconst = self.const[ckeys[i]]
            # For each DOF contrained on the temporary node
            for j in range(0,len(tmpconst)):
                #Insert a zero for the "displacement"
                u = np.insert(u,self.nodeDict[ckeys[i]]*6+(tmpconst[j]-1),0,axis=0)
        self.u[analysis_name] = u
        #Solve for the reaction forces in the elements
        # For all of the beam elements in the model
        for elem in self.elems:
            # If the element is a Tbeam
            if (elem.type=='Tbeam'):
                #Populate the local nodal displacements:
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                U1 = u[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6]
                U2 = u[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6]
                elem.saveNodalDispl(U1,U2,analysis_name=analysis_name)
            elif elem.type=='EBbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                elem.U1 = u[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6]
                elem.U2 = u[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6]
                #Solve for the reaction forces on the first node
                Ke12 = elem.Ke[0:6,6:12]
                #elem.F1 = np.dot(Ke12,elem.U2)+np.dot(Ke13,elem.U3)
                elem.F1 = np.dot(Ke12,elem.U1-elem.U2)
                #Solve for the reaction forces on the second node
                Ke21 = elem.Ke[6:12,0:6]
                elem.F2 = np.dot(Ke21,elem.U1-elem.U2)


class LoadSet:
    """Creates a Model which is used to organize and analyze FEM.
    
    The primary use of LoadSet is to fascilitate the application of many
    different complex loads to a finite element model.
    
    :Attributes:
    
    - `LID (int)`: The integer identifier for the load set object.
    - `pointLoads (dict[pointLoads[NID,F])`: A dictionary mapping applied point
        loads to the node ID's of the node where the load is applied.
    - `distributedLoads (dict[EID,f])`: A dictionary mapping the distributed
        load vector to the element ID of the element where the load is applied.
    
    
    :Methods:
    
    - `__init__`: The constructor of the class. This method initializes the
        dictionaries used by the loads
    - `addPointLoad`: Adds point loads to the pointLoads dictionary attribute.
    - `addDictibutedLoad`: Adds distributed loads to the distributedLoads
        dictionary attribute.

    """
    def __init__(self,LID):
        """Initialized the load set object.
        
        This method is a simple constructor for the load set object.
        
        :Args:
        
        - `LID (int)`: The integer ID linked with the load set object.
        
        :Returns:
        
        - None
        
        """
        self.LID = LID
        self.pointLoads = {}
        self.distributedLoads = {}
    def addPointLoad(self,F,NID):
        """Initialized the load set ibject.
        
        This method is a simple constructor for the load set object.
        
        :Args:
        
        - `LID (int)`: The integer ID linked with the load set object.
        
        :Returns:
        
        - None
        
        """
        if NID in self.pointLoads.keys():
            self.pointLoads[NID]=self.pointLoads[NID]+F
        else:
            self.pointLoads[NID]=F
    def addDistributedLoad(self,f,eid):
        """Initialized the load set ibject.
        
        This method is a simple constructor for the load set object.
        
        :Args:
        
        - `LID (int)`: The integer ID linked with the load set object.
        
        :Returns:
        
        - None
        
        """
        if eid in self.distributedLoads.keys():
            self.distributedLoads[eid]=self.distributedLoads[eid]+f
        else:
            self.distributedLoads[eid]=f
            
class ConstraintSet:
    """Creates a Model which is used to organize and analyze FEM.
    
    The primary use of LoadSet is to fascilitate the application of many
    different complex loads to a finite element model.
    
    :Attributes:
    
    - `LID (int)`: The integer identifier for the load set object.
    - `pointLoads (dict[pointLoads[NID,F])`: A dictionary mapping applied point
        loads to the node ID's of the node where the load is applied.
    - `distributedLoads (dict[EID,f])`: A dictionary mapping the distributed
        load vector to the element ID of the element where the load is applied.
    
    
    :Methods:
    
    - `__init__`: The constructor of the class. This method initializes the
        dictionaries used by the loads
    - `addPointLoad`: Adds point loads to the pointLoads dictionary attribute.
    - `addDictibutedLoad`: Adds distributed loads to the distributedLoads
        dictionary attribute.

    """
    def __init__(self,CID):
        """Initialized the load set ibject.
        
        This method is a simple constructor for the load set object.
        
        :Args:
        
        - `LID (int)`: The integer ID linked with the load set object.
        
        :Returns:
        
        - None
        
        """
        self.CID = CID
        self.cons = {}
    def addConstraints(self,NID,DOF):
        """Initialized the load set ibject.
        
        This method is a simple constructor for the load set object.
        
        :Args:
        
        - `LID (int)`: The integer ID linked with the load set object.
        
        :Returns:
        
        - None
        
        """
        self.cons[NID] = DOF
            
        