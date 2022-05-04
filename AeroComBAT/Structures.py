import numpy as np
import operator
import matplotlib as mpl
import pyqtgraph as pg

from .tabulate import tabulate
from .Utils import *
from .Visualizer import VisualModel

from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix, eye, csc_matrix, coo_matrix
from scipy.sparse.linalg import minres, spsolve, inv#, dsolve, bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk
from scipy import linalg
from numpy.linalg import solve
import numpy.polynomial.polynomial as poly

import time

import collections as coll

import pyqtgraph.opengl as gl


class Node:
    """Creates a node object.

    Creates a node object for global beam analysis.

    :Attributes:

    - `NID (int)`: The integer identifier given to the object.
    - `x1 (float array)`: The array containing the 3 x-y-z coordinates of the
        node.
    - `summary (str)`: A string which is a tabulated respresentation and
        summary of the important attributes of the object.

    :Methods:

    - `printSummary`: This method prints out basic information about the node
        object, such as it's node ID and it's x-y-z coordinates

    """
    def __init__(self,NID,x,y,z):
        """Initializes the node object.

        :Args:

        - `nid (int)`: The desired integer node ID
        - `x (float)`: The global x-coordinate of the node.
        - `y (float)`: The global y-coordinate of the node.
        - `z (float)`: The global z-coordinate of the node.

        :Returns:

        - None

        """
        # Verify that a correct NID was given
        if type(NID) is int:
            self.NID = NID
        else:
            raise TypeError('The node ID given was not an integer.')
        if not (isinstance(x,float) or isinstance(y,float) or isinstance(z,float)):
            raise ValueError('The x, y, and z coordinates must be floats.')
        self.x = [x,y,z]
        self.type='Node'
    def printSummary(self):
        """Prints basic information about the node.

        The printSummary method prints out basic node attributes in an organized
        fashion. This includes the node ID and x-y-z global coordinates.

        :Args:

        - None

        :Returns:

        - A printed table including the node ID and it's coordinates

        """
        print(tabulate(([[self.NID,self.x]]),('NID','Coordinates'),tablefmt="fancy_grid"))
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'NODE,{},{},{},{}'.format(self.NID,self.x[0],self.x[1],self.x[2])
class NodeLibrary:
    """Creates a node library object.

    This node library holds the nodes to be used for beam element
    analysis. Furthermore, it can be used to generate new node objects
    to be automatically stored within it. See the Node class for further
    details.

    :Attributes:

    - `nodeDict (dict)`: A dictionary which stores node objects as the
        values with the NIDs as the associated keys.

    :Methods:

    - `addNode`: Adds a node to the NodeLib object dictionary.
    - `getNode`: Returns a node object provided an NID
    - `printSummary`: Prints a summary of all of the nodes held within the
        nodeDict dictionary.

    """
    def __init__(self):
        """Initialize NodeLib object.

        The initialization method is mainly used to initialize a dictionary
        which houses node objects.

        :Args:

        - None

        :Returns:

        - None

        """
        self.type='NodeLibrary'
        self.nodeDict = {}

    def add(self,NID, x, y, z):
        """Add a node to the nodeLib object.

        This is the primary method of the class, used to create new node
        obects and then add them to the library for later use.

        :Args:

        - `nid (int)`: The desired integer node ID
        - `x (float)`: The global x-coordinate of the node.
        - `y (float)`: The global y-coordinate of the node.
        - `z (float)`: The global z-coordinate of the node.

        :Returns:

        - None

        """
        if NID in self.nodeDict.keys():
            print('WARNING: Overwritting node %d' %(NID))
            self.nodeDict[NID] = Node(NID, x, y, z)
        else:
            self.nodeDict[NID] = Node(NID, x, y, z)

    def get(self,NID):
        """Method that returns a node from the node libary

        :Args:

        - `NID (int)`: The ID of the node which is desired

        :Returns:

        - `(obj): A node object associated with the key NID

        """
        if not NID in self.nodeDict.keys():
            raise KeyError('The NID provided is not linked with any nodes'+
                'within the supplied node library.')
        return self.nodeDict[NID]
    def getIDs(self):
        return self.nodeDict.keys()
    def delete(self,NID):
        if not NID in self.nodeDict.keys():
            raise KeyError('The NID provided is not linked with any nodes '+
                'within the supplied node library.')
        del self.nodeDict[NID]
    def printSummary(self):
        """Prints summary of all nodes in NodeLib

        A method used to print out tabulated summary of all of the nodes
        held within the node library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the nodes.

        """
        if len(self.nodeDict)==0:
            print('The node library is currently empty.\n')
        else:
            print('The nodes are:')
            for NID, node in self.nodeDict.items():
                node.printSummary()
    def writeToFile(self):
        """Prints summary of all nodes in NodeLib

        A method used to print out tabulated summary of all of the nodes
        held within the node library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the nodes.

        """
        print_statement = []
        if len(self.nodeDict)==0:
            print('The node library is currently empty.\n')
        else:
            for NID, node in self.nodeDict.items():
                print_statement += [node.writeToFile()]
        return print_statement
class XNode:
    """Creates a local cross-sectional node object.

    Creates a node object to be used in cross-sectional analysis.

    :Attributes:

    - `NID (int)`: The integer identifier given to the object.
    - `x1 (float)`: The An array containing the 3 x-y-z coordinates of the
        node.
    - `summary (str)`: A string which is a tabulated respresentation and
        summary of the important attributes of the object.

    :Methods:

    - `printSummary`: This method prints out basic information about the node
        object, such as it's node ID and it's x-y-z coordinates

    """
    def __init__(self,NID,x,y):
        """Initializes the node object.

        :Args:

        - `nid (int)`: The desired integer node ID
        - `x (float)`: The local cross-sectional x-coordinate of the node.
        - `y (float)`: The local cross-sectional y-coordinate of the node.

        :Returns:

        - None

        """
        # Verify that a correct NID was given
        self.XID = None
        if type(NID) is int:
            self.NID = NID
        else:
            raise TypeError('The node ID given was not an integer.')
        if not (isinstance(x,float) or isinstance(y,float)):
            raise ValueError('The x, y, and z coordinates must be floats.')
        self.x = [x,y,0.]
        self.type='XNode'
        self.EIDs = []
    def setXID(self,XID):
        self.XID = XID
    def addEID(self,EID):
        if not EID in self.EIDs:
            self.EIDs += [EID]
    def translate(self,dx,dy):
        self.x = [self.x[0]+dx,self.x[1]+dy,0.]
    def printSummary(self):
        """Prints basic information about the node.

        The printSummary method prints out basic node attributes in an organized
        fashion. This includes the node ID and x-y-z global coordinates.

        :Args:

        - None

        :Returns:

        - A printed table including the node ID and it's coordinates

        """
        print('XNODE {}:'.format(self.NID))
        print(tabulate(([self.x[:2]]),('x-coordinate','y-coordinate'),tablefmt="fancy_grid"))
        print('Referenced by elements: {}'.format(self.EIDs))
        print('Referenced by cross-section {}'.format(self.XID))
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XNODE,{},{},{}'.format(self.NID,self.x[0],self.x[1])
    def writeToNeutral(self):
        return '{},0,0,1,46,0,0,0,0,0,0,{},{},0,0,0,\n'.format(self.NID,self.x[0],self.x[1])

class XNodeLibrary:
    """Creates a cross-sectional node library object.

    This cross-sectional node library holds the nodes to be used for cross-sectional element
    analysis. Furthermore, it can be used to generate new cross-section node objects
    to be automatically stored within it. See the XNode class for further
    details.

    :Attributes:

    - `xnodeDict (dict)`: A dictionary which stores xnode objects as the
        values with the XNIDs as the associated keys.

    :Methods:

    - `addXNode`: Adds an xnode to the XNodeLib object dictionary.
    - `getXNode`: Returns an xnode object provided an XNID
    - `printSummary`: Prints a summary of all of the xnodes held within the
        xnodeDict dictionary.

    """
    def __init__(self):
        """Initialize NodeLib object.

        The initialization method is mainly used to initialize a dictionary
        which houses node objects.

        :Args:

        - None

        :Returns:

        - None

        """
        self.type='XNodeLibrary'
        self.xnodeDict = {}

    def add(self,XNID, x, y):
        """Add a node to the nodeLib object.

        This is the primary method of the class, used to create new xnode
        obects and then add them to the library for later use.

        :Args:

        - `xnid (int)`: The desired integer node ID
        - `x (float)`: The global x-coordinate of the node.
        - `y (float)`: The global y-coordinate of the node.

        :Returns:

        - None

        """
        if XNID in self.xnodeDict.keys():
            print('WARNING: Overwritting node %d' %(XNID))
            self.xnodeDict[XNID] = XNode(XNID, x, y)
        else:
            self.xnodeDict[XNID] = XNode(XNID, x, y)

    def get(self,XNID):
        """Method that returns an xnode from the xnode libary

        :Args:

        - `XNID (int)`: The ID of the xnode which is desired

        :Returns:

        - `(obj): An xnode object associated with the key XNID

        """
        if not XNID in self.xnodeDict.keys():
            raise KeyError('The XNID provided is not linked with any xnodes'+
                'within the supplied xnode library.')
        return self.xnodeDict[XNID]
    def getIDs(self):
        return self.xnodeDict.keys()
    def delete(self,XNID):
        if not XNID in self.xnodeDict.keys():
            raise KeyError('The XNID provided is not linked with any xnodes '+
                'within the supplied xnode library.')
        del self.xnodeDict[XNID]
    def printSummary(self):
        """Prints summary of all xnodes in XNodeLib

        A method used to print out tabulated summary of all of the xnodes
        held within the xnode library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the xnodes.

        """
        if len(self.xnodeDict)==0:
            print('The xnode library is currently empty.\n')
        else:
            print('The xnodes are:')
            for XNID, xnode in self.xnodeDict.items():
                xnode.printSummary()
    def writeToFile(self):
        """Prints summary of all xnodes in XNodeLib

        A method used to print out tabulated summary of all of the xnodes
        held within the xnode library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the xnodes.

        """
        print_statement = []
        if len(self.xnodeDict)==0:
            print('The xnode library is currently empty.\n')
        else:
            for XNID, xnode in self.xnodeDict.items():
                print_statement += [xnode.writeToFile()]
        return print_statement

class Material:
    """creates a linear elastic material object.

    This class creates a material object which can be stored within a
    material library object. The material can be in general orthotropic.

    :Attributes:

    - `name (str)`: A name for the material.
    - `MID (int)`: An integer identifier for the material.
    - `matType (str)`: A string expressing what type of material it is.
        Currently, the supported materials are isotropic, transversely
        isotropic, and orthotropic.
    - `summary (str)`: A string which is a tabulated respresentation and
        summary of the important attributes of the object.
    - `t (float)`: A single float which represents the thickness of a ply if
        the material is to be used in a composite.
    - `rho (float)`: A single float which represents the density of the
        materials.
    - `Smat (6x6 numpy Array[float])`: A numpy array representing the
        compliance matrix in the fiber coordinate system.*
    - `Cmat (6x6 numpy Array[float])`: A numpy array representing the
        stiffness matrix in the fiber coordinate system.*

    :Methods:

    - `printSummary`: This method prints out basic information about the
        material, including the type, the material constants, material
        thickness, as well as the tabulated stiffness or compliance
        matricies if requested.

    .. Note:: The CQUADX element assumes that the fibers are oriented along
        the (1,0,0) in the global coordinate system.

    """ # why is thickness defined in material and not ply?
    def __init__(self,MID,name,matType,mat_constants,mat_t=0.,**kwargs):
        """Creates a material object

        The main purpose of this class is assembling the constitutive
        relations. Regardless of the analysis


        :Args:

        - `MID (int)`: Material ID.
        - `name (str)`: Name of the material.
        - `matType (str)`: The type of the material. Supported material types
            are "ISO", "TISO", and "ORTHO".
        - `mat_constants (1xX Array[Float])`: The requisite number of material
            constants required for any structural analysis. Note, this
            array includes the material density. For example, an isotropic
            material needs 2 elastic material constants, so the total
            length of mat_constants would be 3, 2 elastic constants and the
            density.
        - `mat_t (float)`: The thickness of 1-ply of the material


        :Returns:

        - None

        .. Note:: While this class supports material direction rotations, it is more
            robust to simply let the CQUADX and Mesher class handle all material
            rotations.

        """
        # Initialize Material Name
        #self.type='MAT'
        self.name = name
        # Material identification
        # Error checking to verify ID is of type int
        if type(MID) is int:
            self.MID = MID
        else:
            raise TypeError('The material ID given was not an integer') #repeats
        # Material Type(string) - isotropic, transversely isotropic, otrthotropic
        self.matType = matType
        # Material Constants(array if floats) - depends on matType
        saved_mat_const = []
        failure_const = []

        # ISOTROPIC MATERIAL
        if matType=='ISO' and len(mat_constants)==6:
            self.type = 'MAT_ISO'
            # mat_constants expected = [E, nu, rho]
            E = mat_constants[0]
            nu = mat_constants[1]
            rho = mat_constants[2]
            Ftu = mat_constants[3]
            Fcy = mat_constants[4]
            Fsu = mat_constants[5]
            G = E/(2*(1+nu))
            saved_mat_const = [E, E, E, nu, nu, nu, G, G, G, rho]
            # failure_const = [Xt,Xc,Yt,Yc,Zt,Zc,Syz,Sxz,Sxy]
            failure_const = [Ftu,Fcy,Ftu,Fcy,Ftu,Fcy,Fsu,Fsu,Fsu]
            self.summary = tabulate([[MID,'ISO',E,nu,G,rho,mat_t]],\
            ('MID','Type','E','nu','G','rho','t'),tablefmt="fancy_grid")
            self.strengths = tabulate([[Ftu,Fcy,Fsu]],\
            ('Ftu','Fcy','Fsu'),tablefmt="fancy_grid")

        # TRANSVERSELY ISOTROPIC MATERIAL
        elif matType=='TISO' and len(mat_constants)==11:
            self.type = 'MAT_TISO'
            # mat_constants expected = [E1, E2, nu_23, nu_12, G_12, rho]
            E1 = mat_constants[0]
            E2 = mat_constants[1]
            nu_23 = mat_constants[2]
            nu_12 = mat_constants[3]
            G_12 = mat_constants[4]
            G_23 = E2/(2*(1+nu_23))
            rho = mat_constants[5]
            Xt = mat_constants[6]
            Xc = mat_constants[7]
            Yt = mat_constants[8]
            Yc = mat_constants[9]
            S = mat_constants[10]
            # failure_const = [Xt,Xc,Yt,Yc,Zt,Zc,Syz,Sxz,Sxy]
            failure_const = [Xt,Xc,Yt,Yc,Yt,Yc,S,S,S]
            saved_mat_const = [E1, E2, E2, nu_23, nu_12, nu_12, G_23, G_12, G_12, rho]
            self.summary = tabulate([[MID,'TISO',E1,E2,nu_23,nu_12,G_23,G_12,rho,mat_t]],\
            ('MID','Type','E1','E2','nu_23','nu_12','G_23','G_12',\
            'rho','t'),tablefmt="fancy_grid")
            self.strengths = tabulate([[Xt,Xc,Yt,Yc,S]],\
            ('Xt','Xc','Yt','Yc','S'),tablefmt="fancy_grid")

        # ORTHOTROPIC MATERIAL
        elif matType=='ORTHO' and len(mat_constants)==19:
            self.type = 'MAT_ORTHO'
            # mat_constants expected = [E1,E2,E3,nu_23,nu_13,nu_12,G_23,G_13,G_12,rho]
            saved_mat_const = mat_constants #re-order
            E1 = mat_constants[0]
            E2 = mat_constants[1]
            E3 = mat_constants[2]
            nu_23 = mat_constants[3]
            nu_13 = mat_constants[4]
            nu_12 = mat_constants[5]
            G_23 = mat_constants[6]
            G_13 = mat_constants[7]
            G_12 = mat_constants[8]
            rho = mat_constants[9]
            Xt = mat_constants[10]
            Xc = mat_constants[11]
            Yt = mat_constants[12]
            Yc = mat_constants[13]
            Zt = mat_constants[14]
            Zc = mat_constants[15]
            Syz = mat_constants[16]
            Sxz = mat_constants[17]
            Sxy = mat_constants[18]
            # failure_const = [Xt,Xc,Yt,Yc,Zt,Zc,Syz,Sxz,Sxy]
            failure_const = [Xt,Xc,Yt,Yc,Zt,Zc,Syz,Sxz,Sxy]
            self.summary = tabulate([[MID,'ORTHO',E1,E2,E3,nu_23,nu_13,nu_12,G_23,G_13,G_12,rho,mat_t]],\
            ('MID','Type','E1','E2','E3','nu_23','nu_13','nu_12',\
            'G_23','G_13','G_12','rho','t'),tablefmt="fancy_grid")
            self.strengths = tabulate([[Xt,Xc,Yt,Yc,Zt,Zc,Syz,Sxz,Sxy]],\
            ('Xt','Xc','Yt','Yc','Zt','Zc','Syz','Sxz','Sxy'),tablefmt="fancy_grid")
        else:
            raise ValueError('\nMaterial %s was not entered correctly. Possible '
            'material types include "ISO", "TISO", or "ORTHO." In '
            'addition, "mat_constants" must then be of length 6, 11, or 19 '
            'respectively. Refer to documentation for more clarification.\n' %(name))
        # Store material constants such that:
        self.E1 = saved_mat_const[0]
        self.E2 = saved_mat_const[1]
        self.E3 = saved_mat_const[2]
        self.nu_23 = saved_mat_const[3]
        self.nu_13 = saved_mat_const[4]
        self.nu_12 = saved_mat_const[5]
        self.G_23 = saved_mat_const[6]
        self.G_13 = saved_mat_const[7]
        self.G_12 = saved_mat_const[8]
        self.rho = saved_mat_const[9]
        self.t = mat_t

        # Store material strengths
        # failure_const = [Xt,Xc,Yt,Yc,Zt,Zc,S]
        self.Xt = failure_const[0]
        self.Xc = failure_const[1]
        self.Yt = failure_const[2]
        self.Yc = failure_const[3]
        self.Zt = failure_const[4]
        self.Zc = failure_const[5]
        self.Syz = failure_const[6]
        self.Sxz = failure_const[7]
        self.Sxy = failure_const[8]
        # Initialize the compliance matrix in the local fiber 123 CSYS:
        self.Smat = np.array([[1./self.E1,-self.nu_12/self.E1,-self.nu_13/self.E1,0.,0.,0.],\
                              [-self.nu_12/self.E1,1./self.E2,-self.nu_23/self.E2,0.,0.,0.],\
                              [-self.nu_13/self.E1,-self.nu_23/self.E2,1./self.E3,0.,0.,0.],\
                              [0.,0.,0.,1./self.G_23,0.,0.],\
                              [0.,0.,0.,0.,1./self.G_13,0.],\
                              [0.,0.,0.,0.,0.,1./self.G_12]])
        # Solve for the material stiffness matrix
        self.Cmat = np.linalg.inv(self.Smat)
    def printSummary(self,**kwargs):
        """Prints a tabulated summary of the material.

        This method prints out basic information about the
        material, including the type, the material constants, material
        thickness, as well as the tabulated stiffness or compliance
        matricies if requested.

        :Args:

        - `compliance (str)`: A boolean input to signify if the compliance
            matrix should be printed.
        - `stiffness (str)`: A boolean input to signify if the stiffness matrix
            should be printed.

        :Returns:

        - String print out containing the material name, as well as material
            constants and other defining material attributes. If requested
            this includes the material stiffness and compliance matricies.

        """
        # Print Name
        print(self.name)
        # Print string summary attribute
        print('Mechanical Properties:')
        print(self.summary)
        print('Material Strengths:')
        print(self.strengths)
        # Print compliance matrix if requested
        if kwargs.pop('compliance',False):
            print('COMPLIANCE MATRIX')
            print('xyz cross-section CSYS:')
            print(tabulate(self.Smat,tablefmt="fancy_grid"))
        # Print Stiffness matrix if requested
        if kwargs.pop('stiffness',False):
            print('STIFFNESS MATRIX')
            print('xyz cross-section CSYS:')
            print(tabulate(np.around(self.Cmat,decimals=4),tablefmt="fancy_grid"))
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        if self.type=='MAT_ISO':
            return 'MAT_ISO,{},{},{},{},{},{},{},{},{}'.format(self.MID,\
                            self.name,self.E1,self.nu_12,self.rho,self.t,\
                            self.Xt,self.Xc,self.Sxy)
        elif self.type=='MAT_TISO':
            return 'MAT_TISO,{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(self.MID,\
                            self.name,self.E1,self.E2,self.nu_23,self.nu_12,\
                            self.G_12,self.rho,self.t,self.Xt,self.Xc,self.Yt,\
                            self.Yc,self.Sxy)
        elif self.type=='MAT_ORTHO':
            return 'MAT_ORTHO,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(self.MID,\
                            self.name,self.E1,self.E2,self.E3,self.nu_23,\
                            self.nu_13,self.nu_12,self.G_23,self.G_13,\
                            self.G_12,self.rho,self.t,self.Xt,self.Xc,self.Yt,\
                            self.Yc,self.Zt,self.Zc,self.Syz,self.Sxz,self.Sxy)
    def writeToNeutral(self):
        matStr = ''
        if self.type=='MAT_ISO':
            typeInt = 0
        elif self.type=='MAT_TISO':
            typeInt = 2
        elif self.type=='MAT_ORTHO':
            typeInt = 2
        matStr += '{},-601,55,{},0,1,0,\n'.format(self.MID,typeInt)
        matStr += '{}\n'.format(self.name)
        matStr += '10,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '25,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,\n'
        matStr += '200,\n'
        matStr += '{},{},{},{},{},{},{},{},{},{},\n'.format(self.E1,self.E2,self.E3,\
                                                        self.G_23,self.G_13,self.G_12,\
                                                        self.nu_23,self.nu_13,self.nu_12,\
                                                        self.Cmat[0,0])
        matStr += '{},{},{},{},{},{},{},{},{},{},\n'.format(self.Cmat[0,1],self.Cmat[0,2],\
                                                        self.Cmat[0,3],self.Cmat[0,4],\
                                                        self.Cmat[0,5],self.Cmat[1,1],\
                                                        self.Cmat[1,2],self.Cmat[1,3],\
                                                        self.Cmat[1,4],self.Cmat[1,5])
        matStr += '{},{},{},{},{},{},{},{},{},{},\n'.format(self.Cmat[2,2],self.Cmat[2,3],\
                                                        self.Cmat[2,4],self.Cmat[2,5],\
                                                        self.Cmat[3,3],self.Cmat[3,4],\
                                                        self.Cmat[3,5],self.Cmat[4,4],\
                                                        self.Cmat[4,5],self.Cmat[5,5])
        matStr += '0.,0.,0.,0.,0.,0.,0,0.,0.,0,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,{},\n'.format(self.rho)
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\n'
        matStr += '50,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '70,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'
        matStr += '0,0,0,0,0,0,0,0,0,0,\n'

        propStr = ''
        propStr += '{},110,{},17,1,0,0,\n'.format(self.MID,self.MID)
        propStr += '{}\n'.format(self.name)
        propStr += '0,0,0,0,0,0,0,0,\n'
        propStr += '10,\n'
        propStr += '{},{},0,0,0,0,0,0,\n'.format(self.MID,self.MID)
        propStr += '0,0,\n'
        propStr += '78,\n'
        propStr += '0.1,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,0.,0.,\n'
        propStr += '0.,0.,0.,\n'
        propStr += '0,\n'
        propStr += '0,\n'

        return matStr, propStr

class MaterialLibrary:
    """Creates a material library object.

    This material library holds the materials to be used for any type of
    analysis. Furthermore, it can be used to generate new material objects
    to be automatically stored within it. See the Material class for suported
    material types.

    :Attributes:

    - `matDict (dict)`: A dictionary which stores material objects as the
        values with the MIDs as the associated keys.

    :Methods:

    - `addMat`: Adds a material to the MaterialLib object dictionary.
    - `getMat`: Returns a material object provided an MID
    - `printSummary`: Prints a summary of all of the materials held within the
        matDict dictionary.

    """
    def __init__(self):
        """Initialize MaterialLib object.

        The initialization method is mainly used to initialize a dictionary
        which houses material objects.

        :Args:

        - None

        :Returns:

        - None

        """
        self.type='MaterialLibrary'
        self.matDict = {}
    def add(self,MID, mat_name, mat_type, mat_constants,mat_t=0.,**kwargs):
        """Add a material to the MaterialLib object.

        This is the primary method of the class, used to create new material
        obects and then add them to the library for later use.

        :Args:

        - `MID (int)`: Material ID.
        - `name (str)`: Name of the material.
        - `matType (str)`: The type of the material. Supported material types
            are "iso", "trans_iso", and "ortho".
        - `mat_constants (1xX Array[Float])`: The requisite number of material
            constants required for any structural analysis. Note, this
            array includes the material density. For example, an isotropic
            material needs 2 elastic material constants, so the total
            length of mat_constants would be 3, 2 elastic constants and the
            density.
        - `mat_t (float)`: The thickness of 1-ply of the material
        - `th (1x3 Array[float])`: The angles about which the material can be
            rotated when it is initialized. In degrees.
        - `overwrite (bool)`: Input used in order to define whether the
            material being added can overwrite another material already
            held by the material library with the same MID.

        :Returns:

        - None

        """
        # Optional argument for material direction rotation
        th = kwargs.pop('th', [0,0,0])
        if MID in self.matDict.keys():
            print('WARNING: Overwritting material %d' %(MID))
            self.matDict[MID] = Material(MID, mat_name, mat_type, mat_constants,mat_t,th=th)
        else:
            self.matDict[MID] = Material(MID, mat_name, mat_type, mat_constants,mat_t,th=th)

    def get(self,MID):
        """Method that returns a material from the material libary

        :Args:

        - `MID (int)`: The ID of the material which is desired

        :Returns:

        - `(obj): A material object associated with the key MID

        """
        if not MID in self.matDict.keys():
            raise KeyError('MID {} is not linked with any materials within the\
                           supplied material library.'.format(MID))
        return self.matDict[MID]
    def getIDs(self):
        return self.matDict.keys()
    def delete(self,MID):
        if not MID in self.matDict.keys():
            raise KeyError('MID {} is not linked with any materials within the\
                           supplied material library.'.format(MID))
        del self.matDict[MID]
    def printSummary(self):
        """Prints summary of all Materials in MaterialLib

        A method used to print out tabulated summary of all of the materials
        held within the material library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the materials.

        """
        if len(self.matDict)==0:
            print('The material library is currently empty.\n')
        else:
            print('The materials are:')
            for MID, mat in self.matDict.items():
                mat.printSummary()
    def writeToFile(self):
        """Prints summary of all Materials in MaterialLib

        A method used to print out tabulated summary of all of the materials
        held within the material library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the materials.

        """
        print_statement = []
        if len(self.matDict)==0:
            print('The material library is currently empty.\n')
        else:
            for MID, mat in self.matDict.items():
                print_statement += [mat.writeToFile()]
        return print_statement
class Ply:
    """Creates a CLT ply object.

    A class inspired by CLT, this class can be used to generate laminates
    to be used for CLT or cross-sectional analysis. It is likely that ply
    objects won't be created individually and then assembeled into a lamiante.
    More likely is that the plies will be generated within the laminate object.
    It should also be noted that it is assumed that the materials used are
    effectively at most transversely isotropic.

    :Attributes:

    - `E1 (float)`: Stiffness in the fiber direction.
    - `E2 (float)`: Stiffness transverse to the fiber direction.
    - `nu_12 (float)`: In plane poisson ratio.
    - `G_12 (float)`: In plane shear modulus.
    - `t (float)`: Thickness of the ply.
    - `Qbar (1x6 np.array[float])`: The terms in the rotated, reduced stiffness
        matrix. Ordering is as follows: [Q11,Q12,Q16,Q22,Q26,Q66]
    - `MID (int)`: An integer refrencing the material ID used for the
        constitutive relations.
    - `th (float)`: The angle about which the fibers are rotated in the plane
        in degrees.

    :Methods:

    - `genQ`: Given the in-plane stiffnesses used by the material of the ply,
        the method calculates the terms of ther reduced stiffness matrix.
    - `printSummary`: This prints out a summary of the object, including
        thickness, referenced MID and in plane angle orientation theta in
        degrees.

    """
    def __init__(self,Material,th):
        """Initializes the ply.

        This method initializes information about the ply such as in-plane
        stiffness repsonse.

        :Args:

        - `Material (obj)`: A material object, most likely coming from a
            material library.
        - `th (float)`: The angle about which the fibers are rotated in the
            plane in degrees.

        :Returns:

        - None

        """
        self.type='Ply'
        self.E1 = Material.E1
        self.E2 = Material.E2
        self.nu_12 = Material.nu_12
        self.G_12 = Material.G_12
        self.t = Material.t
        self.Q = self.genQ(self.E1,self.E2,self.nu_12,self.G_12)
        self.Qbar = self.rotRedStiffMat(self.Q,th)
        self.QbarMat = np.array([[self.Qbar[0],self.Qbar[1],self.Qbar[2]],\
                                 [self.Qbar[1],self.Qbar[3],self.Qbar[4]],\
                                 [self.Qbar[2],self.Qbar[4],self.Qbar[5]]])
        self.MID = Material.MID
        self.th = th
    def genQ(self,E1,E2,nu12,G12):
        """A method for calculating the reduced compliance of the ply.

        Intended primarily as a private method but left public, this method,
        for those unfarmiliar with CLT, calculates the terms in the reduced stiffness
        matrix given the in plane ply stiffnesses. It can be thus inferred that
        this requires the assumption of plane stres. This method is primarily
        used during the ply instantiation.

        :Args:

        - `E1 (float)`: The fiber direction stiffness.
        - `E2 (float)`: The stiffness transverse to the fibers.
        - `nu12 (float)`: The in-plane poisson ratio.
        - `G12 (float)`: The in-plane shear stiffness.

        :Returns:

        - `(1x4 np.array[float])`: The terms used in the reduced stiffness
            matrix. The ordering is: [Q11,Q12,Q22,Q66].

        """
        # Calculate the other in-plane poisson ratio.
        nu21 = nu12*E2/E1
        return [E1/(1-nu12*nu21),nu12*E2/(1-nu12*nu21),E2/(1-nu12*nu21),G12]
    def rotRedStiffMat(self,Q,th):
        """Calculate terms in the rotated, reduced stiffness matrix.

        Intended primarily as a private method but left public, this method,
        this method is used to rotate the plies reduced compliance matrix to
        the local laminate coordinate system.

        :Args:

        - `Q (1x4 np.array[float])`: The reduced compliance array containing
            [Q11,Q12,Q22,Q66]
        - `th(float)`: The angle the fibers are to be rotated in plane of the
            laminate.

        :Returns:

        - `(1x6 np.array[float])`: The reduced and rotated stiffness matrix terms
            for the ply. The ordering is: [Q11, Q12, Q16, Q22, Q26, Q66].

        """
        # Convert the angle to radians
        th = np.deg2rad(th)
        # Pre-calculate cosine of theta
        m = np.cos(th)
        # Pre-calculate sine of theta
        n = np.sin(th)
        # Compute the rotated, reduced stiffness matrix terms:
        Q11bar = Q[0]*m**4+2*(Q[1]+2*Q[3])*n**2*m**2+Q[2]*n**4
        Q12bar = (Q[0]+Q[2]-4*Q[3])*n**2*m**2+Q[1]*(n**4+m**4)
        Q16bar = (Q[0]-Q[1]-2*Q[3])*n*m**3+(Q[1]-Q[2]+2*Q[3])*n**3*m
        Q22bar = Q[0]*n**4+2*(Q[1]+2*Q[3])*n**2*m**2+Q[2]*m**4
        Q26bar = (Q[0]-Q[1]-2*Q[3])*n**3*m+(Q[1]-Q[2]+2*Q[3])*n*m**3
        Q66bar = (Q[0]+Q[2]-2*Q[1]-2*Q[3])*n**2*m**2+Q[3]*(n**4+m**4)
        return [Q11bar,Q12bar,Q16bar,Q22bar,Q26bar,Q66bar]
    def printSummary(self):
        """Prints a summary of the ply object.

        A method for printing a summary of the ply properties, such as
        the material ID, fiber orientation and ply thickness.

        :Args:

        - None

        :Returns:

        - `(str)`: Printed tabulated summary of the ply.

        """
        headers = ['MID','Theta, degrees','Thickness']
        print(tabulate(([[self.MID,self.th, self.t]]),headers))

class Laminate:
    """Creates a CLT laminate object.

    This class has two main uses. It can either be used for CLT analysis, or it
    can be used to build up a 2D mesh for a descretized cross-section.

    :Attributes:

    - `mesh (NxM np.array[int])`: This 2D array holds NIDs and is used
        to represent how nodes are organized in the 2D cross-section of
        the laminate.
    - `xmesh (NxM np.array[int])`: This 2D array holds the rigid x-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `ymesh (NxM np.array[int])`: This 2D array holds the rigid y-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `zmesh (NxM np.array[int])`: This 2D array holds the rigid z-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `H (float)`: The total laminate thickness.
    - `rho_A (float)`: The laminate area density.
    - `plies (1xN array[obj])`: Contains an array of ply objects used to
        construct the laminate.
    - `t (1xN array[float])`: An array containing all of the ply thicknesses.
    - `ABD (6x6 np.array[float])`: The CLT 6x6 matrix relating in-plane strains
        and curvatures to in-plane force and moment resultants.
    - `abd (6x6 np.array[float])`: The CLT 6x6 matrix relating in-plane forces
        and moments resultants to in-plane strains and curvatures.
    - `z (1xN array[float])`: The z locations of laminate starting and ending
        points. This system always starts at -H/2 and goes to H/2
    - `equivMat (obj)`: This is orthotropic material object which exhibits
        similar in-plane stiffnesses.
    - `forceRes (1x6 np.array[float])`: The applied or resulting force and
        moment resultants generated during CLT analysis.
    - `globalStrain (1x6 np.array[float])`:  The applied or resulting strain
        and curvatures generated during CLT analysis.


    :Methods:

    - `printSummary`: This method prints out defining attributes of the
        laminate, such as the ABD matrix and layup schedule.

    """
    def __init__(self,LAMID,n_i_tmp,m_i_tmp,matLib,**kwargs):
        """Initializes the Laminate object

        The way the laminate initialization works is you pass in two-three
        arrays and a material library. The first array contains information
        about how many plies you want to stack, the second array determines
        what material should be used for those plies, and the third array
        determines at what angle those plies lie. The class was developed this
        way as a means to fascilitate laminate optimization by quickly changing
        the number of plies at a given orientation and using a given material.

        :Args:

        - `n_i_tmp (1xN array[int])`: An array containing the number of plies
            using a material at a particular orientation such as:
            (theta=0,theta=45...)
        - `m_i_tmp (1xN array[int])`: An array containing the material to be
            used for the corresponding number of plies in the n_i_tmp array
        - `matLib (obj)`: The material library holding different material
            objects.
        - `sym (bool)`: Whether the laminate is symetric. (False by default)
        - `th (1xN array[float])`: An array containing the orientation at which
            the fibers are positioned within the laminate.

        :Returns:

        - None

        .. Note:: If you wanted to create a [0_2/45_2/90_2/-45_2]_s laminate of the
            same material, you could call laminate as:

            lam = Laminate([2,2,2,2],[1,1,1,1],matLib,sym=True)

            Or:

            lam = Laminate([2,2,2,2],[1,1,1,1],matLib,sym=True,th=[0,45,90,-45])

            Both of these statements are equivalent. If no theta array is
            provided and n_i_tmp is not equal to 4, then Laminate will default
            your fibers to all be running in the 0 degree orientation.

        """
        # Initialize attribute handles for latter X-Section meshing assignment
        self.type='Laminate'
        self.LAMID = LAMID
        # Assign symetric laminate parameter
        sym = kwargs.pop('sym',False)
        # Verify that n_i_tmp and m_i_tmp are the same length
        if not len(n_i_tmp)==len(m_i_tmp):
            raise ValueError('n_i_tmp and m_i_tmp must be the same length.\n')
        # If no th provided, assign and n_i_tmp is a 4 length array, make
        # th=[0,45,90,-45].
        if len(n_i_tmp)==4:
            th = kwargs.pop('th',[0,45,90,-45])
        # Otherwise make th 0 for the length of n_i_tmp
        else:
            th = kwargs.pop('th',[0]*len(n_i_tmp))
        # If the laminate is symmetric, reflect n_i_tmp and m_i_tmp
        self.sym = sym
        if sym:
            n_i_tmp = n_i_tmp+n_i_tmp[::-1]
            m_i_tmp = m_i_tmp+m_i_tmp[::-1]
            th = th+th[::-1]
        self.ni = n_i_tmp
        self.mi = m_i_tmp
        self.thi = []
        #Calculate the total laminate thickness and area density:
        H = 0.
        rho_A = 0.
        for i in range(0,len(th)):
            tmp_mat = matLib.matDict[m_i_tmp[i]]
            H += tmp_mat.t*n_i_tmp[i]
            rho_A += tmp_mat.t*n_i_tmp[i]*tmp_mat.rho
        # Assign the total laminate thickness H
        self.H = H
        # Assign the laminate area density
        self.rho_A = rho_A
        z = np.zeros(sum(n_i_tmp)+1)
        z[0] = -self.H/2.
        # Initialize ABD Matrix, thermal and moisture unit forces, and the area
        # density.
        ABD = np.zeros((6,6))
        #TODO: Add thermal and moisture support
        # NM_T = np.zeros((6,1))
        # NM_B = np.zeros((6,1))
        # Counter for ease of programming. Could go back and fix:
        c = 0
        # Initialize plies object array
        self.plies = []
        # Initialize thickness float array
        self.t = []
        # For all plies
        for i in range(0,len(th)):
            # Select the temporary material for the ith set of plies
            tmp_mat = matLib.matDict[m_i_tmp[i]]
            # For the number of times the ply material and orientation are
            # repeated
            for j in range(0,n_i_tmp[i]):
                # Create a new ply
                tmp_ply = Ply(tmp_mat,th[i])
                # Add the new ply to the array of plies held by the laminate
                self.plies+=[tmp_ply]
                # Update z-position array
                z[c+1] = z[c]+tmp_mat.t
                # Add terms to the ABD matrix for laminate reponse
                ABD[0:3,0:3] += tmp_ply.QbarMat*(z[c+1]-z[c])
                ABD[0:3,3:6] += (1./2.)*tmp_ply.QbarMat*(z[c+1]**2-z[c]**2)
                ABD[3:6,0:3] += (1./2.)*tmp_ply.QbarMat*(z[c+1]**2-z[c]**2)
                ABD[3:6,3:6] += (1./3.)*tmp_ply.QbarMat*(z[c+1]**3-z[c]**3)
                c += 1
                # Create array of all laminate thicknesses
                self.t += [tmp_mat.t]
                self.thi += [tmp_ply.th]
        # Assign the ABD matrix to the object
        self.ABD = ABD
        # Assign the inverse of the ABD matrix to the object
        self.abd = np.linalg.inv(ABD)
        # Assign the coordinates for the laminate (demarking the interfaces
        # between plies within the laminate) to the object
        self.z = z
        # Generate equivalent in-plane engineering properties:
        Ex = (ABD[0,0]*ABD[1,1]-ABD[0,1]**2)/(ABD[1,1]*H)
        Ey = (ABD[0,0]*ABD[1,1]-ABD[0,1]**2)/(ABD[0,0]*H)
        G_xy = ABD[2,2]/H
        nu_xy = ABD[0,1]/ABD[1,1]
        # nuyx = ABD[0,1]/ABD[0,0]
        mat_constants = [Ex, Ey, nu_xy, 0., G_xy, rho_A,1.,1.,1.,1.,1.]
        # Create an equivalent material object for the laminate
        self.equivMat = Material(101, 'Equiv Lam Mat', 'TISO', mat_constants,mat_t=H)
        # Initialize Miscelanoes Parameters:
        self.forceRes = np.zeros(6)
        self.globalStrain = np.zeros(6)
    def printSummary(self,**kwargs):
        """Prints a summary of information about the laminate.

        This method can print both the ABD matrix and ply information schedule
        of the laminate.

        :Args:

        - `ABD (bool)`: This optional argument asks whether the ABD matrix
            should be printed.
        - `decimals (int)`: Should the ABD matrix be printed, python should
            print up to this many digits after the decimal point.
        - `plies (bool)`: This optional argument asks whether the ply schedule
            for the laminate should be printed.

        :Returns:

        - None

        """
        ABD = kwargs.pop('ABD',True)
        decimals = kwargs.pop('decimals',4)
        plies = kwargs.pop('plies',True)
        if ABD:
            print('ABD Matrix:')
            print(tabulate(np.around(self.ABD,decimals=decimals),tablefmt="fancy_grid"))
        if plies:
            for ply in self.plies:
                ply.printSummary()
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        lam_card = 'LAMINATE,{},{},{},{},{}'.format(self.LAMID,self.NiLSID,\
                            self.MiLSID,self.THiLSID,self.sym)
        Ni_card = 'LIST,{},INT'.format(self.NiLSID)
        for n in self.ni:
            Ni_card += ','+str(n)
        Mi_card = 'LIST,{},INT'.format(self.MiLSID)
        for m in self.mi:
            Mi_card += ','+str(m)
        THi_card = 'LIST,{},FLOAT'.format(self.THiLSID)
        for th in self.thi:
            THi_card += ','+str(th)
        return [lam_card,Ni_card,Mi_card,THi_card]
class LaminateLibrary:

    def __init__(self):
        self.type='LaminateLibrary'
        self.lamDict = {}
    def add(self,LAMID, n_i, m_i, matLib,**kwargs):
        overwrite = kwargs.pop('overwrite',False)
        if LAMID in self.lamDict.keys() and not overwrite:
            raise Exception('You may not overwrite a library Laminate'+\
                ' entry without adding the optional argument overwrite=True')
        # Save material
        self.lamDict[LAMID] = Laminate(LAMID,n_i,m_i,matLib,**kwargs)
    def get(self,LAMID):
        if not LAMID in self.lamDict.keys():
            raise KeyError('The LAMID provided is not linked with any laminates '+
                'within the supplied laminate library.')
        return self.lamDict[LAMID]
    def getIDs(self):
        return self.lamDict.keys()
    def delete(self,LAMID):
        if not LAMID in self.lamDict.keys():
            raise KeyError('The LAMID provided is not linked with any laminates'+
                'within the supplied laminate library.')
        del self.lamDict[LAMID]
    def printSummary(self):
        if len(self.lamDict)==0:
            print('The laminate library is currently empty.\n')
        else:
            print('The laminates are:')
            for LAMID, lam in self.lamDict.items():
                lam.printSummary()
    def writeToFile(self):
        """Prints summary of all Laminates in LaminateLib

        A method used to print out tabulated summary of all of the materials
        held within the material library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the materials.

        """
        print_statement = []
        if len(self.lamDict)==0:
            print('The laminate library is currently empty.\n')
        else:
            for LAMID, lam in self.lamDict.items():
                print_statement += lam.writeToFile()
        return print_statement


class XELEMENT:
    """ Creates a linear, 2D 8 node quadrilateral element object.

    The main purpose of this class is to assist in the cross-sectional
    analysis of a beam.

    :Attributes:

    - `type (str)`: A string designating it a XQUAD8 element.
    - `nd (int)`: The number of degrees of freedome 3*number nodes
    - `th (1x3 Array[float])`: The euler angle rotations that define the element
        material direction
    - `Rsiginv (3x3 Array[float])`: The transformation matrix to convert global
        stresses to local element stresses
    - `Repsinv (3x3 Array[float])`: The transformation matrix to convert global
        strains to local element strains
    - `CSYS (obj)`: An opengl object of the element material CSYS
    - `NIDs (1x4 Array[int])`: An array of ints for the element node IDs
    - `nodes (1x4 Array[obj])`: An array of node objects
    - `EID (int)`: Element ID
    - `rho (float)`: Element mass volume density
    - `MID (int)`: Material ID of element
    - `material (obj)`: Material object referenced by the element
    - `mass (float)`: Element mass
    - `xis_recov (1x9 array[float])`: List of xi master element coordinates to
        be used for element data recovery
    - `etas_recov (1x9 array[float])`: List of eta master element coordinates to
        be used for element data recovery
    - `xis_int (1x3 array[float])`: List of xi master element coordinates to
        be used for element numerical integration
    - `etas_int (1x3 array[float])`: List of eta master element coordinates to
        be used for element numerical integration
    - `w_xis_int (1x3 array[float])`: List of xi master element weights to
        be used for element numerical integration
    - `w_etas_int (1x3 array[float])`: List of eta master element weights to
        be used for element numerical integration
    - `f2strn (6*nx6 array[float])`: A 2D array that when multiplied by a 6x1
        Force vector, returns a 6*nx1 column vector of element strains where
        n is the number of strain sample points. Note that the order of the
        strains is: eps=[eps_xx,eps_yy,eps_xy,eps_xz,eps_yz,eps_zz]
    - `f2sig (6*nx6 array[float])`: A 2D array that when multiplied by a 6x1
        Force vector, returns a 6*nx1 column vector of element stresses where
        n is the number of stress sample points. Note that the order of the
        stresses is: eps=[sig_xx,sig_yy,sig_xy,sig_xz,sig_yz,sig_zz]
    - `Q (6x6 array[float])`: The 6x6 constitutive relationship for the element
    - `xs (1xn array[float])`: A 1xn array of x coordinates for the element
        where n is the number of nodes
    - `ys (1xn array[float])`: A 1xn array of y coordinates for the element
        where n is the number of nodes
    - `U (ndx1 array[float])`: This column vector contains the elements
        3 DOF (x-y-z) displacements in the local xsect CSYS due to cross-
        section warping effects.
    - `Eps (6xn array[float])`: A matrix containing the 3D strain state
        within the element where n is the number of strain sample points
    - `Sig (6xn array[float])`: A matrix containing the 3D stress state
        within the element where n is the number of stress sample points

    :Methods:

    - `x`: Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
    - `y`: Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
    - `J`: Calculates the jacobian of the element provided the desired master
        coordinates eta and xi.
    - `Jdet_inv: Calculates the inverse jacobian and it's determinent and
        returns both as a 3x3 array[float] and float
    - `N`: Calculates and returns the shape function weighting matrix provided
        the desired master coordinates
    - `dNdxi`: Calculates and returns the derivative of the shape function
        weighting matrix provided the desired master coordinates
    - `dNdeta`: Calculates and returns thederivative of the shape function
        weighting matrix provided the desired master coordinates
    - `initializeElement`: A function to be run before using the element in
        cross-sectional analysis. Since the element can be translated within
        the cross-section plane to improve iterative matrix solution, this can
        only be run once all elements have been added to a cross-section and
        translated
    - `resetResults`: Initializes the displacement (U), strain (Eps), and
        stress (Sig) attributes of the element.
    - `calcStrain`: Provided a force vector F, this method computes the element
        strain in the global coordinate system
    - `calcStress`: Provided a force vector F, this method computes the element
        stress in the local coordinate system
    - `calcDisp`: Provided a force vector F, this method computes the element
        nodal displacements due to warping in the local cross-sectional CSYS
    - `getDeformed`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element warped
        displacements in the local xsect CSYS.
    - `getStressState`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element stress at four
        points. The 3D stress state is processed to return the Von-Mises
        or Maximum Principal stress state.
    - `printSummary`: Prints out a tabulated form of the element ID, as well
        as the node ID's referenced by the element.

    """
    def __init__(self,EID,nodes,material,etype,nd,**kwargs):
        """ Initializes the element.

        :Args:

        - `EID (int)`: An integer identifier for the CQUADX element.
        - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
            used to create the element.
        - `MID (int)`: An integer refrencing the material ID used for the
            constitutive relations.
        - `matLib (obj)`: A material library object containing a dictionary
            with the material corresponding to the provided MID.
        - `xsect (bool)`: A boolean to determine whether this quad element is
            to be used for cross-sectional analysis. Defualt value is True.
        - `th (1x3 Array[float])`: Array containing the Euler-angles expressing
            how the element constitutive relations should be rotated from
            the material fiber frame to the global CSYS. In degrees.

        :Returns:

        - None

        .. Note:: The reference coordinate system for cross-sectional analysis is a
        local coordinate system in which the x and y axes are planer with the
        element, and the z-axis is perpendicular to the plane of the element.

        """
        # Initialize Euler-angles for material orientation in the xsect CSYS
        self.type = etype
        self.nd = nd
        th = kwargs.pop('th', [0.,0.,0.])
        self.th = th
        self.XID = None

        Rxsectsiginv, Rxsectepsinv = genCompRy(90)
        Rxsiginv, Rxepsinv = genCompRx(-th[0])
        Rysiginv, Ryepsinv = genCompRy(-th[1])
        Rzsiginv, Rzepsinv = genCompRz(-th[2])
        self.Rsiginv = np.dot(Rxsectsiginv,np.dot(Rxsiginv,np.dot(Rysiginv,Rzsiginv)))
        self.Repsinv = np.dot(Rxsectepsinv,np.dot(Rxepsinv,np.dot(Ryepsinv,Rzepsinv)))

        CSYS = gl.GLAxisItem()
        CSYS.rotate(-90,0.,1.,0.)
        CSYS.rotate(th[0],1.,0.,0.)
        CSYS.rotate(th[1],0.,1.,0.)
        CSYS.rotate(th[2],0.,0.,1.)
        self.CSYS = CSYS
        # Error checking on EID input
        if type(EID) is int:
            self.EID = EID
        else:
            raise TypeError('The element ID must be an integer')
        if not len(nodes) == nd/3:
            raise ValueError('A {} element requires {} nodes, {} were supplied \
                             in the nodes array'.format(etype,int(nd/3),len(nodes)))
        nids = []
        for node in nodes:
            node.addEID(EID)
            nids+= [node.NID]
        if not len(np.unique(nids))==nd/3:
            raise ValueError('The node objects used to create this {} \
                             share at least 1 NID. Make sure that no repeated\
                             node objects were used.'.format(etype))
        # Initialize the warping displacement, strain and stress results
        self.resetResults()
        # Populate the NIDs array with the IDs of the nodes used by the element
        self.NIDs = nids
        self.nodes = nodes

        self.rho = material.rho
        # Store the MID
        self.MID = material.MID
        self.material = material
        # Initialize the mass per unit length (or thickness) of the element
        self.mass = 0

        if etype=='XQUAD4':
            self.xis_recov = [-1,1,1,-1]
            self.etas_recov = [-1,-1,1,1]
            self.etas_int = np.array([-1,1])*np.sqrt(3)/3
            self.xis_int = np.array([-1,1])*np.sqrt(3)/3
            self.w_etas_int = np.array([1,1])
            self.w_xis_int = np.array([1,1])
            self.quadFactor = 1.
        elif etype=='XQUAD6':
            self.xis_recov = [-1,0,1,1,0,-1]
            self.etas_recov = [-1,-1,-1,1,1,1]
            self.etas_int = np.array([-1,1])*np.sqrt(3)/3
            self.xis_int = np.array([-1,0,1])*np.sqrt(3./5)
            self.w_etas_int = np.array([1,1])
            self.w_xis_int = np.array([5./9,8./9,5./9])
            self.quadFactor = 1.
        elif etype=='XQUAD8':
            self.xis_recov = [-1,0,1,1,1,0,-1,-1,0]
            self.etas_recov = [-1,-1,-1,0,1,1,1,0,0]
            self.etas_int = np.array([-1,0,1])*np.sqrt(3./5)
            self.xis_int = np.array([-1,0,1])*np.sqrt(3./5)
            self.w_etas_int = np.array([5./9,8./9,5./9])
            self.w_xis_int = np.array([5./9,8./9,5./9])
            self.quadFactor = 1.
        elif etype=='XQUAD9':
            self.xis_recov = [-1,0,1]*3
            self.etas_recov = [-1,]*3+[0,]*3+[1,]*3
            self.etas_int = np.array([-1,0,1])*np.sqrt(3./5)
            self.xis_int = np.array([-1,0,1])*np.sqrt(3./5)
            self.w_etas_int = np.array([5./9,8./9,5./9])
            self.w_xis_int = np.array([5./9,8./9,5./9])
            self.quadFactor = 1.
        elif etype=='XTRIA3':
            self.xis_recov = [0,0,1]
            self.etas_recov = [0,1,0]
            self.etas_int = np.array([1./3])
            self.xis_int = np.array([1./3])
            self.w_etas_int = np.array([1.])
            self.w_xis_int = np.array([1.])
            self.quadFactor = 0.5
        elif etype=='XTRIA6':
            self.xis_recov = [0,1./2,1,0,1./2,0]
            self.etas_recov = [0,0,0,1./2,1./2,1]
            self.etas_int = np.array([1./2,0,1./2])
            self.xis_int = np.array([0,1./2,1./2])
            self.w_etas_int = np.array([1./3,1./3,1./3])
            self.w_xis_int = np.array([1./3,1./3,1./3])
            self.quadFactor = 0.5

        # Initialize strain vectors
        self.f2strn = None
        # Initialize stress vectors
        self.f2sig = None
        # Rotate the materials compliance matrix as necessary:
        Selem = transformCompl(np.copy(material.Smat),th,xsect=True)
        # Reorder Selem for cross-sectional analysis:
        # Initialize empty compliance matrix
        Sxsect = np.zeros((6,6))
        # Initialize reorganization key
        shuff = [0,1,5,4,3,2]
        for i in range(0,6):
            for j in range(0,6):
                Sxsect[shuff[i],shuff[j]] = Selem[i,j]
        # Store the re-ordered material stiffness matrix:
        self.Q = np.linalg.inv(Sxsect)

        # Generate X and Y coordinates of the nodes
        xs = np.zeros(int(nd/3))
        ys = np.zeros(int(nd/3))
        for i in range(0,int(nd/3)):
            tempxyz = nodes[i].x
            xs[i] = tempxyz[0]
            ys[i] = tempxyz[1]
        # Save for ease of strain calculation on strain recovery
        self.xs = xs
        self.ys = ys

    def setXID(self,XID):
        self.XID = XID

    def x(self,eta,xi):
        """Calculate the x-coordinate within the element.

        Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `x (float)`: The x-coordinate within the element.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        return np.dot(self.N(eta,xi),self.xs)

    def y(self,eta,xi):
        """Calculate the y-coordinate within the element.

        Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `y (float)': The y-coordinate within the element.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        return np.dot(self.N(eta,xi),self.ys)

    def Z(self,eta,xi):
        """Calculates transformation matrix relating stress to force-moments.

        Intended primarily as a private method but left public, this method
        calculates the transformation matrix that converts stresses to force
        and moment resultants.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Z (3x6 np.array[float])`: The stress-resutlant transformation array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        return np.array([[1.,0,0,0,0,-self.y(eta,xi)],\
                         [0,1.,0,0,0,self.x(eta,xi)],\
                         [0,0,1.,self.y(eta,xi),-self.x(eta,xi),0]])

    def initializeElement(self):
        self.mass = 0.
        nodes = self.nodes
        nd = self.nd
        CSYS = self.CSYS
        # Determine the direction of the element unit normal
        sign = self.getNormalSign()
        # Initialize Matricies for later use in xsect equilibrium solution:
        self.Ae = np.zeros((6,6))
        self.Re = np.zeros((nd,6))
        self.Ee = np.zeros((nd,nd))
        self.Ce = np.zeros((nd,nd))
        self.Le = np.zeros((nd,6))
        self.Me = np.zeros((nd,nd))
        # Generate X and Y coordinates of the nodes
        xs = np.zeros(int(nd/3))
        ys = np.zeros(int(nd/3))
        for i in range(0,int(nd/3)):
            tempxyz = nodes[i].x
            xs[i] = tempxyz[0]
            ys[i] = tempxyz[1]
        # Save for ease of strain calculation on strain recovery
        self.xs = xs
        self.ys = ys
        # Initialize coordinates for Guass Quadrature Integration
        etas = self.etas_int
        xis = self.xis_int
        w_etas = self.w_etas_int
        w_xis = self.w_xis_int
        S = np.zeros((6,3));S[3,0]=1;S[4,1]=1;S[5,2]=1
        # Evaluate/sum the cross-section matricies at the Guass points
        for k in range(0,np.size(xis)):
            for l in range(0,np.size(etas)):
                #Get Z Matrix
                Zmat = self.Z(etas[l],xis[k])
                #Get BN Matricies
                #Jmat = J(etas[l],xis[k])
                #Get determinant of the Jacobian Matrix
                #Jdet = abs(np.linalg.det(Jmat))
                #Jmatinv = np.linalg.inv(Jmat)
                Jdet, Jmatinv = self.Jdet_inv(etas[l],xis[k])
                Bxi = np.zeros((6,3))
                Beta = np.zeros((6,3))
                Bxi[0,0] = Bxi[2,1] = Bxi[3,2] = Jmatinv[0,0]
                Bxi[1,1] = Bxi[2,0] = Bxi[4,2] = Jmatinv[1,0]
                Beta[0,0] = Beta[2,1] = Beta[3,2] = Jmatinv[0,1]
                Beta[1,1] = Beta[2,0] = Beta[4,2] = Jmatinv[1,1]
                BN = np.dot(Bxi,self.dNdxi(etas[l],xis[k])) +\
                                       np.dot(Beta,self.dNdeta(etas[l],xis[k]))

                #Get a few last minute matricies
                SZ = np.dot(S,Zmat)
                Nmat = self.Nmat(etas[l],xis[k])
                SN = np.dot(S,Nmat)

                # Calculate the mass per unit length of the element
                self.mass += self.rho*Jdet*w_etas[l]*w_xis[k]*self.quadFactor

                #Add to Ae Matrix
                self.Ae += np.dot(SZ.T,np.dot(self.Q,SZ))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
                #Add to Re Matrix
                self.Re += np.dot(BN.T,np.dot(self.Q,SZ))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
                #Add to Ee Matrix
                self.Ee += np.dot(BN.T,np.dot(self.Q,BN))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
                #Add to Ce Matrix
                self.Ce += np.dot(BN.T,np.dot(self.Q,SN))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
                #Add to Le Matrix
                self.Le += np.dot(SN.T,np.dot(self.Q,SZ))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
                #Add to Me Matrix
                self.Me += np.dot(SN.T,np.dot(self.Q,SN))*Jdet*w_etas[l]*w_xis[k]*self.quadFactor
        self.Aeflat = flatten(self.Ae,6,6)
        self.Reflat = flatten(self.Re,nd,6)
        self.Eeflat = flatten(self.Ee,nd,nd)
        self.Ceflat = flatten(self.Ce,nd,nd)
        self.Leflat = flatten(self.Le,nd,6)
        self.Meflat = flatten(self.Me,nd,nd)
        CSYS.translate(self.x(0,0),self.y(0,0),0)
        xmag = self.x(1,0)-self.x(0,0)
        ymag = self.y(1,0)-self.y(0,0)
        mag = np.sqrt(xmag**2+ymag**2)
        CSYS.setSize(mag,mag,mag)
        self.CSYS = CSYS
        self.normal_vec = np.array(((self.x(0,0),self.y(0,0), 0.0), (self.x(0,0), self.y(0,0), sign*mag)))
        self.normal = gl.GLLinePlotItem(pos=self.normal_vec,color=(1.0, 0.0, 0.0, 1.0),antialias=True)

    def resetResults(self):
        """Resets stress, strain and warping displacement results.

        Method is mainly intended to prevent results for one analysis or
        sampling location in the matrix to effect the results in another.

        :Args:

        - None

        :Returns:

        - None

        """
        # Initialize array for element warping displacement results
        nd = self.nd
        self.U = {}
        self.U[-1] = np.zeros((nd,1))
        # Initialize strain vectors
        self.Eps = {}
        self.Eps[-1] = np.zeros((2*nd,1))
        # Initialize stress vectors
        self.Sig = {}
        self.Sig[-1] = np.zeros((2*nd,1))
    def calcStrain(self,LCID,F):
        self.Eps[LCID] = np.dot(self.f2strn,F)
    def calcStress(self,LCID,F):
        self.Sig[LCID] = np.dot(self.f2sig,F)
    def calcDisp(self,LCID,F):
        self.U[LCID] = np.dot(self.f2disp,F)

    def getContour(self,LCIDs,crit='VonMis',centroid=False):
        """Returns the stress state of the element.

        Provided an analysis has been conducted, this method
        returns a 2x2 np.array[float] containing the element the 3D stress
        state at the four guass points by default.*

        :Args:

        - `crit (str)`: Determines what criteria is used to evaluate the 3D
            stress state at the sample points within the element. By
            default the Von Mises stress is returned. Currently supported
            options include: Von Mises ('VonMis'), maximum principle stress
            ('MaxPrin'), the minimum principle stress ('MinPrin'), and the
            local cross-section stress states 'sig_xx' where the subindeces can
            go from 1-3. The keyword 'none' is also an option.

        :Returns:

        - `sigData (2x2 np.array[float])`: The stress state evaluated at four
            points within the CQUADX element.

        .. Note:: The XSect method calcWarpEffects is what determines where strain
        and stresses are sampled. By default it samples this information at the
        Guass points where the stress/strain will be most accurate.

        """
        data_env = []
        for LCID in LCIDs:
            n = len(self.xis_recov)
            if not LCID in self.Sig.keys():
                LCID_stress=-1
                print('User requested stress for cross-section element {}, '\
                  'however stress for that load case has not been computed.'.format(self.EID))
            else:
                LCID_stress=LCID
            if not LCID in self.Eps.keys():
                LCID_strain=-1
                print('User requested strain for cross-section element {}, '\
                  'however strain for that load case has not been computed.'.format(self.EID))
            else:
                LCID_strain=LCID

            sigState = self.Sig[LCID_stress]
            epsState = self.Eps[LCID_strain]
            # Initialize the blank stress 2x2 array
            data = []
            # For all four points
                # Determine what criteria is to be used to evaluate the stress
                # State
            if crit=='Von Mises Stress':
                for i in range(0,n):
                    data += [np.sqrt(0.5*((sigState[6*i+0,0]-sigState[6*i+1,0])**2+\
                        (sigState[6*i+1,0]-sigState[6*i+5,0])**2+\
                        (sigState[6*i+5,0]-sigState[6*i+0,0])**2+\
                        6*(sigState[6*i+2,0]**2+sigState[6*i+3,0]**2+sigState[6*i+4,0]**2)))]
            elif crit=='Maximum Principle Stress':
                for i in range(0,n):
                    tmpSigTens = np.array([[sigState[6*i+0,0],sigState[6*i+2,0],sigState[6*i+3,0]],\
                        [sigState[6*i+2,0],sigState[6*i+1,0],sigState[6*i+4,0]],\
                        [sigState[6*i+3,0],sigState[6*i+4,0],sigState[6*i+5,0]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    data += [max(eigs)]
            elif crit=='Minimum Principle Stress':
                for i in range(0,n):
                    tmpSigTens = np.array([[sigState[6*i+0,0],sigState[6*i+2,0],sigState[6*i+3,0]],\
                        [sigState[6*i+2,0],sigState[6*i+1,0],sigState[6*i+4,0]],\
                        [sigState[6*i+3,0],sigState[6*i+4,0],sigState[6*i+5,0]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    data += [min(eigs)]
            elif crit=='Sigma_xx':
                for i in range(0,n):
                    data += [sigState[6*i+0,0]]
            elif crit=='Sigma_yy':
                for i in range(0,n):
                    data += [sigState[6*i+1,0]]
            elif crit=='Sigma_xy':
                for i in range(0,n):
                    data += [sigState[6*i+2,0]]
            elif crit=='Sigma_xz':
                for i in range(0,n):
                    data += [sigState[6*i+3,0]]
            elif crit=='Sigma_yz':
                for i in range(0,n):
                    data += [sigState[6*i+4,0]]
            elif crit=='Sigma_zz':
                for i in range(0,n):
                    data += [sigState[6*i+5,0]]
            elif crit=='Sigma_11':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    #print(fiberSigVec)
                    data += [fiberSigVec[0]]
            elif crit=='Sigma_22':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    data += [fiberSigVec[1]]
            elif crit=='Sigma_12':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    data += [fiberSigVec[5]]
            elif crit=='Sigma_13':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    data += [fiberSigVec[4]]
            elif crit=='Sigma_23':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    data += [fiberSigVec[3]]
            elif crit=='Sigma_33':
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])
                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    data += [fiberSigVec[2]]
            elif crit=='Eps_11':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[0]]
            elif crit=='Eps_22':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[1]]
            elif crit=='Eps_12':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[5]]
            elif crit=='Eps_13':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[4]]
            elif crit=='Eps_23':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[3]]
            elif crit=='Eps_33':
                for i in range(0,n):
                    tmpEpsVec = np.array([epsState[6*i+0,0],epsState[6*i+1,0],\
                                          epsState[6*i+5,0],epsState[6*i+4,0],\
                                          epsState[6*i+3,0],epsState[6*i+2,0]])
                    fiberEpsVec = np.dot(self.Repsinv,tmpEpsVec)
                    data += [fiberEpsVec[2]]
            elif crit=='Maximum Principle Strain':
                for i in range(0,n):
                    tmpEps = np.array([[epsState[6*i+0,0],epsState[6*i+2,0],epsState[6*i+3,0]],
                                       [epsState[6*i+2,0],epsState[6*i+1,0],epsState[6*i+4,0]],
                                       [epsState[6*i+3,0],epsState[6*i+4,0],epsState[6*i+5,0]]])
                    eigs,trash = np.linalg.eig(tmpEps)
                    data += [max(eigs)]
            elif crit=='Minimum Principle Strain':
                for i in range(0,n):
                    tmpEps = np.array([[epsState[6*i+0,0],epsState[6*i+2,0],epsState[6*i+3,0]],
                                       [epsState[6*i+2,0],epsState[6*i+1,0],epsState[6*i+4,0]],
                                       [epsState[6*i+3,0],epsState[6*i+4,0],epsState[6*i+5,0]]])
                    eigs,trash = np.linalg.eig(tmpEps)
                    data += [min(eigs)]
            elif crit=='Max Abs Principle Strain':
                for i in range(0,n):
                    tmpEps = np.array([[epsState[6*i+0,0],epsState[6*i+2,0],epsState[6*i+3,0]],
                                       [epsState[6*i+2,0],epsState[6*i+1,0],epsState[6*i+4,0]],
                                       [epsState[6*i+3,0],epsState[6*i+4,0],epsState[6*i+5,0]]])
                    eigs,trash = np.linalg.eig(tmpEps)
                    data += [max(abs(eigs))]
            elif crit=='Hoff':
                Xt = self.material.Xt
                Xc = self.material.Xc
                Yt = self.material.Yt
                Yc = self.material.Yc
                Zt = self.material.Zt
                Zc = self.material.Zc
                S12 = self.material.Sxy
                S13 = self.material.Sxz
                S23 = self.material.Syz
                C1 = .5*(1/(Zt*Zc)+1/(Yt*Yc)-1/(Xt*Xc))
                C2 = .5*(1/(Zt*Zc)-1/(Yt*Yc)+1/(Xt*Xc))
                C3 = .5*(-1/(Zt*Zc)+1/(Yt*Yc)+1/(Xt*Xc))
                C4 = 1/Xt-1/Xc
                C5 = 1/Yt-1/Yc
                C6 = 1/Zt-1/Zc
                C7 = 1/S23**2
                C8 = 1/S13**2
                C9 = 1/S12**2
                for i in range(0,n):
                    tmpSigVec = np.array([sigState[6*i+0,0],sigState[6*i+1,0],\
                                          sigState[6*i+5,0],sigState[6*i+4,0],\
                                          sigState[6*i+3,0],sigState[6*i+2,0]])

                    fiberSigVec = np.dot(self.Rsiginv,tmpSigVec)
                    F = C1*(fiberSigVec[1]-fiberSigVec[2])**2+\
                        C2*(fiberSigVec[2]-fiberSigVec[0])**2+\
                        C3*(fiberSigVec[0]-fiberSigVec[1])**2+\
                        C4*fiberSigVec[0]+C5*fiberSigVec[1]+C6*fiberSigVec[2]+\
                        C7*fiberSigVec[3]**2+C8*fiberSigVec[4]**2+C9*fiberSigVec[5]**2
                    data += [F]
            else:
                for i in range(0,int(self.nd/3)):
                    data += [0.]
            if centroid:
                if self.type in ['XQUAD4','XQUAD6','XQUAD8','XQUAD9']:
                    Ntmp = self.N(0,0)
                else:
                    Ntmp = self.N(1/3.,1/3.)
                if self.type=='XQUAD8':
                    tmpData = data[-1]
                elif self.type=='XQUAD9':
                    tmpData = data[4]
                else:
                    tmpData = 0.
                    for i in range(0,len(data)):
                        tmpData += data[i]*Ntmp[i]
                data = [tmpData]
            if len(data_env)==0:
                data_env=data
            else:
                for i in range(0,len(data_env)):
                    if abs(data_env[i])<abs(data[i]):
                        data_env[i]=data[i]
        return data_env

    def clearXSectionMatricies(self):
        """Clears large matricies associated with cross-sectional analaysis.

        Intended primarily as a private method but left public, this method
        clears the matricies associated with cross-sectional analysis. This is
        mainly done as a way of saving memory.

        """
        self.Ae = None
        self.Ce = None
        self.Ee = None
        self.Le = None
        self.Me = None
        self.Re = None

class XQUAD4(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XQUAD4',12,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[1].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = .25*(1-xi)*(1-eta)
        N[1] = .25*(1+xi)*(1-eta)
        N[2] = .25*(1+xi)*(1+eta)
        N[3] = .25*(1-xi)*(1+eta)
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = .25*(1-xi)*(1-eta)
        N2 = .25*(1+xi)*(1-eta)
        N3 = .25*(1+xi)*(1+eta)
        N4 = .25*(1-xi)*(1+eta)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        return Nmat

    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = -(1-eta)/4
        dNdxi[1] = (1-eta)/4
        dNdxi[2] = (1+eta)/4
        dNdxi[3] = -(1+eta)/4
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = -(1-xi)/4
        dNdeta[1] = -(1+xi)/4
        dNdeta[2] = (1+xi)/4
        dNdeta[3] = (1-xi)/4

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        # DN/Dxi
        xs = self.xs
        ys = self.ys
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = -(1-eta)/4
        dNdxi[1] = (1-eta)/4
        dNdxi[2] = (1+eta)/4
        dNdxi[3] = -(1+eta)/4
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = -(1-xi)/4
        dNdeta[1] = -(1+xi)/4
        dNdeta[2] = (1+xi)/4
        dNdeta[3] = (1-xi)/4

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        det = J11*J22-J12*J21
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = -(1-eta)/4
        dNdxi2 = (1-eta)/4
        dNdxi3 = (1+eta)/4
        dNdxi4 = -(1+eta)/4
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = -(1-xi)/4
        dNdeta2 = -(1+xi)/4
        dNdeta3 = (1+xi)/4
        dNdeta4 = (1-xi)/4
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        edges = (
                 (0+offset,1+offset),
                 (1+offset,2+offset),
                 (2+offset,3+offset),
                 (3+offset,0+offset))
        surfaces = (
                    (0+offset,1+offset,2+offset),
                    (0+offset,2+offset,3+offset))
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour

    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3','NID 4')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()

    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XQUAD4,{},{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3],\
                       self.MID,self.th[0],self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,4,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},{},0,0,0,0,0,0,\n'.format(self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8

class XQUAD6(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XQUAD6',18,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[5].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = .25*(xi**2-xi)*(1-eta)
        N[1] = .5*(1-xi**2)*(1-eta)
        N[2] = .25*(xi**2+xi)*(1-eta)
        N[3] = .25*(xi**2+xi)*(1+eta)
        N[4] = .5*(1-xi**2)*(1+eta)
        N[5] = .25*(xi**2-xi)*(1+eta)
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = .25*(xi**2-xi)*(1-eta)
        N2 = .5*(1-xi**2)*(1-eta)
        N3 = .25*(xi**2+xi)*(1-eta)
        N4 = .25*(xi**2+xi)*(1+eta)
        N5 = .5*(1-xi**2)*(1+eta)
        N6 = .25*(xi**2-xi)*(1+eta)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        Nmat[0:3,12:15] = N5*I3
        Nmat[0:3,15:18] = N6*I3
        return Nmat


    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = (-eta + 1)*(0.5*xi - 0.25)
        dNdxi[1] = -1.0*xi*(-eta + 1)
        dNdxi[2] = (-eta + 1)*(0.5*xi + 0.25)
        dNdxi[3] = (eta + 1)*(0.5*xi + 0.25)
        dNdxi[4] = -1.0*xi*(eta + 1)
        dNdxi[5] = (eta + 1)*(0.5*xi - 0.25)
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = -0.25*xi**2 + 0.25*xi
        dNdeta[1] = 0.5*xi**2 - 0.5
        dNdeta[2] = -0.25*xi**2 - 0.25*xi
        dNdeta[3] = 0.25*xi**2 + 0.25*xi
        dNdeta[4] = -0.5*xi**2 + 0.5
        dNdeta[5] = 0.25*xi**2 - 0.25*xi

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        # DN/Dxi
        xs = self.xs
        ys = self.ys
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = (-eta + 1)*(0.5*xi - 0.25)
        dNdxi[1] = -1.0*xi*(-eta + 1)
        dNdxi[2] = (-eta + 1)*(0.5*xi + 0.25)
        dNdxi[3] = (eta + 1)*(0.5*xi + 0.25)
        dNdxi[4] = -1.0*xi*(eta + 1)
        dNdxi[5] = (eta + 1)*(0.5*xi - 0.25)
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = -0.25*xi**2 + 0.25*xi
        dNdeta[1] = 0.5*xi**2 - 0.5
        dNdeta[2] = -0.25*xi**2 - 0.25*xi
        dNdeta[3] = 0.25*xi**2 + 0.25*xi
        dNdeta[4] = -0.5*xi**2 + 0.5
        dNdeta[5] = 0.25*xi**2 - 0.25*xi

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        det = J11*J22-J12*J21
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = (-eta + 1)*(0.5*xi - 0.25)
        dNdxi2 = -1.0*xi*(-eta + 1)
        dNdxi3 = (-eta + 1)*(0.5*xi + 0.25)
        dNdxi4 = (eta + 1)*(0.5*xi + 0.25)
        dNdxi5 = -1.0*xi*(eta + 1)
        dNdxi6 = (eta + 1)*(0.5*xi - 0.25)
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        dNdxi_mat[0:3,12:15] = dNdxi5*I3
        dNdxi_mat[0:3,15:18] = dNdxi6*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = -0.25*xi**2 + 0.25*xi
        dNdeta2 = 0.5*xi**2 - 0.5
        dNdeta3 = -0.25*xi**2 - 0.25*xi
        dNdeta4 = 0.25*xi**2 + 0.25*xi
        dNdeta5 = -0.5*xi**2 + 0.5
        dNdeta6 = 0.25*xi**2 - 0.25*xi
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        dNdeta_mat[0:3,12:15] = dNdeta5*I3
        dNdeta_mat[0:3,15:18] = dNdeta6*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        edges = (
                 (0+offset,1+offset),
                 (1+offset,2+offset),
                 (2+offset,3+offset),
                 (3+offset,4+offset),
                 (4+offset,5+offset),
                 (5+offset,0+offset),)
        surfaces = (
                    (0+offset,1+offset,4+offset),
                    (0+offset,4+offset,5+offset),
                    (1+offset,2+offset,3+offset),
                    (1+offset,3+offset,4+offset),
                    )
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour

    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3','NID 4','NID 5','NID 6')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()

    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XQUAD6,{},{},{},{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3],\
                       self.NIDs[4],self.NIDs[5],self.MID,self.th[0],\
                       self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,4,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},{},0,0,0,0,0,0,\n'.format(self.NIDs[0],self.NIDs[2],self.NIDs[3],self.NIDs[5])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8

class XQUAD8(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XQUAD8',24,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[6].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = -.25*(1-xi)*(1-eta)*(1+xi+eta)
        N[1] = 0.5*(1-xi**2)*(1-eta)
        N[2] = -.25*(1+xi)*(1-eta)*(1-xi+eta)
        N[3] = .5*(1+xi)*(1-eta**2)
        N[4] = -.25*(1+xi)*(1+eta)*(1-xi-eta)
        N[5] = .5*(1-xi**2)*(1+eta)
        N[6] = -.25*(1-xi)*(1+eta)*(1+xi-eta)
        N[7] = .5*(1-xi)*(1-eta**2)
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = -.25*(1-xi)*(1-eta)*(1+xi+eta)
        N2 = 0.5*(1-xi**2)*(1-eta)
        N3 = -.25*(1+xi)*(1-eta)*(1-xi+eta)
        N4 = .5*(1+xi)*(1-eta**2)
        N5 = -.25*(1+xi)*(1+eta)*(1-xi-eta)
        N6 = .5*(1-xi**2)*(1+eta)
        N7 = -.25*(1-xi)*(1+eta)*(1+xi-eta)
        N8 = .5*(1-xi)*(1-eta**2)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        Nmat[0:3,12:15] = N5*I3
        Nmat[0:3,15:18] = N6*I3
        Nmat[0:3,18:21] = N7*I3
        Nmat[0:3,21:24] = N8*I3
        return Nmat


    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = (-eta + 1)*(0.25*xi - 0.25) + 0.25*(-eta + 1)*(eta + xi + 1)
        dNdxi[1] = -1.0*xi*(-eta + 1)
        dNdxi[2] = -(-eta + 1)*(-0.25*xi - 0.25) - 0.25*(-eta + 1)*(eta - xi + 1)
        dNdxi[3] = -0.5*eta**2 + 0.5
        dNdxi[4] = -(eta + 1)*(-0.25*xi - 0.25) - 0.25*(eta + 1)*(-eta - xi + 1)
        dNdxi[5] = -1.0*xi*(eta + 1)
        dNdxi[6] = (eta + 1)*(0.25*xi - 0.25) + 0.25*(eta + 1)*(-eta + xi + 1)
        dNdxi[7] = 0.5*eta**2 - 0.5
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
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
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        # DN/Dxi
        xs = self.xs
        ys = self.ys
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = (-eta + 1)*(0.25*xi - 0.25) + 0.25*(-eta + 1)*(eta + xi + 1)
        dNdxi[1] = -1.0*xi*(-eta + 1)
        dNdxi[2] = -(-eta + 1)*(-0.25*xi - 0.25) - 0.25*(-eta + 1)*(eta - xi + 1)
        dNdxi[3] = -0.5*eta**2 + 0.5
        dNdxi[4] = -(eta + 1)*(-0.25*xi - 0.25) - 0.25*(eta + 1)*(-eta - xi + 1)
        dNdxi[5] = -1.0*xi*(eta + 1)
        dNdxi[6] = (eta + 1)*(0.25*xi - 0.25) + 0.25*(eta + 1)*(-eta + xi + 1)
        dNdxi[7] = 0.5*eta**2 - 0.5
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
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
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = (-eta + 1)*(0.25*xi - 0.25) + 0.25*(-eta + 1)*(eta + xi + 1)
        dNdxi2 = -1.0*xi*(-eta + 1)
        dNdxi3 = -(-eta + 1)*(-0.25*xi - 0.25) - 0.25*(-eta + 1)*(eta - xi + 1)
        dNdxi4 = -0.5*eta**2 + 0.5
        dNdxi5 = -(eta + 1)*(-0.25*xi - 0.25) - 0.25*(eta + 1)*(-eta - xi + 1)
        dNdxi6 = -1.0*xi*(eta + 1)
        dNdxi7 = (eta + 1)*(0.25*xi - 0.25) + 0.25*(eta + 1)*(-eta + xi + 1)
        dNdxi8 = 0.5*eta**2 - 0.5
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        dNdxi_mat[0:3,12:15] = dNdxi5*I3
        dNdxi_mat[0:3,15:18] = dNdxi6*I3
        dNdxi_mat[0:3,18:21] = dNdxi7*I3
        dNdxi_mat[0:3,21:24] = dNdxi8*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = (-eta + 1)*(0.25*xi - 0.25) - (0.25*xi - 0.25)*(eta + xi + 1)
        dNdeta2 = 0.5*xi**2 - 0.5
        dNdeta3 = (-eta + 1)*(-0.25*xi - 0.25) - (-0.25*xi - 0.25)*(eta - xi + 1)
        dNdeta4 = -2*eta*(0.5*xi + 0.5)
        dNdeta5 = -(eta + 1)*(-0.25*xi - 0.25) + (-0.25*xi - 0.25)*(-eta - xi + 1)
        dNdeta6 = -0.5*xi**2 + 0.5
        dNdeta7 = -(eta + 1)*(0.25*xi - 0.25) + (0.25*xi - 0.25)*(-eta + xi + 1)
        dNdeta8 = -2*eta*(-0.5*xi + 0.5)
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        dNdeta_mat[0:3,12:15] = dNdeta5*I3
        dNdeta_mat[0:3,15:18] = dNdeta6*I3
        dNdeta_mat[0:3,18:21] = dNdeta7*I3
        dNdeta_mat[0:3,21:24] = dNdeta8*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        xi = 0.
        eta = 0.
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        N = np.zeros(8)
        N[0] = -.25*(1-xi)*(1-eta)*(1+xi+eta)
        N[1] = 0.5*(1-xi**2)*(1-eta)
        N[2] = -.25*(1+xi)*(1-eta)*(1-xi+eta)
        N[3] = .5*(1+xi)*(1-eta**2)
        N[4] = -.25*(1+xi)*(1+eta)*(1-xi-eta)
        N[5] = .5*(1-xi**2)*(1+eta)
        N[6] = -.25*(1-xi)*(1+eta)*(1+xi-eta)
        N[7] = .5*(1-xi)*(1-eta**2)
        coords += (tuple([np.dot(N,self.xs),np.dot(N,self.ys),0]),)
        u_warp += (tuple(np.dot(self.Nmat(eta,xi),utmp)[:,0]),)
        edges = (
                 (0+offset,1+offset),
                 (1+offset,2+offset),
                 (2+offset,5+offset),
                 (5+offset,8+offset),
                 (8+offset,7+offset),
                 (7+offset,6+offset),
                 (6+offset,3+offset),
                 (3+offset,0+offset),)
        surfaces = (
                    (0+offset,1+offset,8+offset),
                    (0+offset,8+offset,7+offset),
                    (1+offset,2+offset,3+offset),
                    (1+offset,3+offset,8+offset),
                    (8+offset,3+offset,4+offset),
                    (8+offset,4+offset,5+offset),
                    (7+offset,8+offset,5+offset),
                    (7+offset,5+offset,6+offset),
                    )
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour
    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3','NID 4','NID 5','NID 6','NID 7','NID 8')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XQUAD8,{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3],\
                       self.NIDs[4],self.NIDs[5],self.NIDs[6],self.NIDs[7],\
                       self.MID,self.th[0],self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,5,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},{},{},{},{},{},0,0,\n'.format(self.NIDs[0],self.NIDs[2],\
                                                   self.NIDs[4],self.NIDs[6],\
                                                   self.NIDs[1],self.NIDs[3],\
                                                   self.NIDs[5],self.NIDs[7])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8

class XQUAD9(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XQUAD9',27,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[6].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = .25*(xi**2-xi)*(eta**2-eta)
        N[1] = .5*(1-xi**2)*(eta**2-eta)
        N[2] = .25*(xi**2+xi)*(eta**2-eta)
        N[3] = .5*(xi**2-xi)*(1-eta**2)
        N[4] = (1-xi**2)*(1-eta**2)
        N[5] = .5*(xi**2+xi)*(1-eta**2)
        N[6] = .25*(xi**2-xi)*(eta**2+eta)
        N[7] = .5*(1-xi**2)*(eta**2+eta)
        N[8] = .25*(xi**2+xi)*(eta**2+eta)
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = .25*(xi**2-xi)*(eta**2-eta)
        N2 = .5*(1-xi**2)*(eta**2-eta)
        N3 = .25*(xi**2+xi)*(eta**2-eta)
        N4 = .5*(xi**2-xi)*(1-eta**2)
        N5 = (1-xi**2)*(1-eta**2)
        N6 = .5*(xi**2+xi)*(1-eta**2)
        N7 = .25*(xi**2-xi)*(eta**2+eta)
        N8 = .5*(1-xi**2)*(eta**2+eta)
        N9 = .25*(xi**2+xi)*(eta**2+eta)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        Nmat[0:3,12:15] = N5*I3
        Nmat[0:3,15:18] = N6*I3
        Nmat[0:3,18:21] = N7*I3
        Nmat[0:3,21:24] = N8*I3
        Nmat[0:3,24:27] = N9*I3
        return Nmat


    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = .25*(eta**2-eta)*(2*xi-1)
        dNdxi[1] = -(eta**2-eta)*xi
        dNdxi[2] = .25*(eta**2-eta)*(2*xi+1)
        dNdxi[3] = .5*(1-eta**2)*(2*xi-1)
        dNdxi[4] = -2*(1-eta**2)*xi
        dNdxi[5] = .5*(1-eta**2)*(2*xi+1)
        dNdxi[6] = .25*(eta**2+eta)*(2*xi-1)
        dNdxi[7] = -(eta**2+eta)*xi
        dNdxi[8] = .25*(eta**2+eta)*(2*xi+1)
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = .25*(xi**2-xi)*(2*eta-1)
        dNdeta[1] = .5*(2*eta-1)*(1-xi**2)
        dNdeta[2] = .25*(2*eta-1)*(xi**2+xi)
        dNdeta[3] = -eta*(xi**2-xi)
        dNdeta[4] = -2*eta*(1-xi**2)
        dNdeta[5] = -eta*(xi**2+xi)
        dNdeta[6] = .25*(1+2*eta)*(xi**2-xi)
        dNdeta[7] = .5*(2*eta+1)*(1-xi**2)
        dNdeta[8] = .25*(1+2*eta)*(xi**2+xi)

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = .25*(eta**2-eta)*(2*xi-1)
        dNdxi[1] = -(eta**2-eta)*xi
        dNdxi[2] = .25*(eta**2-eta)*(2*xi+1)
        dNdxi[3] = .5*(1-eta**2)*(2*xi-1)
        dNdxi[4] = -2*(1-eta**2)*xi
        dNdxi[5] = .5*(1-eta**2)*(2*xi+1)
        dNdxi[6] = .25*(eta**2+eta)*(2*xi-1)
        dNdxi[7] = -(eta**2+eta)*xi
        dNdxi[8] = .25*(eta**2+eta)*(2*xi+1)
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = .25*(xi**2-xi)*(2*eta-1)
        dNdeta[1] = .5*(2*eta-1)*(1-xi**2)
        dNdeta[2] = .25*(2*eta-1)*(xi**2+xi)
        dNdeta[3] = -eta*(xi**2-xi)
        dNdeta[4] = -2*eta*(1-xi**2)
        dNdeta[5] = -eta*(xi**2+xi)
        dNdeta[6] = .25*(1+2*eta)*(xi**2-xi)
        dNdeta[7] = .5*(2*eta+1)*(1-xi**2)
        dNdeta[8] = .25*(1+2*eta)*(xi**2+xi)

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        det = J11*J22-J12*J21
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = .25*(eta**2-eta)*(2*xi-1)
        dNdxi2 = -(eta**2-eta)*xi
        dNdxi3 = .25*(eta**2-eta)*(2*xi+1)
        dNdxi4 = .5*(1-eta**2)*(2*xi-1)
        dNdxi5 = -2*(1-eta**2)*xi
        dNdxi6 = .5*(1-eta**2)*(2*xi+1)
        dNdxi7 = .25*(eta**2+eta)*(2*xi-1)
        dNdxi8 = -(eta**2+eta)*xi
        dNdxi9 = .25*(eta**2+eta)*(2*xi+1)
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        dNdxi_mat[0:3,12:15] = dNdxi5*I3
        dNdxi_mat[0:3,15:18] = dNdxi6*I3
        dNdxi_mat[0:3,18:21] = dNdxi7*I3
        dNdxi_mat[0:3,21:24] = dNdxi8*I3
        dNdxi_mat[0:3,24:27] = dNdxi9*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = .25*(xi**2-xi)*(2*eta-1)
        dNdeta2 = .5*(2*eta-1)*(1-xi**2)
        dNdeta3 = .25*(2*eta-1)*(xi**2+xi)
        dNdeta4 = -eta*(xi**2-xi)
        dNdeta5 = -2*eta*(1-xi**2)
        dNdeta6 = -eta*(xi**2+xi)
        dNdeta7 = .25*(1+2*eta)*(xi**2-xi)
        dNdeta8 = .5*(2*eta+1)*(1-xi**2)
        dNdeta9 = .25*(1+2*eta)*(xi**2+xi)
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        dNdeta_mat[0:3,12:15] = dNdeta5*I3
        dNdeta_mat[0:3,15:18] = dNdeta6*I3
        dNdeta_mat[0:3,18:21] = dNdeta7*I3
        dNdeta_mat[0:3,21:24] = dNdeta8*I3
        dNdeta_mat[0:3,24:27] = dNdeta9*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        edges = (
                 (0+offset,1+offset),
                 (1+offset,2+offset),
                 (2+offset,5+offset),
                 (5+offset,8+offset),
                 (8+offset,7+offset),
                 (7+offset,6+offset),
                 (6+offset,3+offset),
                 (3+offset,0+offset),)
        surfaces = (
                    (0+offset,1+offset,4+offset),
                    (0+offset,4+offset,3+offset),
                    (1+offset,2+offset,5+offset),
                    (1+offset,5+offset,4+offset),
                    (3+offset,4+offset,7+offset),
                    (3+offset,7+offset,6+offset),
                    (4+offset,5+offset,8+offset),
                    (4+offset,8+offset,7+offset),
                    )
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour
    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3','NID 4','NID 5','NID 6','NID 7','NID 8','NID 9')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XQUAD9,{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3],\
                       self.NIDs[4],self.NIDs[5],self.NIDs[6],self.NIDs[7],\
                       self.NIDs[8],self.MID,self.th[0],self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,5,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},{},{},{},{},{},0,0,\n'.format(self.NIDs[0],self.NIDs[2],\
                                                   self.NIDs[8],self.NIDs[6],\
                                                   self.NIDs[1],self.NIDs[5],\
                                                   self.NIDs[7],self.NIDs[3])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8

class XTRIA3(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XTRIA3',9,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[1].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = xi
        N[1] = eta
        N[2] = 1-xi-eta
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = xi
        N2 = eta
        N3 = 1-xi-eta
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        return Nmat

    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = 1
        dNdxi[1] = 0
        dNdxi[2] = -1
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = 0
        dNdeta[1] = 1
        dNdeta[2] = -1

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        # DN/Dxi
        xs = self.xs
        ys = self.ys
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = 1
        dNdxi[1] = 0
        dNdxi[2] = -1
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = 0
        dNdeta[1] = 1
        dNdeta[2] = -1

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        det = J11*J22-J12*J21
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = 1
        dNdxi2 = 0
        dNdxi3 = -1
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = 0
        dNdeta2 = 1
        dNdeta3 = -1
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        edges = (
                 (0+offset,1+offset),
                 (1+offset,2+offset),
                 (2+offset,3+offset))
        surfaces = (
                    (0+offset,1+offset,2+offset),)
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour

    def printSummary(self,nodes=False):
        """A method for printing a summary of the XTRIA element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()

    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XTRIA3,{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],\
                       self.MID,self.th[0],self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,2,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},0,0,0,0,0,0,0,\n'.format(self.NIDs[0],self.NIDs[1],self.NIDs[2])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8

class XTRIA6(XELEMENT):
    def __init__(self,EID,nodes,material,**kwargs):
        super().__init__(EID,nodes,material,'XTRIA6',18,**kwargs)

    def getNormalSign(self):
        nodes = self.nodes
        # Determine the direction of the element unit normal
        x1tmp = np.array(nodes[1].x)-np.array(nodes[0].x)
        x2tmp = np.array(nodes[2].x)-np.array(nodes[0].x)
        sign = 1
        if x1tmp[0]*x2tmp[1]-x1tmp[1]*x2tmp[0]<0:
            sign = -1
        return sign

    def N(self,eta,xi):
        N = np.zeros(int(self.nd/3))
        N[0] = xi*(2*xi-1)
        N[1] = eta*(2*eta-1)
        N[2] = (1-eta-xi)*(2*(1-eta-xi)-1)
        N[3] = 4*xi*eta
        N[4] = 4*eta*(1-eta-xi)
        N[5] = 4*xi*(1-eta-xi)
        return N

    def Nmat(self,eta,xi):
        """Generates the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        Nmat = np.zeros((3,self.nd))
        N1 = xi*(2*xi-1)
        N2 = eta*(2*eta-1)
        N3 = (1-eta-xi)*(2*(1-eta-xi)-1)
        N4 = 4*xi*eta
        N5 = 4*eta*(1-eta-xi)
        N6 = 4*xi*(1-eta-xi)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        Nmat[0:3,12:15] = N5*I3
        Nmat[0:3,15:18] = N6*I3
        return Nmat


    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.

        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        xs = self.xs
        ys = self.ys
        # DN/Dxi
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = 4*xi-1
        dNdxi[1] = 0
        dNdxi[2] = 4*xi+4*eta-3
        dNdxi[3] = 4*eta
        dNdxi[4] = -4*eta
        dNdxi[5] = 4-4*eta-8*xi
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = 0
        dNdeta[1] = 4*eta-1
        dNdeta[2] = 4*xi+4*eta-3
        dNdeta[3] = 4*xi
        dNdeta[4] = 4-8*eta-4*xi
        dNdeta[5] = -4*xi

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def Jdet_inv(self,eta,xi):
        # DN/Dxi
        xs = self.xs
        ys = self.ys
        dNdxi = np.zeros(int(self.nd/3))
        dNdxi[0] = 4*xi-1
        dNdxi[1] = 0
        dNdxi[2] = 4*xi+4*eta-3
        dNdxi[3] = 4*eta
        dNdxi[4] = -4*eta
        dNdxi[5] = 4-4*eta-8*xi
        # DN/Deta
        dNdeta = np.zeros(int(self.nd/3))
        dNdeta[0] = 0
        dNdeta[1] = 4*eta-1
        dNdeta[2] = 4*xi+4*eta-3
        dNdeta[3] = 4*xi
        dNdeta[4] = 4-8*eta-4*xi
        dNdeta[5] = -4*xi

        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        det = J11*J22-J12*J21
        if det==0:
            print('WARNING: Element {} has an indeterminate jacobian. Please check the element.\n'.format(self.EID))
        Jinvmat = (1/det)*np.array([[J22,-J12,0],[-J21,J11,0],[0,0,1]])
        return abs(det), Jinvmat

    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdxi_mat = np.zeros((3,self.nd))
        # DN/Dxi
        dNdxi1 = 4*xi-1
        dNdxi2 = 0
        dNdxi3 = 4*xi+4*eta-3
        dNdxi4 = 4*eta
        dNdxi5 = -4*eta
        dNdxi6 = 4-4*eta-8*xi
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        dNdxi_mat[0:3,12:15] = dNdxi5*I3
        dNdxi_mat[0:3,15:18] = dNdxi6*I3
        return dNdxi_mat

    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.

        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.

        :Args:

        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*

        :Returns:

        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.

        .. Note:: Xi and eta can both vary between -1 and 1 respectively.

        """
        dNdeta_mat = np.zeros((3,self.nd))
        # DN/Deta
        dNdeta1 = 0
        dNdeta2 = 4*eta-1
        dNdeta3 = 4*xi+4*eta-3
        dNdeta4 = 4*xi
        dNdeta5 = 4-8*eta-4*xi
        dNdeta6 = -4*xi
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        dNdeta_mat[0:3,12:15] = dNdeta5*I3
        dNdeta_mat[0:3,15:18] = dNdeta6*I3
        return dNdeta_mat

    def getGlData(self,LCIDs,contour=None,offset=0):
        coords = ()
        u_warp = ()
        # Initialize the full warping displacement vector
        if len(LCIDs)==1:
            utmp = self.U[LCIDs[0]]
        else:
            utmp = self.U[-1]
        for i in range(0,int(self.nd/3)):
            coords += (tuple(self.nodes[i].x),)
            u_warp += (tuple(utmp[3*i:3*i+3,:].T[0]),)
        edges = (
                 (0+offset,3+offset),
                 (3+offset,5+offset),
                 (5+offset,0+offset),
                 (3+offset,1+offset),
                 (1+offset,4+offset),
                 (4+offset,3+offset),
                 (5+offset,4+offset),
                 (4+offset,2+offset),
                 (2+offset,5+offset),
                 (5+offset,3+offset),
                 (3+offset,4+offset),
                 (4+offset,5+offset),)
        surfaces = (
                    (0+offset,3+offset,5+offset),
                    (3+offset,1+offset,4+offset),
                    (5+offset,4+offset,2+offset),
                    (5+offset,3+offset,4+offset),
                    )
        contour = self.getContour(LCIDs,crit=contour)
        return coords, u_warp, edges, surfaces, contour
    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.

        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.

        :Args:

        - None

        :Returns:

        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.

        """
        print('ELEMENT {}:'.format(self.EID))
        print('Element Type: {}'.format(self.type))
        print('Referenced by cross-section {}'.format(self.XID))
        print('Node IDs:')
        headers = ('NID 1','NID 2','NID 3','NID 4','NID 5','NID 6')
        print(tabulate([self.NIDs],headers,tablefmt="fancy_grid"))
        print('Material ID: {}'.format(self.MID))
        print('Material rotations:')
        headers = ('Rx (deg)','Ry (deg)','Rz (deg)')
        print(tabulate([self.th],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()
    def writeToFile(self):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        return 'XTRIA6,{},{},{},{},{},{},{},{},{},{},{}'.format(self.EID,\
                       self.NIDs[0],self.NIDs[1],self.NIDs[2],self.NIDs[3],\
                       self.NIDs[4],self.NIDs[5],\
                       self.MID,self.th[0],self.th[1],self.th[2])
    def writeToNeutral(self):
        s1 = '{},124,{},17,3,1,0,0,0,0,0,0,0,0,0,\n'.format(self.EID,self.MID)
        s2 = '0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s3 = '{},{},{},0,{},{},{},0,0,0,\n'.format(self.NIDs[0],self.NIDs[1],\
                                                 self.NIDs[2],self.NIDs[3],\
                                                 self.NIDs[4],self.NIDs[5])
        s4 = '0,0,0,0,0,0,0,0,0,0,\n'
        s5 = '0.,0.,0.,0,0,0,0,0,0,\n'
        s6 = '0.,0.,0.,\n'
        s7 = '0.,0.,0.,\n'
        s8 = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n'
        s9 = '0,\n'
        return s1+s2+s3+s4+s5+s6+s7+s8


class XElementLibrary:
    """Creates an element cross-section library object.

    This element cross-section library holds the cross-sectional elements to be
    used for cross-section objects. Furthermore, it can be used to generate new
    cross-sectional element objects to be automatically stored within it. See
    the "X" element classes for further details.

    :Attributes:

    - `xelemDict (dict)`: A dictionary which stores xelem objects as the
        values with the XEIDs as the associated keys.

    :Methods:

    - `addXElement`: Adds an xelement to the XElemLib object dictionary.
    - `getXElement`: Returns an xelement object provided an XEID
    - `printSummary`: Prints a summary of all of the nodes held within the
        xelemDict dictionary.

    """
    def __init__(self):
        """Initialize XElemLib object.

        The initialization method is mainly used to initialize a dictionary
        which houses xelem objects.

        :Args:

        - None

        :Returns:

        - None

        """
        self.type='XElemLibrary'
        self.xelemDict = {}

    def add(self,xEID,nodes,material,elemType,**kwargs):
        """Add a node to the nodeLib object.

        This is the primary method of the class, used to create new node
        obects and then add them to the library for later use.

        :Args:

        - `xEID (int)`: The desired integer node ID
        - `nodes (1xN array[obj])`: A 1xN array of node objects.
        - `material (obj)`: The material object used by the element.
        - `elemType (str)`: A string calssifying the element being created.
            Supported elements include XQUAD4, XQUAD6, XQUAD8, XQUAD9

        :Returns:

        - None

        """
        if xEID in self.xelemDict.keys():
            print('WARNING: Overwritting cross-section element %d' %(xEID))
        if elemType=='XQUAD4':
            self.xelemDict[xEID] = XQUAD4(xEID,nodes,material,**kwargs)
        elif elemType=='XQUAD6':
            self.xelemDict[xEID] = XQUAD6(xEID,nodes,material,**kwargs)
        elif elemType=='XQUAD8':
            self.xelemDict[xEID] = XQUAD8(xEID,nodes,material,**kwargs)
        elif elemType=='XQUAD9':
            self.xelemDict[xEID] = XQUAD9(xEID,nodes,material,**kwargs)
        elif elemType=='XTRIA3':
            self.xelemDict[xEID] = XTRIA3(xEID,nodes,material,**kwargs)
        elif elemType=='XTRIA6':
            self.xelemDict[xEID] = XTRIA6(xEID,nodes,material,**kwargs)
        else:
            raise ValueError('You selected element type: {}. Please enter an \
                             element type that is supported.'.format(elemType))

    def get(self,xEID):
        """Method that returns a cross-section element from the cross-section
        element libary.

        :Args:

        - `xEID (int)`: The ID of the cross-section element which is desired

        :Returns:

        - `(obj): A cross-section element object associated with the key xEID

        """
        if not xEID in self.xelemDict.keys():
            print(xEID)
            raise KeyError('The xEID {} provided is not linked with any elements'+
                ' within the supplied cross-section element library.')
        return self.xelemDict[xEID]
    def getIDs(self):
        return self.xelemDict.keys()
    def delete(self,xEID):
        if not xEID in self.xelemDict.keys():
            raise KeyError('The xEID provided is not linked with any elements '+
                'within the supplied cross-section element library.')
        del self.xelemDict[xEID]
    def printSummary(self):
        """Prints summary of all cross-section elements in xelemLib

        A method used to print out tabulated summary of all of the elements
        held within the cross-section element library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the elements.

        """
        if len(self.xelemDict)==0:
            print('The cross-section element library is currently empty.\n')
        else:
            print('The cross-section elements are:')
            for xEID, elem in self.xelemDict.items():
                elem.printSummary()
    def writeToFile(self):
        """Prints summary of all xelements in xelementLib

        A method used to print out tabulated summary of all of the xelements
        held within the node library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the nodes.

        """
        print_statement = []
        if len(self.xelemDict)==0:
            print('The cross-section element library is currently empty.\n')
        else:
            for xEID, elem in self.xelemDict.items():
                print_statement += [elem.writeToFile()]
        return print_statement

class Mesh:
    def __init__(self,meshType,**kwargs):
        self.sxnid = kwargs.pop('sxnid',1)-1
        self.sxeid = kwargs.pop('sxeid',1)-1
        self.nodeDict = {self.sxnid:None}
        self.elemDict = {self.sxeid:None}
        # Select meshing routine
        matLib = kwargs.pop('matLib',None)
        elemType = kwargs.pop('elemType','XQUAD4')
        if meshType == 'solidBox':
            L1 = kwargs.pop('L1',1.)
            L2 = kwargs.pop('L2',1.)
            elemX = kwargs.pop('elemX',2)
            elemY = kwargs.pop('elemY',2)
            MID = kwargs.pop('MID',None)
            print('Solid Rectangle meshing commencing:')
            self.solidBox(L1, L2, elemX, elemY, matLib, MID, elemType=elemType)
            print('Solid Rectangle meshing done')
            self.name=meshType
        elif meshType == 'laminate':
            L1 = kwargs.pop('L1',1.)
            elemAR = kwargs.pop('elemAR',2)
            if not matLib:
                raise ValueError("You must supply a material library object to"
                                 " use the laminate meshing routine.")
            laminate = kwargs.pop('laminate',None)
            if not laminate:
                raise ValueError("You must supply a laminate object to use the"
                                  "laminate meshing routine.")
            print('Laminate meshing commencing:')
            self.laminate(L1,laminate,elemAR,matLib,elemType)
            print('Laminate meshing done')
            self.name=meshType
        elif meshType == 'compositeTube':
            R = kwargs.pop('R',1.)
            laminates = kwargs.pop('laminates',[])
            elemAR = kwargs.pop('elemAR',2.)
            print('Composite Tube meshing commencing:')
            self.compositeTube(R,laminates,elemAR,matLib,elemType)
            print('Composite Tube meshing done')
            self.name=meshType
        elif meshType == 'cchannel':
            self.name=meshType
            L1 = kwargs.pop('L1',1.)
            L2 = kwargs.pop('L2',1.)
            elemAR = kwargs.pop('elemAR',2)
            if not matLib:
                raise ValueError("You must supply a material library object to"
                                 " use the laminate meshing routine.")
            laminate = kwargs.pop('laminate',None)
            if not laminate:
                raise ValueError("You must supply a laminate object to use the"
                                  "laminate meshing routine.")
            print('C-Channel meshing commencing:')
            self.cchannel(L1,L2,laminate,elemAR,matLib,elemType)
        elif meshType == 'general':
            print('General Cross-section selected.')
            self.nodeDict =  kwargs.pop('nodeDict',{self.sxnid:None})
            self.elemDict = kwargs.pop('elemDict',{self.sxeid:None})

    def __meshRegion__(self,elemType,elemY,elemX,MeshNID,matLib,laminate=None,MID=None,reverse_lam=False):
        elemDict = self.elemDict
        nodeDict = self.nodeDict
        nids_2_remove = []
        if not (laminate or MID):
            raise ValueError("When meshing a region, either a single MID or a "
                             "laminate object must be provided.")
        if type(MID)==int:
            material = matLib.get(MID)
            if elemType=='XQUAD4':
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[i+1,j],MeshNID[i+1,j+1],MeshNID[i,j+1],MeshNID[i,j]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        elemDict[newEID] = XQUAD4(newEID,nodes,material)
            elif elemType=='XQUAD6':
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[i+1,2*j],MeshNID[i+1,2*j+1],MeshNID[i+1,2*j+2],\
                        MeshNID[i,2*j+2],MeshNID[i,2*j+1],MeshNID[i,2*j]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        elemDict[newEID] = XQUAD6(newEID,nodes,material)
            elif elemType=='XQUAD8':
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[2*i+2,2*j],MeshNID[2*i+2,2*j+1],MeshNID[2*i+2,2*j+2],\
                        MeshNID[2*i+1,2*j+2],MeshNID[2*i,2*j+2],MeshNID[2*i,2*j+1],\
                        MeshNID[2*i,2*j],MeshNID[2*i+1,2*j]]
                        nids_2_remove += [MeshNID[2*i+1,2*j+1]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        elemDict[newEID] = XQUAD8(newEID,nodes,material)
            elif elemType=='XQUAD9':
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[2*i+2,2*j],MeshNID[2*i+2,2*j+1],MeshNID[2*i+2,2*j+2],\
                        MeshNID[2*i+1,2*j],MeshNID[2*i+1,2*j+1],MeshNID[2*i+1,2*j+2],\
                        MeshNID[2*i,2*j],MeshNID[2*i,2*j+1],MeshNID[2*i,2*j+2]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        elemDict[newEID] = XQUAD9(newEID,nodes,material)

        elif laminate:
            xVector = np.array([1.,0,0])
            yVector = np.array([0,1.,0])
            a=0
            b=1
            if reverse_lam:
                a=-1
                b=-1
            if elemType=='XQUAD4':
                if len(laminate.plies)==elemY:
                    ply_axis_y = True
                else:
                    ply_axis_y = False
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[i+1,j],MeshNID[i+1,j+1],MeshNID[i,j+1],MeshNID[i,j]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        if ply_axis_y:
                            vec1 = np.array(nodes[1].x)-np.array(nodes[0].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[2].x)-np.array(nodes[3].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            MID = laminate.plies[a+b*i].MID
                            th = [laminate.thi[a+b*i],0.,np.rad2deg(phi)]
                        else:
                            vec1 = np.array(nodes[0].x)-np.array(nodes[3].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[1].x)-np.array(nodes[2].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            MID = laminate.plies[a+b*j].MID
                            th = [laminate.thi[a+b*j],0.,np.rad2deg(phi)]
                        material = matLib.get(MID)

                        elemDict[newEID] = XQUAD4(newEID,nodes,material,th=th)
            elif elemType=='XQUAD6':
                if len(laminate.plies)==elemY:
                    ply_axis_y = True
                else:
                    ply_axis_y = False
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        if ply_axis_y:
                            NIDs = [MeshNID[i+1,2*j],MeshNID[i+1,2*j+1],MeshNID[i+1,2*j+2],\
                            MeshNID[i,2*j+2],MeshNID[i,2*j+1],MeshNID[i,2*j]]
                            nodes = [nodeDict[NID] for NID in NIDs]
                            vec1 = np.array(nodes[2].x)-np.array(nodes[0].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[3].x)-np.array(nodes[5].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            MID = laminate.plies[a+b*i].MID
                            th = [laminate.thi[a+b*i],0.,np.rad2deg(phi)]
                        else:
                            NIDs = [MeshNID[2*i,j],MeshNID[2*i+1,j],MeshNID[2*i+2,j],\
                            MeshNID[2*i+2,j+1],MeshNID[2*i+1,j+1],MeshNID[2*i,j+1]]
                            nodes = [nodeDict[NID] for NID in NIDs]
                            vec1 = np.array(nodes[2].x)-np.array(nodes[0].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[3].x)-np.array(nodes[5].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            MID = laminate.plies[a+b*j].MID
                            th = [laminate.thi[a+b*j],0.,np.rad2deg(phi)]
                        material = matLib.get(MID)


                        elemDict[newEID] = XQUAD6(newEID,nodes,material,th=th)
            elif elemType=='XQUAD8':
                if len(laminate.plies)==elemY:
                    ply_axis_y = True
                else:
                    ply_axis_y = False
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[2*i+2,2*j],MeshNID[2*i+2,2*j+1],MeshNID[2*i+2,2*j+2],\
                        MeshNID[2*i+1,2*j+2],MeshNID[2*i,2*j+2],MeshNID[2*i,2*j+1],\
                        MeshNID[2*i,2*j],MeshNID[2*i+1,2*j]]
                        nids_2_remove += [MeshNID[2*i+1,2*j+1]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        if ply_axis_y:
                            vec1 = np.array(nodes[2].x)-np.array(nodes[0].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[4].x)-np.array(nodes[6].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            th = [laminate.thi[a+b*i],0.,np.rad2deg(phi)]
                            MID = laminate.plies[a+b*i].MID
                        else:
                            vec1 = np.array(nodes[0].x)-np.array(nodes[7].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[2].x)-np.array(nodes[4].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            th = [laminate.thi[a+b*j],0.,np.rad2deg(phi)]
                            MID = laminate.plies[a+b*j].MID
                        material = matLib.get(MID)
                        elemDict[newEID] = XQUAD8(newEID,nodes,material,th=th)
            elif elemType=='XQUAD9':
                if len(laminate.plies)==elemY:
                    ply_axis_y = True
                else:
                    ply_axis_y = False
                for i in range(0,elemY):
                    for j in range(0,elemX):
                        newEID = int(max(elemDict.keys())+1)
                        NIDs = [MeshNID[2*i+2,2*j],MeshNID[2*i+2,2*j+1],MeshNID[2*i+2,2*j+2],\
                        MeshNID[2*i+1,2*j],MeshNID[2*i+1,2*j+1],MeshNID[2*i+1,2*j+2],\
                        MeshNID[2*i,2*j],MeshNID[2*i,2*j+1],MeshNID[2*i,2*j+2]]
                        nodes = [nodeDict[NID] for NID in NIDs]
                        if ply_axis_y:
                            vec1 = np.array(nodes[2].x)-np.array(nodes[0].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[8].x)-np.array(nodes[6].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            th = [laminate.thi[a+b*i],0.,np.rad2deg(phi)]
                            MID = laminate.plies[a+b*i].MID
                        else:
                            vec1 = np.array(nodes[0].x)-np.array(nodes[6].x)
                            vec1norm = np.linalg.norm(vec1)
                            vec2 = np.array(nodes[2].x)-np.array(nodes[8].x)
                            vec2norm = np.linalg.norm(vec2)
                            if np.dot(xVector,vec1)>0:
                                phi1 = -np.arccos(np.dot(vec1,yVector)/(vec1norm))+2*np.pi
                                phi2 = -np.arccos(np.dot(vec2,yVector)/(vec2norm))+2*np.pi
                                phi = (phi1+phi2)/2.
                            else:
                                phi1 = np.arccos(np.dot(vec1,yVector)/(vec1norm))
                                phi2 = np.arccos(np.dot(vec2,yVector)/(vec2norm))
                                phi = (phi1+phi2)/2.
                            th = [laminate.thi[a+b*j],0.,np.rad2deg(phi)]
                            MID = laminate.plies[a+b*j].MID
                        material = matLib.get(MID)
                        elemDict[newEID] = XQUAD9(newEID,nodes,material,th=th)

        self.elemDict = elemDict
        #print(nodeDict.keys())
        try:
            del self.nodeDict[self.sxnid]
            del self.elemDict[self.sxeid]
        except:
            pass
        for NID in nids_2_remove:
            del self.nodeDict[NID]

    def solidBox(self,L1, L2, elemX, elemY, matLib, MID, elemType='XQUAD4'):
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = self.nodeDict
        # Initialize the z location of the cross-section
        if elemType=='XQUAD4':
            nnx = elemX+1
            nny = elemY+1
        elif elemType=='XQUAD6':
            nnx = 2*elemX+1
            nny = elemY+1
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx = 2*elemX+1
            nny = 2*elemY+1
        # Create Mesh
        xvec = np.linspace(-L1/2,L1/2,nnx)
        yvec = np.linspace(-L2/2,L2/2,nny)[::-1]
        # NID Mesh
        MeshNID = np.zeros((nny,nnx),dtype=int)
        xmesh,ymesh = np.meshgrid(xvec,yvec)
        for i in range(0,nny):
            for j in range(0,nnx):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh[i,j],ymesh[i,j])

        #xsect.nodeDict = nodeDict
        self.__meshRegion__(elemType,elemY,elemX,MeshNID,matLib,MID=MID)

    def laminate(self,L1,laminate,elemAR,matLib,elemType):
        nodeDict = self.nodeDict
        elemY = len(laminate.t)
        elemX = int(L1/(min(laminate.t)*elemAR))
        if elemType=='XQUAD4':
            nnx = elemX+1
            nny = elemY+1
            yvec = laminate.z
        elif elemType=='XQUAD6':
            nnx = 2*elemX+1
            nny = elemY+1
            yvec = laminate.z
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx = 2*elemX+1
            nny = 2*elemY+1
            yvec = [-laminate.H/2]
            for i in range(0,len(laminate.t)):
                yvec += [yvec[2*i]+laminate.t[i]/2,yvec[2*i]+laminate.t[i]]
        xvec = np.linspace(-L1/2,L1/2,nnx)
        yvec = yvec[::-1]
        #yvec = laminate.z#np.linspace(-laminate.H/2,laminate.H/2,nny)[::-1]
        MeshNID = np.zeros((nny,nnx),dtype=int)
        xmesh,ymesh = np.meshgrid(xvec,yvec)
        for i in range(0,nny):
            for j in range(0,nnx):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh[i,j],ymesh[i,j])
        self.__meshRegion__(elemType,elemY,elemX,MeshNID,matLib,laminate=laminate)

    def compositeTube(self,R,laminates,elemAR,matLib,elemType):
        numSections = len(laminates)
        phis = [0]
        dPhi = 2*np.pi/numSections
        for i in range(0,numSections):
            phis += [phis[i]+dPhi]
        nodeDict = self.nodeDict
        MeshNIDs = []
        elemXs = []
        elemYs = []
        nids_2_remove = []
        mint = 1e6
        for k in range(0,len(laminates)):
            laminate = laminates[k]
            if min(laminate.t)<mint:
                mint = min(laminate.t)
            elemY = len(laminate.t)
            elemX = int(dPhi*R/(min(laminate.t)*elemAR))
            if elemType=='XQUAD4':
                nnx = elemX+1
                nny = elemY+1
                yvec = laminate.z+laminate.H/2
            elif elemType=='XQUAD6':
                nnx = 2*elemX+1
                nny = elemY+1
                yvec = laminate.z+laminate.H/2
            elif elemType=='XQUAD9' or 'XQUAD8':
                nnx = 2*elemX+1
                nny = 2*elemY+1
                yvec = [0]
                for m in range(0,len(laminate.t)):
                    yvec += [yvec[2*m]+laminate.t[m]/2,yvec[2*m]+laminate.t[m]]
                yvec = np.array(yvec)
            elemXs += [elemX]
            elemYs += [elemY]
            phivec = np.linspace(phis[k],phis[k+1],nnx)
            Rvec = -yvec+R
            MeshNID = np.zeros((nny,nnx),dtype=int)
            phimesh,Rmesh = np.meshgrid(phivec,Rvec)
            xmesh = Rmesh*np.cos(phimesh)
            ymesh = Rmesh*np.sin(phimesh)
            for i in range(0,nny):
                for j in range(0,nnx):
                    newNID = int(max(nodeDict.keys())+1)
                    MeshNID[i,j] = newNID
                    #Add node to NID Dictionary
                    nodeDict[newNID] = XNode(newNID,xmesh[i,j],ymesh[i,j])
            MeshNIDs += [MeshNID]
        for k in range(0,len(laminates)):
            MeshNID1 = MeshNIDs[k]
            if k==len(laminates)-1:
                MeshNID2 = MeshNIDs[0]
            else:
                MeshNID2 = MeshNIDs[k+1]
            nny1 = np.size(MeshNID1,axis=0)
            nny2 = np.size(MeshNID2,axis=0)
            for i in range(0,min(nny1,nny2)):
                node1 = nodeDict[MeshNID1[-1-i,-1]]
                node2 = nodeDict[MeshNID2[-1-i,0]]
                if np.linalg.norm(np.array(node1.x)-np.array(node2.x))<mint*1e-2:
                    nids_2_remove += MeshNID2[-1-i,0]
                    MeshNID2[-1-i,0] = MeshNID1[-1-i,-1]
                else:
                    raise ValueError("The meshes between laminates does not"
                                     " match up. Make sure that the laminate"
                                     " thicknesses match up from bottom to"
                                     " top (ie, the first ply in the layup"
                                     " to the last ply in the layup).")
        for k in range(0,len(laminates)):
            #try:
            self.__meshRegion__(elemType,elemYs[k],elemXs[k],MeshNIDs[k],\
                                matLib,laminate=laminates[k])
            #except Exception as e: print(str(e))
        for NID in nids_2_remove:
            del self.nodeDict[NID]
        #xsect.nodeDict = nodeDict
    def cchannel(self,L1,L2,laminate,elemAR,matLib,elemType):
        # Meshes C-chanel cross-section with constant laminate
        nodeDict = self.nodeDict
        MeshNIDs = []
        elemXs = []
        elemYs = []
        nids_2_remove = []

        # Establish coordinates for top cap laminate
        elemX1 = len(laminate.t)
        elemY1 = int(L1/(min(laminate.t)*elemAR))
        elemXs += [elemX1]
        elemYs += [elemY1]
        if elemType=='XQUAD4':
            nnx1 = elemX1+1
            nny1 = elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = -L2/2+laminate.H/2+laminate.z
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = np.linspace(L1,laminate.z[j]+laminate.H/2,nny1)
        elif elemType=='XQUAD6':
            nnx1 = elemX1+1
            nny1 = 2*elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = -L2/2+laminate.H/2+laminate.z
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = np.linspace(L1,laminate.z[j]+laminate.H/2,nny1)
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx1 = 2*elemX1+1
            nny1 = 2*elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = [-L2/2]
            yvec1 = [0]
            for l in range(0,len(laminate.t)):
                xvec1 += [xvec1[2*l]+laminate.t[l]/2,xvec1[2*l]+laminate.t[l]]
                yvec1 += [yvec1[2*l]+laminate.t[l]/2,yvec1[2*l]+laminate.t[l]]
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = np.linspace(L1,yvec1[j],nny1)
        MeshNID1 = np.zeros((nny1,nnx1),dtype=int)
        for i in range(0,nny1):
            for j in range(0,nnx1):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID1[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh1[i,j],ymesh1[i,j])
        MeshNIDs += [MeshNID1]


        # Establish coordinates for web laminate
        elemX2 = int(L2/(min(laminate.t)*elemAR))
        elemY2 = len(laminate.t)
        elemXs += [elemX2]
        elemYs += [elemY2]
        if elemType=='XQUAD4':
            nnx2 = elemX2+1
            nny2 = elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            yvec2 = laminate.H/2+laminate.z[::-1]
            for i in range(0,nny2):
                xmesh2[i,:] = np.linspace(-L2/2+laminate.H/2+laminate.z[-1-i],L2/2-laminate.z[-1-i]-laminate.H/2,nnx2)
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        elif elemType=='XQUAD6':
            nnx2 = 2*elemX2+1
            nny2 = elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            yvec2 = laminate.H/2+laminate.z[::-1]
            for i in range(0,nny2):
                xmesh2[i,:] = np.linspace(-L2/2+laminate.H/2+laminate.z[-1-i],L2/2-laminate.z[-1-i]-laminate.H/2,nnx2)
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx2 = 2*elemX2+1
            nny2 = 2*elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            yvec2 = [laminate.H]
            for l in range(0,len(laminate.t)):
                yvec2 += [yvec2[2*l]-laminate.t[-1-l]/2,yvec2[2*l]-laminate.t[-1-l]]
            for i in range(0,nny2):
                xmesh2[i,:] = np.linspace(-L2/2+yvec2[i],L2/2-yvec2[i],nnx2)
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        MeshNID2 = np.zeros((nny2,nnx2),dtype=int)
        for i in range(0,nny2):
            for j in range(0,nnx2):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID2[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh2[i,j],ymesh2[i,j])
        MeshNIDs += [MeshNID2]

        # Establish coordinates for bottom cap laminate
        elemXs += [elemX1]
        elemYs += [elemY1]
        xmesh_dim0 = np.size(xmesh1,axis=0)
        xmesh_dim1 = np.size(xmesh1,axis=1)
        xmesh3 = np.zeros((xmesh_dim0,xmesh_dim1))
        ymesh3 = np.zeros((xmesh_dim0,xmesh_dim1))
        for i in range(0,np.size(xmesh1,axis=0)):
            for j in range(0,np.size(xmesh1,axis=1)):
                xmesh3[i,j] = -xmesh1[i,-1-j]
                ymesh3[i,j] = ymesh1[i,-1-j]
        MeshNID3 = np.zeros((nny1,nnx1),dtype=int)
        for i in range(0,nny1):
            for j in range(0,nnx1):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID3[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh3[i,j],ymesh3[i,j])
        MeshNIDs += [MeshNID3]

        # Connect Mesh Region 1 to Mesh Region 2
        nids_2_remove += list(MeshNID2[:,0])
        MeshNID2[:,0] = MeshNID1[-1,:][::-1]

        # Connect Mesh Region 2 to Mesh Region 3
        nids_2_remove += list(MeshNID2[:,-1])
        MeshNID2[:,-1] = MeshNID3[-1,:]
#        for k in range(0,len(MeshNIDs)-1):
#            MeshNID_0 = MeshNIDs[k]
#            MeshNID_1 = MeshNIDs[k+1]
#            nn_match = np.size(MeshNID_0,axis=0)
#            for i in range(0,nn_match):
#                node1 = nodeDict[MeshNID1[-1-i,-1]]
#                node2 = nodeDict[MeshNID2[-1-i,0]]
#                if np.linalg.norm(np.array(node1.x)-np.array(node2.x))<mint*1e-2:
#                    nids_2_remove += MeshNID2[-1-i,0]
#                    MeshNID2[-1-i,0] = MeshNID1[-1-i,-1]
#                else:
#                    raise ValueError("The meshes between laminates does not"
#                                     " match up. Make sure that the laminate"
#                                     " thicknesses match up from bottom to"
#                                     " top (ie, the first ply in the layup"
#                                     " to the last ply in the layup).")
        reverse_lam = [False,True,True]
        for k in range(0,len(MeshNIDs)):
            #try:
#            print(len(laminate.plies))
#            print(elemYs[k])
#            print(elemXs[k])
            self.__meshRegion__(elemType,elemYs[k],elemXs[k],MeshNIDs[k],\
                                matLib,laminate=laminate,reverse_lam=reverse_lam[k])
            #except Exception as e: print(str(e))
        for NID in nids_2_remove:
            del self.nodeDict[NID]

    def cchanel_spar(self,L1,L2,L3,laminates,elemAR,matLib,elemType):
        # Meshes C-chanel cross-section with constant laminate
        nodeDict = self.nodeDict
        MeshNIDs = []
        elemXs = []
        elemYs = []
        nids_2_remove = []

        lam1 = laminates[0]
        lam2 = laminates[1]
        lam3 = laminates[2]

        # Establish coordinates for top cap laminate
        elemX1 = int((L1-lam2.H)/(min(lam1.t)*elemAR))
        elemY1 = len(lam1.t)
#        elemXs += [elemX1]
#        elemYs += [elemY1]
        if elemType=='XQUAD4':
            nnx1 = elemX1+1
            nny1 = elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = np.linspace(lam2.H,L1-lam2.H,nnx1)
            yvec1 = L2/2-lam1.H/2-lam1.z
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = yvec1
        elif elemType=='XQUAD6':
            nnx1 = 2*elemX1+1
            nny1 = elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = np.linspace(lam2.H,L1-lam2.H,nnx1)
            yvec1 = L2/2-lam1.H/2-lam1.z
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = yvec1
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx1 = 2*elemX1+1
            nny1 = 2*elemY1+1
            xmesh1 = np.zeros((nny1,nnx1))
            ymesh1 = np.zeros((nny1,nnx1))
            xvec1 = np.linspace(lam2.H,L1-lam2.H,nnx1)
            yvec1 = [L2/2]
            for l in range(0,len(lam1.t)):
                yvec1 += [yvec1[2*l]-lam1.t[l]/2,yvec1[2*l]-lam1.t[l]]
            for i in range(0,nny1):
                xmesh1[i,:] = xvec1
            for j in range(0,nnx1):
                ymesh1[:,j] = yvec1
        MeshNID1 = np.zeros((nny1,nnx1),dtype=int)
        for i in range(0,nny1):
            for j in range(0,nnx1):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID1[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh1[i,j],ymesh1[i,j])
        MeshNIDs += [MeshNID1]


        # Establish coordinates for web laminate
        elemX2 = len(lam2.t)
        elemY2 = int((L2-lam1.H-lam3.H)/(min(lam2.t)*elemAR))
        elemXs += [elemX2]
        elemYs += [elemY2]
        if elemType=='XQUAD4':
            nnx2 = elemX2+1
            nny2 = elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            xvec2 = lam2.z+lam2.H/2
            yvec2 = np.linspace(L2/2-lam1.H,-L2/2+lam3.H,nny2)
            for i in range(0,nny2):
                xmesh2[i,:] = xvec2
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        elif elemType=='XQUAD6':
            nnx2 = elemX2+1
            nny2 = 2*elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            xvec2 = lam2.z+lam2.H/2
            yvec2 = np.linspace(L2/2-lam1.H,-L2/2+lam3.H,nny2)
            for i in range(0,nny2):
                xmesh2[i,:] = xvec2
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx2 = 2*elemX2+1
            nny2 = 2*elemY2+1
            xmesh2 = np.zeros((nny2,nnx2))
            ymesh2 = np.zeros((nny2,nnx2))
            xvec2 = [0.]
            yvec2 = np.linspace(L2/2-lam1.H,-L2/2+lam3.H,nny2)
            for l in range(0,len(lam2.t)):
                xvec2 += [xvec2[2*l]+lam2.t[l]/2,xvec2[2*l]+lam2.t[l]]
            for i in range(0,nny2):
                xmesh2[i,:] = xvec2
            for j in range(0,nnx2):
                ymesh2[:,j] = yvec2
        MeshNID2 = np.zeros((nny2,nnx2),dtype=int)
        for i in range(0,nny2):
            for j in range(0,nnx2):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID2[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh2[i,j],ymesh2[i,j])
        MeshNIDs += [MeshNID2]

        # Establish coordinates for bottom cap laminate
        elemX3 = int((L3-lam2.H)/(min(lam3.t)*elemAR))
        elemY3 = len(lam3.t)
#        elemXs += [elemX1]
#        elemYs += [elemY1]
        if elemType=='XQUAD4':
            nnx3 = elemX3+1
            nny3 = elemY3+1
            xmesh3 = np.zeros((nny3,nnx3))
            ymesh3 = np.zeros((nny3,nnx3))
            xvec3 = np.linspace(lam2.H,L3-lam2.H,nnx3)
            yvec3 = -L2/2+lam3.H/2+lam3.z[::-1]
            for i in range(0,nny3):
                xmesh3[i,:] = xvec3
            for j in range(0,nnx3):
                ymesh3[:,j] = yvec3
        elif elemType=='XQUAD6':
            nnx3 = 2*elemX3+1
            nny3 = elemY3+1
            xmesh3 = np.zeros((nny3,nnx3))
            ymesh3 = np.zeros((nny3,nnx3))
            xvec3 = np.linspace(lam2.H,L3-lam2.H,nnx3)
            yvec3 = -L2/2+lam3.H/2+lam3.z[::-1]
            for i in range(0,nny3):
                xmesh3[i,:] = xvec3
            for j in range(0,nnx3):
                ymesh3[:,j] = yvec3
        elif elemType=='XQUAD9' or 'XQUAD8':
            nnx3 = 2*elemX3+1
            nny3 = 2*elemY3+1
            xmesh3 = np.zeros((nny3,nnx3))
            ymesh3 = np.zeros((nny3,nnx3))
            xvec3 = np.linspace(lam2.H,L3-lam2.H,nnx3)
            yvec3 = [-L2/2+lam3.H]
            for l in range(0,len(lam3.t)):
                yvec3 += [yvec3[2*l]-lam3.t[-1-l]/2,yvec3[2*l]-lam3.t[-1-l]]
            for i in range(0,nny3):
                xmesh3[i,:] = xvec3
            for j in range(0,nnx3):
                ymesh3[:,j] = yvec3
        MeshNID3 = np.zeros((nny3,nnx3),dtype=int)
        for i in range(0,nny3):
            for j in range(0,nnx3):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID3[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh3[i,j],ymesh3[i,j])
        MeshNIDs += [MeshNID3]

        # Connect Mesh Region 1 to Mesh Region 2
        nids_2_remove += list(MeshNID2[0,-1])
        MeshNID2[0,-1] = MeshNID1[-1,0]

        # Connect Mesh Region 2 to Mesh Region 3
        nids_2_remove += list(MeshNID2[-1,-1])
        MeshNID2[-1,-1] = MeshNID3[0,0]

        reverse_lam = [False,False,True]
        for k in range(0,len(MeshNIDs)):
            self.__meshRegion__(elemType,elemYs[k],elemXs[k],MeshNIDs[k],\
                                matLib,laminate=laminates[k],reverse_lam=reverse_lam[k])

        for NID in nids_2_remove:
            del self.nodeDict[NID]

        # MESH INTERSECTION REGIONS
        # Match plies between laminates 1 and 2
        corner_11_plies = []
        corner_12_plies = []
        lam_2_ind = 0
        for i in range(0,len(lam1.plies)):
            ply1 = lam1.plies[i]
            for j in range(lam_2_ind,len(lam2.plies)):
                ply2 = lam2.plies[j]
                if ply1.MID==ply2.MID and ply1.th==ply2.th:
                    corner_11_plies += [i]
                    corner_12_plies += [j]
                    lam_2_ind = j+1
                    break
        # Establish Nodal coordinates for intersection region 12
        xvec_inter_1 = xvec2
        yvec_inter_1 = yvec1
        xmesh12 = np.zeros((nny1,nnx2))
        ymesh12 = np.zeros((nny1,nnx2))
        for i in range(0,nny1):
            xmesh12[i,:] = xvec_inter_1
        for j in range(0,nnx2):
            ymesh12[:,j] = yvec_inter_1
        MeshNID12 = np.zeros((nny1,nnx2),dtype=int)
        for i in range(0,nny1):
            for j in range(0,nnx2):
                newNID = int(max(nodeDict.keys())+1)
                MeshNID12[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = XNode(newNID,xmesh12[i,j],ymesh12[i,j])
        MeshNIDs += [MeshNID12]
        nids_2_remove += list(MeshNID12[:,-1])
        MeshNID12[:,-1] = MeshNID1[:,0]
        nids_2_remove += list(MeshNID12[-1,:])
        MeshNID12[-1,:] = MeshNID2[0,:]
        # Mesh elements in inersection region 12
        elemDict = self.elemDict

        if elemType=='XQUAD4':
            for i in range(0,elemY1):
                for j in range(0,elemX2):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [MeshNID12[i+1,j],MeshNID12[i+1,j+1],MeshNID12[i,j+1],MeshNID12[i,j]]
                    nodes = [nodeDict[NID] for NID in NIDs]

                    elemDict[newEID] = XQUAD4(newEID,nodes,material,th=th)

    def boxBeam(self,xsect,meshSize,x0,xf,matlib):
        """Meshes a box beam cross-section.

        This meshing routine takes several parameters including a cross-section
        object `xsect`. This cross-section object should also contain the
        laminate objects used to construct it. There are no restrictions place
        on these laminates. Furthermore the outer mold line of this cross-
        section can take the form of any NACA 4-series airfoil. Finally, the
        convention is that for the four laminates that make up the box-beam,
        the the first ply in the laminate (which in CLT corresponds to the last
        ply in the stack) is located on the outside of the box beam. This
        convention can be seen below:

        .. image:: images/boxBeamGeom.png
            :align: center

        :Args:

        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.

        :Returns:

        - None

        """
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # The chord length of the airfoil profile
        c = Airfoil.c
        # Initialize the z location of the cross-section
        zc = 0
        # Initialize the Euler angler rotation about the local xsect z-axis for
        # any the given laminate. Note that individual elements might
        # experience further z-axis orientation if there is curvature in in the
        # OML of the cross-section.
        thz = [0,90,180,270]

        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==4:
            raise ValueError('The box beam cross-section was selected, but 4 '\
                'laminates were not provided')
        # Determine the number of plies per each laminate
        nlam1 = len(laminates[0].plies)
        nlam2 = len(laminates[1].plies)
        nlam3 = len(laminates[2].plies)
        nlam4 = len(laminates[3].plies)
        # Define boundary curves:
        # Note, the following curves represent the x-coordinate mesh
        # seeding along key regions, such as the connection region
        # between laminate 1 and 2
        x2 = np.zeros(len(laminates[1].plies))
        x4 = np.zeros(len(laminates[3].plies))
        x3 = np.linspace(x0+laminates[1].H/c,xf-laminates[3].H/c,int(((xf-laminates[3].H/c)\
            -(x0+laminates[1].H/c))/(meshSize*min(laminates[0].t)/c)))[1:]
        x5 = np.linspace(x0+laminates[1].H/c,xf-laminates[3].H/c,int(((xf-laminates[3].H/c)\
            -(x0+laminates[1].H/c))/(meshSize*min(laminates[2].t)/c)))[1:]
        # Populates the x-coordinates of the mesh seeding in curves x2 and
        # x4, which are the joint regions between the 4 laminates.
        x2 = x0+(laminates[1].z+laminates[1].H/2)/c
        x4 = xf-(laminates[3].z[::-1]+laminates[3].H/2)/c

        x1top = np.hstack((x2,x3,x4[1:]))
        x3bot = np.hstack((x2,x5,x4[1:]))

        # GENERATE LAMINATE 1 AND 3 MESHES
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 1 and 3). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        lam1Mesh = np.zeros((1+nlam1,len(x1top)),dtype=int)
        lam1xMesh = np.zeros((1+nlam1,len(x1top)))
        lam1yMesh = np.zeros((1+nlam1,len(x1top)))
        lam3Mesh = np.zeros((1+nlam3,len(x3bot)),dtype=int)
        lam3xMesh = np.zeros((1+nlam3,len(x3bot)))
        lam3yMesh = np.zeros((1+nlam3,len(x3bot)))
        #Generate the xy points of the top airfoil curve
        xu,yu,trash1,trash2 = Airfoil.points(x1top)
        #Generate the xy points of the bottom airfoil curve
        trash1,trash2,xl,yl = Airfoil.points(x3bot)
        #Generate the node objects for laminate 1
        ttmp = [0]+(laminates[0].z+laminates[0].H/2)
        for i in range(0,nlam1+1):
            for j in range(0,len(x1top)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam1Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xu[j],yu[j]-ttmp[i],zc]))
                lam1xMesh[i,j] = xu[j]
                lam1yMesh[i,j] = yu[j]-ttmp[i]
        #Generate  the node objects for laminate 3
        ttmp = [0]+laminates[2].z+laminates[2].H/2
        for i in range(0,nlam3+1):
            for j in range(0,len(x3bot)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam3Mesh[-1-i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xl[j],yl[j]+ttmp[i],zc]))
                lam3xMesh[-1-i,j] = xl[j]
                lam3yMesh[-1-i,j] = yl[j]+ttmp[i]
        #GENERATE LAMINATE 2 AND 4 MESHES
        #Define the mesh seeding for laminate 2
        meshLen2 = int(((yu[0]-laminates[0].H)-(yl[0]+laminates[2].H))/(meshSize*min(laminates[1].t)))
        #Define the mesh seeding for laminate 4
        meshLen4 = int(((yu[-1]-laminates[0].H)-(yl[-1]+laminates[2].H))/(meshSize*min(laminates[3].t)))
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 2 and 4). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        lam2Mesh = np.zeros((meshLen2,nlam2+1),dtype=int)
        lam2xMesh = np.zeros((meshLen2,nlam2+1))
        lam2yMesh = np.zeros((meshLen2,nlam2+1))
        lam4Mesh = np.zeros((meshLen4,nlam4+1),dtype=int)
        lam4xMesh = np.zeros((meshLen4,nlam4+1))
        lam4yMesh = np.zeros((meshLen4,nlam4+1))
        #Add connectivity nodes for lamiante 2
        lam2Mesh[0,:] = lam1Mesh[-1,0:nlam2+1]
        lam2xMesh[0,:] = lam1xMesh[-1,0:nlam2+1]
        lam2yMesh[0,:] = lam1yMesh[-1,0:nlam2+1]
        lam2Mesh[-1,:] = lam3Mesh[0,0:nlam2+1]
        lam2xMesh[-1,:] = lam3xMesh[0,0:nlam2+1]
        lam2yMesh[-1,:] = lam3yMesh[0,0:nlam2+1]
        #Generate the node objects for laminate 2
        for i in range(0,nlam2+1):
            lam2xMesh[:,i] = np.linspace(lam2xMesh[0,i],lam2xMesh[-1,i],meshLen2).T
            lam2yMesh[:,i] = np.linspace(lam2yMesh[0,i],lam2yMesh[-1,i],meshLen2).T
            for j in range(1,np.size(lam2xMesh,axis=0)-1):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam2Mesh[j,i] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam2xMesh[j,i],lam2yMesh[j,i],zc]))
        #Add connectivity nodes for lamiante 4
        lam4Mesh[0,:] = lam1Mesh[-1,-(nlam2+1):]
        lam4xMesh[0,:] = lam1xMesh[-1,-(nlam2+1):]
        lam4yMesh[0,:] = lam1yMesh[-1,-(nlam2+1):]
        lam4Mesh[-1,:] = lam3Mesh[0,-(nlam2+1):]
        lam4xMesh[-1,:] = lam3xMesh[0,-(nlam2+1):]
        lam4yMesh[-1,:] = lam3yMesh[0,-(nlam2+1):]
        #Generate the node objects for laminate 4
        for i in range(0,nlam4+1):
            lam4xMesh[:,i] = np.linspace(lam4xMesh[0,i],lam4xMesh[-1,i],meshLen4).T
            lam4yMesh[:,i] = np.linspace(lam4yMesh[0,i],lam4yMesh[-1,i],meshLen4).T
            for j in range(1,np.size(lam4Mesh,axis=0)-1):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam4Mesh[j,i] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam4xMesh[j,i],lam4yMesh[j,i],zc]))
        # Save meshes:
        xsect.laminates[0].mesh = lam1Mesh
        xsect.laminates[0].xmesh = lam1xMesh
        xsect.laminates[0].ymesh = lam1yMesh
        xsect.laminates[0].zmesh = np.zeros((1+nlam1,len(x1top)))

        xsect.laminates[1].mesh = lam2Mesh
        xsect.laminates[1].xmesh = lam2xMesh
        xsect.laminates[1].ymesh = lam2yMesh
        xsect.laminates[1].zmesh = np.zeros((meshLen2,nlam2+1))

        xsect.laminates[2].mesh = lam3Mesh
        xsect.laminates[2].xmesh = lam3xMesh
        xsect.laminates[2].ymesh = lam3yMesh
        xsect.laminates[2].zmesh = np.zeros((1+nlam3,len(x3bot)))

        xsect.laminates[3].mesh = lam4Mesh
        xsect.laminates[3].xmesh = lam4xMesh
        xsect.laminates[3].ymesh = lam4yMesh
        xsect.laminates[3].zmesh = np.zeros((meshLen4,nlam4+1))

        xsect.nodeDict = nodeDict
        xsect.xdim = max([np.max(lam1xMesh),np.max(lam2xMesh),np.max(lam3xMesh),np.max(lam4xMesh)])\
            -max([np.min(lam1xMesh),np.min(lam2xMesh),np.min(lam3xMesh),np.min(lam4xMesh)])
        xsect.ydim = max([np.max(lam1yMesh),np.max(lam2yMesh),np.max(lam3yMesh),np.max(lam4yMesh)])\
            -max([np.min(lam1yMesh),np.min(lam2yMesh),np.min(lam3yMesh),np.min(lam4yMesh)])

        for k in range(0,len(xsect.laminates)):
            ylen = np.size(xsect.laminates[k].mesh,axis=0)-1
            xlen = np.size(xsect.laminates[k].mesh,axis=1)-1
            # Ovearhead for later plotting of the cross-section. Will allow
            # for discontinuities in the contour should it arise (ie in
            # stress or strain contours).
            xsect.laminates[k].plotx = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].ploty = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotz = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotc = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].EIDmesh = np.zeros((ylen,xlen),dtype=int)
            for i in range(0,ylen):
                for j in range(0,xlen):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [xsect.laminates[k].mesh[i+1,j],xsect.laminates[k].mesh[i+1,j+1],\
                        xsect.laminates[k].mesh[i,j+1],xsect.laminates[k].mesh[i,j]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    # If the laminate is horizontal (i.e. divisible by 2)
                    if k % 2==0:
                        # Section determines how curvature in the beam causes
                        # slight variations in fiber rotation.
                        deltax1 = xsect.laminates[k].xmesh[i,j+1]-xsect.laminates[k].xmesh[i,j]
                        deltay1 = xsect.laminates[k].ymesh[i,j+1]-xsect.laminates[k].ymesh[i,j]
                        deltax2 = xsect.laminates[k].xmesh[i+1,j+1]-xsect.laminates[k].xmesh[i+1,j]
                        deltay2 = xsect.laminates[k].ymesh[i+1,j+1]-xsect.laminates[k].ymesh[i+1,j]
                        thz_loc = np.rad2deg(np.mean([np.arctan(deltay1/deltax1), np.arctan(deltay2/deltax2)]))
                        if k==0:
                            MID = xsect.laminates[k].plies[ylen-i-1].MID
                            th = [0,xsect.laminates[k].plies[ylen-i-1].th,thz[k]+thz_loc]
                        else:
                            MID = xsect.laminates[k].plies[i].MID
                            th = [0,xsect.laminates[k].plies[i].th,thz[k]+thz_loc]

                        #if newEID in [0,1692,1135,1134,2830,2831]:
                        #    print(th)
                    # Else if it is vertical:
                    else:
                        if k==1:
                            MID = xsect.laminates[k].plies[xlen-j-1].MID
                            th = [0,xsect.laminates[k].plies[xlen-j-1].th,thz[k]]
                        else:
                            MID = xsect.laminates[k].plies[j].MID
                            th = [0,xsect.laminates[k].plies[j].th,thz[k]]
                        #MID = xsect.laminates[k].plies[j].MID

                        #if newEID in [0,1692,1135,1134,2830,2831]:
                        #    print(th)
                    elemDict[newEID] = XQUAD4(newEID,nodes,MID,matlib,th=th)
                    xsect.laminates[k].EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]

    def cylindricalTube(self,xsect,r,meshSize,x0,xf,matlib,**kwargs):
        # Initialize the node dictionary, containing all local node objects
        # used by the cross-section
        nodeDict = {-1:None}
        # Initialize the node dictionary, containing all local element objects
        # used by the cross-section
        elemDict = {-1:None}
        # Initialize the X-Section z-coordinate
        zc = kwargs.pop('zc',0)
        # Initialize the laminates
        laminates = xsect.laminates
        # Initialize the number of plies per laminate (must be equal for all)
        nplies = len(laminates[0].plies)
        # Initialize the thickness vectors of plies per laminate (must be equal for all)
        ts = laminates[0].t
        # Determine the dtheta required for the cross-section
        minT = 1e9
        for lam in laminates:
            lamMin = min(lam.t)
            if lamMin<minT:
                minT = lamMin
            # Check the total number of laminates
            if not len(lam.plies)==nplies:
                raise ValueError('Note, for now all laminates must have the'\
                    'same number of plies.')
            # Check that the thicknesses all match
            if not np.array_equal(ts,lam.t):
                raise ValueError('Note, for now all laminates must have the'\
                    'Sane thickness distribution through the thickness of the'\
                    'laminate in order to preserve mesh compatability between'\
                    'laminates.')
        dth = meshSize*minT/r
        thz = []
        for i in range(0,len(laminates)):
            thz = np.append(thz,np.linspace(i*2*np.pi/len(laminates),\
                (i+1)*2*np.pi/len(laminates)),num=int(2*np.pi/(dth*len(laminates))))
        thz = np.unique(thz[0:-1])
        rvec = r+laminates[0].z+laminates[0].H/2
        rmat,thmat = np.meshgrid(rvec,thz)
        mesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)),dtype=int)
        xmesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        ymesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        zmesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        for i in range(0,np.size(rmat,axis=0)):
            for j in range(0,np.size(rmat,axis=1)):
                # Determine temp xy coordinates of the point
                xtmp = rmat[i,j]*np.cos(thmat[i,j])
                ytmp = rmat[i,j]*np.sin(thmat[i,j])
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xtmp,ytmp,zc]))
                xmesh[i,j] = xtmp
                ymesh[i,j] = ytmp
        # Assign parts of the total mesh to each laminate
        bound = np.linspace(0,1,num=len(laminates)+1)
        for i in range(0,len(laminates)):
            laminates[i].mesh = mesh[(thmat<=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].xmesh = xmesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].ymesh = ymesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].zmesh = zmesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].thmesh = thmat[(thmat<=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].EIDmesh = np.zeros((np.size(laminates[i].mesh,axis=0)\
                ,np.size(laminates[i].mesh,axis=1)),dtype=int)
        for lam in laminates:
            for i in range(0,np.size(lam.mesh,axis=0)-1):
                for j in range(0,np.size(lam.mesh,axis=1)-1):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [lam.mesh[i+1,j+1],lam.mesh[i+1,j],\
                        lam.mesh[i,j],lam.mesh[i,j+1]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    th = [0,lam.plies[i].th,lam.thmesh[i,j]]
                    MID = xsect.lam.plies[i].MID
                    elemDict[newEID] = XQUAD4(newEID,nodes,MID,matlib,th=th)
                    xsect.lam.EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def rectBoxBeam(self,xsect,meshSize,x0,xf,matlib):
        """Meshes a box beam cross-section.

        This method meshes a similar cross-section as the boxBeam method. The
        geometry of this cross-section can be seen below. The interfaces
        between the laminates is different, and more restrictive. In this case
        all of the laminates must have the same number of plies, which must
        also all be the same thickness.

        .. image:: images/rectBoxGeom.png
            :align: center

        :Args:

        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.

        :Returns:

        - None

        """
        print('Rectangular Box Meshing Commencing')
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # The chord length of the airfoil profile
        c = Airfoil.c
        # Initialize the z location of the cross-section
        zc = 0
        # Initialize the Euler angler rotation about the local xsect z-axis for
        # any the given laminate. Note that individual elements might
        # experience further z-axis orientation if there is curvature in in the
        # OML of the cross-section.
        thz = [0,90,180,270]

        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==4:
            raise ValueError('The box beam cross-section was selected, but 4 '\
                'laminates were not provided')
        # Determine the number of plies per each laminate
        nlam1 = len(laminates[0].plies)
        nlam2 = len(laminates[1].plies)
        nlam3 = len(laminates[2].plies)
        nlam4 = len(laminates[3].plies)
        # Define boundary curves:
        # Note, the following curves represent the x-coordinate mesh
        # seeding along key regions, such as the connection region
        # between laminate 1 and 2

        # Populates the x-coordinates of the mesh seeding in curves x2 and
        # x4, which are the joint regions between the 4 laminates.


        # Calculate important x points:
        x0 = x0*c
        x1 = x0+laminates[1].H
        xf = xf*c
        x2 = xf-laminates[3].H

        # Calculate important y points:
        y0 = -c/2
        y1 = y0+laminates[2].H
        yf = c/2
        y2 = yf-laminates[0].H

        # Determine the mesh seeding to maintain minimum AR
        lam13xSeeding = np.ceil((xf-x0)/(meshSize*min(laminates[0].t)))
        lam24ySeeding = np.ceil((yf-y0)/(meshSize*min(laminates[0].t)))

        # Define Finite Element Modeling Functions
        def x(eta,xi,xs):
            return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                    xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
        def y(eta,xi,ys):
            return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                    ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))

        # Generate Grids in superelement space
        xis13 = np.linspace(-1,1,lam13xSeeding+1)
        etas13 = np.linspace(1,-1,nlam1+1)
        lam1Mesh = np.zeros((1+nlam1,len(xis13)),dtype=int)
        lam3Mesh = np.zeros((1+nlam3,len(xis13)),dtype=int)
        xis13, etas13 = np.meshgrid(xis13,etas13)
        lam1xMesh = x(etas13,xis13,[x1,x2,xf,x0])
        lam1yMesh = y(etas13,xis13,[y2,y2,yf,yf])
        lam3xMesh = x(etas13,xis13,[x0,xf,x2,x1])
        lam3yMesh = y(etas13,xis13,[y0,y0,y1,y1])

        # GENERATE LAMINATE 1 AND 3 MESHES
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 1 and 3). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node

        for i in range(0,np.size(lam1xMesh,axis=0)):
            for j in range(0,np.size(lam1xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam1Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam1xMesh[i,j],lam1yMesh[i,j],zc]))
        #Generate  the node objects for laminate 3
        #ttmp = [0]+laminates[2].z+laminates[2].H/2
        for i in range(0,np.size(lam3xMesh,axis=0)):
            for j in range(0,np.size(lam3xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam3Mesh[-1-i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam3xMesh[-1-i,j],lam3yMesh[-1-i,j],zc]))
        #GENERATE LAMINATE 2 AND 4 MESHES
        #Define the mesh seeding for laminate 2
        #meshLen2 = int(((yu[0]-laminates[0].H)-(yl[0]+laminates[2].H))/(meshSize*min(laminates[1].t)))
        #Define the mesh seeding for laminate 4
        #meshLen4 = int(((yu[-1]-laminates[0].H)-(yl[-1]+laminates[2].H))/(meshSize*min(laminates[3].t)))
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 2 and 4). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node

        xis24 = np.linspace(-1,1,nlam2+1)
        etas24 = np.linspace(1,-1,lam24ySeeding+1)
        lam2Mesh = np.zeros((len(etas24),1+nlam2),dtype=int)
        lam4Mesh = np.zeros((len(etas24),1+nlam4),dtype=int)
        xis24, etas24 = np.meshgrid(xis24,etas24)
        lam2xMesh = x(etas24,xis24,[x0,x1,x1,x0])
        lam2yMesh = y(etas24,xis24,[y0,y1,y2,yf])
        lam4xMesh = x(etas24,xis24,[x2,xf,xf,x2])
        lam4yMesh = y(etas24,xis24,[y1,y0,yf,y2])

        #Add connectivity nodes for lamiante 2
        lam2Mesh[0,:] = lam1Mesh[:,0]
        lam2xMesh[0,:] = lam1xMesh[:,0]
        lam2yMesh[0,:] = lam1yMesh[:,0]
        lam2Mesh[-1,:] = lam3Mesh[::-1,0]
        lam2xMesh[-1,:] = lam3xMesh[::-1,0]
        lam2yMesh[-1,:] = lam3yMesh[::-1,0]
        #Add connectivity nodes for lamiante 4
        lam4Mesh[0,:] = lam1Mesh[::-1,-1]
        lam4xMesh[0,:] = lam1xMesh[::-1,-1]
        lam4yMesh[0,:] = lam1yMesh[::-1,-1]
        lam4Mesh[-1,:] = lam3Mesh[:,-1]
        lam4xMesh[-1,:] = lam3xMesh[:,-1]
        lam4yMesh[-1,:] = lam3yMesh[:,-1]
        #Generate the node objects for laminate 2
        for i in range(1,np.size(lam2xMesh,axis=0)-1):
            for j in range(0,np.size(lam2xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam2Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam2xMesh[i,j],lam2yMesh[i,j],zc]))
        #Generate the node objects for laminate 4
        for i in range(1,np.size(lam2xMesh,axis=0)-1):
            for j in range(0,np.size(lam2xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam4Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam4xMesh[i,j],lam4yMesh[i,j],zc]))
        # Save meshes:
        xsect.laminates[0].mesh = lam1Mesh
        xsect.laminates[0].xmesh = lam1xMesh
        xsect.laminates[0].ymesh = lam1yMesh
        xsect.laminates[0].zmesh = np.zeros((np.size(lam1Mesh,axis=0),np.size(lam1Mesh,axis=1)))

        xsect.laminates[1].mesh = lam2Mesh
        xsect.laminates[1].xmesh = lam2xMesh
        xsect.laminates[1].ymesh = lam2yMesh
        xsect.laminates[1].zmesh = np.zeros((np.size(lam2Mesh,axis=0),np.size(lam2Mesh,axis=1)))

        xsect.laminates[2].mesh = lam3Mesh
        xsect.laminates[2].xmesh = lam3xMesh
        xsect.laminates[2].ymesh = lam3yMesh
        xsect.laminates[2].zmesh = np.zeros((np.size(lam3Mesh,axis=0),np.size(lam3Mesh,axis=1)))

        xsect.laminates[3].mesh = lam4Mesh
        xsect.laminates[3].xmesh = lam4xMesh
        xsect.laminates[3].ymesh = lam4yMesh
        xsect.laminates[3].zmesh = np.zeros((np.size(lam4Mesh,axis=0),np.size(lam4Mesh,axis=1)))

        xsect.nodeDict = nodeDict

        for k in range(0,len(xsect.laminates)):
            ylen = np.size(xsect.laminates[k].mesh,axis=0)-1
            xlen = np.size(xsect.laminates[k].mesh,axis=1)-1
            # Ovearhead for later plotting of the cross-section. Will allow
            # for discontinuities in the contour should it arise (ie in
            # stress or strain contours).
            xsect.laminates[k].plotx = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].ploty = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotz = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotc = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].EIDmesh = np.zeros((ylen,xlen),dtype=int)
            for i in range(0,ylen):
                for j in range(0,xlen):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [xsect.laminates[k].mesh[i+1,j],xsect.laminates[k].mesh[i+1,j+1],\
                        xsect.laminates[k].mesh[i,j+1],xsect.laminates[k].mesh[i,j]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    if k==0:
                        MID = xsect.laminates[k].plies[-i-1].MID
                        th = [0,xsect.laminates[k].plies[-i-1].th,thz[k]]
                    elif k==1:
                        MID = xsect.laminates[k].plies[-j-1].MID
                        th = [0,xsect.laminates[k].plies[-j-1].th,thz[k]]
                    elif k==2:
                        MID = xsect.laminates[k].plies[i].MID
                        th = [0,xsect.laminates[k].plies[i].th,thz[k]]
                    else:
                        MID = xsect.laminates[k].plies[j].MID
                        th = [0,xsect.laminates[k].plies[j].th,thz[k]]
                    elemDict[newEID] = XQUAD4(newEID,nodes,MID,matlib,th=th)
                    xsect.laminates[k].EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]


    def rectangleHole(self,xsect, nelem, a, b, r, MID, matlib):
        """Meshes a box beam cross-section.

        This method meshes a similar cross-section as the boxBeam method. The
        geometry of this cross-section can be seen below. The interfaces
        between the laminates is different, and more restrictive. In this case
        all of the laminates must have the same number of plies, which must
        also all be the same thickness.

        .. image:: images/rectBoxGeom.png
            :align: center

        :Args:

        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.

        :Returns:

        - None

        """
        print('Box Meshing Commencing')
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        nelem=nelem*8+1
        laminate = xsect.laminates[0]
        # Initialize the z location of the cross-section
        xs = [a/2.,a/2.,0.,-a/2.,-a/2.,-a/2.,0.,a/2.,a/2.]
        ys = [0.,b/2.,b/2.,b/2.,0.,-b/2.,-b/2.,-b/2.,0.]

        xsvec = np.array([])
        ysvec = np.array([])

        for i in range(0,len(xs)-1):
            xsvec = np.append(xsvec,np.linspace(xs[i],xs[i+1],nelem/8.+1)[:-1])
            ysvec = np.append(ysvec,np.linspace(ys[i],ys[i+1],nelem/8.+1)[:-1])

        xc = r*np.cos(np.linspace(0,2*np.pi,nelem))[:-1]
        yc = r*np.sin(np.linspace(0,2*np.pi,nelem))[:-1]

        if not len(xc)==len(xsvec):
            raise ValueError('Circle and square vectors dont match length.')

        xmesh = np.zeros((int(nelem/8-1),len(xc)))
        ymesh = np.zeros((int(nelem/8-1),len(xc)))
        zmesh = np.zeros((int(nelem/8-1),len(xc)))
        Mesh = np.zeros((int(nelem/8-1),len(xc)),dtype=int)

        for i in range(0,len(xc)):
            xmesh[:,i]=np.linspace(xc[i],xsvec[i],nelem/8-1)
            ymesh[:,i]=np.linspace(yc[i],ysvec[i],nelem/8-1)

        for i in range(0,np.size(xmesh,axis=0)):
            for j in range(0,np.size(xmesh,axis=1)):
                newNID = int(max(nodeDict.keys())+1)
                Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xmesh[i,j],ymesh[i,j],zmesh[i,j]]))

        xmesh = np.hstack((xmesh,np.array([xmesh[:,0]]).T))
        ymesh = np.hstack((ymesh,np.array([ymesh[:,0]]).T))
        zmesh = np.hstack((zmesh,np.array([zmesh[:,0]]).T))
        Mesh = np.hstack((Mesh,np.array([Mesh[:,0]],dtype=int).T))

        xsect.nodeDict = nodeDict
        laminate.mesh = Mesh
        laminate.xmesh = xmesh
        laminate.ymesh = ymesh
        laminate.zmesh = zmesh

        EIDmesh = np.zeros((np.size(xmesh,axis=0)-1,np.size(xmesh,axis=1)-1),dtype=int)

        for i in range(0,np.size(xmesh,axis=0)-1):
            for j in range(0,np.size(xmesh,axis=1)-1):
                newEID = int(max(elemDict.keys())+1)
                NIDs = [Mesh[i+1,j],Mesh[i+1,j+1],Mesh[i,j+1],Mesh[i,j]]
                nodes = [xsect.nodeDict[NID] for NID in NIDs]
                elemDict[newEID] = XQUAD4(newEID,nodes,MID,matlib)
                EIDmesh[i,j] = newEID

        xsect.elemDict = elemDict
        ylen = np.size(xmesh,axis=0)-1
        xlen = np.size(xmesh,axis=1)-1
        laminate.plotx = np.zeros((ylen*2,xlen*2))
        laminate.ploty = np.zeros((ylen*2,xlen*2))
        laminate.plotz = np.zeros((ylen*2,xlen*2))
        laminate.plotc = np.zeros((ylen*2,xlen*2))
        laminate.EIDmesh = EIDmesh

        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
class XSect:
    """Creates a beam cross-section object,

    This cross-section can be made of multiple materials which can be in
    general anisotropic. This is the main workhorse within the structures
    library.

    :Attributes:

    - `Color (touple)`: A length 3 touple used to define the color of the
        cross-section.
    - `Airfoil (obj)`: The airfoil object used to define the OML of the cross-
        section.
    - `typeXSect (str)`: Defines what type of cross-section is to be used.
        Currently the only supported type is 'box'.
    - `normalVector (1x3 np.array[float])`: Expresses the normal vector of the
        cross-section.
    - `nodeDict (dict)`: A dictionary of all nodes used to descretize the
        cross-section surface. The keys are the NIDs and the values stored
        are the Node objects.
    - `elemDict (dict)`: A dictionary of all elements used to descretize the
        cross-section surface. the keys are the EIDs and the values stored
        are the element objects.
    - `X (ndx6 np.array[float])`: A very large 2D array. This is one of the
        results of the cross-sectional analysis. This array relays the
        force and moment resultants applied to the cross-section to the
        nodal warping displacements exhibited by the cross-section.
    - `Y (6x6 np.array[float])`: This array relays the force and moment
        resultants applied to the cross-section to the rigid section
        strains and curvatures exhibited by the cross-section.
    - `dXdz (ndx6 np.array[float])`: A very large 2D array. This is one of the
        results of the cross-sectional analysis. This array relays the
        force and moment resultants applied to the cross-section to the
        gradient of the nodal warping displacements exhibited by the
        cross-section with respect to the beam axis.
    - `xt (float)`: The x-coordinate of the tension center (point at which
        tension and bending are decoupled)
    - `yt (float)`: The y-coordinate of the tension center (point at which
        tension and bending are decoupled)
    - `xs (float)`: The x-coordinate of the shear center (point at which shear
        and torsion are decoupled)
    - `ys (float)`: The y-coordinate of the shear center (point at which shear
        and torsion are decoupled)
    - `refAxis (3x1 np.array[float])`: A column vector containing the reference
        axis for the beam.
    - `bendAxes (2x3 np.array[float])`: Contains two row vectors about which
        bending from one axis is decoupled from bending about the other.
    - `F_raw (6x6 np.array[float])`: The 6x6 compliance matrix that results
        from cross-sectional analysis. This is the case where the reference
        axis is at the origin.
    - `K_raw (6x6 np.array[float])`: The 6x6 stiffness matrix that results
        from cross-sectional analysis. This is the case where the reference
        axis is at the origin.
    - `F (6x6 np.array[float])`: The 6x6 compliance matrix for the cross-
        section about the reference axis. The reference axis is by default
        at the shear center.
    - `K (6x6 np.array[float])`: The 6x6 stiffness matrix for the cross-
        section about the reference axis. The reference axis is by default
        at the shear center.
    - `T1 (3x6 np.array[float])`: The transformation matrix that converts
        strains and curvatures from the local xsect origin to the reference
        axis.
    - `T2 (3x6 np.array[float])`: The transformation matrix that converts
        forces and moments from the local xsect origin to the reference
        axis.
    - `x_m (1x3 np.array[float])`: Center of mass of the cross-section about in
        the local xsect CSYS
    - `M (6x6 np.array[float])`: This mass matrix relays linear and angular
        velocities to linear and angular momentum of the cross-section.


    :Methods:

    - `resetResults`: This method resets all results (displacements, strains
        and stresse) within the elements used by the cross-section object.
    - `calcWarpEffects`: Given applied force and moment resultants, this method
        calculates the warping displacement, 3D strains and 3D stresses
        within the elements used by the cross-section.
    - `printSummary`: This method is used to print characteristic attributes of
        the object. This includes the elastic, shear and mass centers, as
        well as the stiffness matrix and mass matrix.
    - `plotRigid`: This method plots the rigid cross-section shape, typically
        in conjunction with a full beam model.
    - `plotWarped`: This method plots the warped cross-section including a
        contour criteria, typically in conjuction with the results of the
        displacement of a full beam model.

    """
    def __init__(self,XID,mesh=None,**kwargs):
        """Instantiates a cross-section object.

        The constructor for the class is effectively responsible for creating
        the 2D desretized mesh of the cross-section. It is important to note
        that while meshing technically occurs in the constructor, the work is
        handeled by another class altogether. While not
        computationally heavily intensive in itself, it is responsible for
        creating all of the framework for the cross-sectional analysis.

        :Args:

        - `XID (int)`: The cross-section integer identifier.
        - `Airfoil (obj)`: An airfoil object used to determine the OML shape of
            the cross-section.
        - `xdim (1x2 array[float])`: The non-dimensional starting and stoping
            points of the cross-section. In other words, if you wanted to
            have your cross-section start at the 1/4 chord and run to the
            3/4 chord of your airfoil, xdim would look like xdim=[0.25,0.75]
        - `laminates (1xN array[obj])`: Laminate objects used to create the
            descretized mesh surface. Do not repeat a laminate within this
            array! It will referrence this object multiple times and not
            mesh the cross-section properly then!
        - `matlib (obj)`: A material library
        - `typeXSect (str)`: The general shape the cross-section should take.
            Note that currently only a box beam profile is supported.
            More shapes and the ability to add stiffeners to the
            cross-section will come in later updates.
        - `meshSize (int)`: The maximum aspect ratio you would like your 2D
            CQUADX elements to exhibit within the cross-section.

        :Returns:

        - None

        """
        #Save the cross-section ID
        self.XID = XID
        #self.elemTypes = ['XQUAD4','XQUAD6','XQUAD8','XQUAD9','XTRIA3','XTRIA6']
        # Save the cross-section type:
        color = kwargs.pop('color',np.append(np.random.rand(3),[1],axis=0))
        self.color = color
        self.normal_vector = np.array([0.,0.,1.])
        self.typeXSect = kwargs.pop('typeXSect','solidBox')

        if mesh==None:
            mesh = Mesh(self.typeXSect,**kwargs)
        self.mesh = mesh
        self.elemDict = mesh.elemDict
        self.nodeDict = mesh.nodeDict

        # Determine Crude Mesh Centroid
        x_sum = 0.
        y_sum = 0.
        for NID, node in self.nodeDict.items():
            node.setXID(XID)
            x_sum += node.x[0]
            y_sum += node.x[1]
        xavg = x_sum/len(self.nodeDict.items())
        yavg = y_sum/len(self.nodeDict.items())

        self.xtransl = -xavg
        self.ytransl = -yavg

        for NID, node in self.nodeDict.items():
            node.translate(self.xtransl,self.ytransl)

        xmax = -1e6
        xmin = 1e6
        ymax = -1e6
        ymin = 1e6
        for NID, node in self.nodeDict.items():
            if node.x[0]>xmax:
                xmax = node.x[0]
            if node.x[0]<xmin:
                xmin = node.x[0]
            if node.x[1]>ymax:
                ymax = node.x[1]
            if node.x[1]<ymin:
                ymin = node.x[1]
        for EID, elem in self.elemDict.items():
            elem.setXID(XID)
            elem.initializeElement()
            elem.color = color
        self.scale = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)
        self.visualization = VisualModel()
        self.plotRigid()
        self.analyzed=False
        self.refAxis = np.array([0.,0.,0.])
        self.area = 0.

        # Establish objects for cross-section locations

    def translateSection(self,x,y):
        """
        This method translates the cross-section mesh for improved
        cross-sectional analysis convergence.
        """
        print('Translating section {} by x={}, y={}'.format(self.XID,x,y))
        for NID, node in self.nodeDict.items():
            node.translate(-self.xtransl+x,-self.ytransl+y)
        for EID, elem in self.elemDict.items():
            elem.initializeElement()
        self.xtransl = x
        self.ytransl = y
        self.analyzed=False
        self.plotRigid()
    def transformLoads(self,loads):
        if not self.analyzed:
            print('WARNING: Transforming loads to a reference axis for a cross'\
                  ' section ({}) which has not been analyzed yet can lead to incorrect'\
                  ' transformations.'.format(self.XID))
        xref = -(self.refAxis[0]-(loads[0]+self.xtransl))
        yref = -(self.refAxis[0]-(loads[1]+self.ytransl))
        T = np.array([[1.,0.,0.,0.,0.,0.],\
                      [0.,1.,0.,0.,0.,0.],\
                      [0.,0.,1.,0.,0.,0.],\
                      [0.,0.,-yref,1.,0.,0.],\
                      [0.,0.,xref,0.,1.,0.],\
                      [yref,-xref,0.,0.,0.,1.]])
        return np.dot(T,np.array([[loads[2]],[loads[3]],[loads[4]],[loads[5]],[loads[6]],[loads[7]]]))
    def plotRigid(self,**kwargs):
        """Plots the rigid cross-section along a beam.

        This method is very useful for visually debugging a structural model.
        It will plot out the rigid cross-section in 3D space with regards to
        the reference axis.

        :Args:

        - `x (1x3 np.array[float])`: The rigid location on your beam you are
            trying to plot:
        - `beam_axis (1x3 np.array[float])`: The vector pointing in the
            direction of your beam axis.
        - `figName (str)`: The name of the figure.
        - `wireMesh (bool)`: A boolean to determine of the wiremesh outline
            should be plotted.*

        :Returns:

        - `(fig)`: Plots the cross-section in a mayavi figure.

        .. Note:: Because of how the mayavi wireframe keyword works, it will
        apear as though the cross-section is made of triangles as opposed to
        quadrilateras. Fear not! They are made of quads, the wireframe is just
        plotted as triangles.

        """
        vertices = ()
        edges = ()
        surfaces = ()
        CSYSs = []
        normals = []
        color = (tuple(self.color),)
        offset = 0
        LCIDs = [-1]
        for EID, elem in self.elemDict.items():
            temp_coords, temp_u_warp, temp_edges, temp_surfaces, \
                temp_contour = elem.getGlData(LCIDs,offset=offset)
            vertices += (temp_coords)
            edges += (temp_edges)
            surfaces += (temp_surfaces)
            offset += len(temp_coords)
            CSYSs += [elem.CSYS]
            normals += [elem.normal]
        self.colors = np.array(color*len(vertices))
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        self.surfaces = np.array(surfaces)
        self.CSYSs = CSYSs
        self.normals = normals


    def xSectionAnalysis(self,**kwargs):
        """Analyzes an initialized corss-section.

        This is the main workhorse of the class. This method assembles the
        finite element model generated using the meshing class, and solve the
        HIGH dimensional equilibrium equations associated with the cross-
        section. In doing so, it generates the warping displacement, the
        section strain, and the gradient of the warping displacement along the
        beam axis as a function of force-moment resultants. With these three
        things, the 3D strains->stresses can be recovered.

        This method has been EXTENSIVELY tested and validated against
        various sources (see theory guide for more info). Since this method
        is so robust, the biggest limitation of the XSect class is what the
        mesher is capable of meshing. Finally, keep in mind that due to the
        high dimensionality of this problem, this method uses up a lot of
        resources (primarily memory). If this method is taking too many
        resources, choose a larger aspect ratio for your XSect initialization.

        :Args:

        - `ref_ax (str or 1x2 array[float])`: Currently there are two supported
            input types for this class. The first is the are string key-words.
            These are 'shearCntr', 'massCntr', and 'origin'. Currently
            'shearCntr' is the default value. Also suported is the ability to
            pass a length 2 array containing the x and y coordinates of the
            reference axis relative to the origin. This would take the form of:
            ref_ax=[1.,3.] to put the reference axis at x,y = 1.,3.

        :Returns:

        - None

        """
        print('\n\nBeggining cross-sectional analysis on section {}...'.format(self.XID))
        t0 = time.time()
        # Initialize the reference axis:
        ref_ax = kwargs.pop('ref_ax','shearCntr')
        tol = kwargs.pop('tol',1e-12)
        print('Selected Tolerance: {}'.format(tol))
        print('Selected reference Axis: {}'.format(ref_ax))
        # Create local reference to the node dictionary
        nodeDict = self.nodeDict
        # Create local reference to the element dictionary
        elemDict = self.elemDict
        # Initialize the D matrix, responsible for decoupling rigid cross-
        # section displacement from warping cross-section displacement
        nd = 3*len(nodeDict.keys())
        NIDs = list(nodeDict.keys())
        dataD = []
        rowsD = []
        columnsD = []

        Xmeshval = []
        Ymeshval = []
        # Create provided local to global node map
        nodeMap = {}
        for i in range(0,len(NIDs)): #TODO merge this code into next for loop (are the same)
            nodeMap[NIDs[i]] = i
        self.nodeMap = nodeMap
        #D = lil_matrix((6,nd), dtype=np.float64)
        for i in range(0,len(nodeDict.keys())):
            NID = NIDs[i]
            tmpNode = nodeDict[NID]
            tempx = tmpNode.x[0]
            tempy = tmpNode.x[1]
            Xmeshval += [tempx]
            Ymeshval += [tempy]
            dataD += [1,1,1,tempy,-tempx,-tempy,tempx]
            columnsD += [0,1,2,3,4,5,5]
            rowsD += [3*i,3*i+1,3*i+2,3*i+2,3*i+2,3*i,3*i+1]
#            D[:,3*i:3*i+3] = lil_matrix(np.array([[1,0,0],\
#                                       [0,1,0],\
#                                       [0,0,1],\
#                                       [0,0,tempy],\
#                                       [0,0,-tempx],\
#                                       [-tempy,tempx,0]]))
        dx = max(Xmeshval)-min(Xmeshval)
        dy = max(Ymeshval)-min(Ymeshval)
        D = coo_matrix((dataD, (rowsD, columnsD)), shape=(nd, 6))
        #D = D.T
        # Initialize Matricies used in solving the equilibruim equations:
        Tr = coo_matrix( ( (-1,1), ((0,1), (4,3)) ), shape=(6, 6))
        print('Creating cross-section submatricies...')
        t1 = time.time()
#        A = np.zeros((6,6))
#        E = np.zeros((nd,nd))
#        L = np.zeros((nd,6))
#        R = np.zeros((nd,6))
#        C = np.zeros((nd,nd))
#        M = np.zeros((nd,nd))
        Z6 = coo_matrix((6,6))
        A = coo_matrix((6,6))
        # Initialize the cross-section mass per unit length
        m = 0.
        # Initialize the first mass moment of inertia about x
        xm = 0.
        # Initialize the first mass moment of inertia about y
        ym = 0.
        #for i in range(0,len(elemDict.keys())):
        # For all elements in the cross-section mesh
        rowsRL = []
        columnsRL = []
        dataR = []
        dataL = []
        rowsECM = []
        columnsECM = []
        dataE = []
        dataC = []
        dataM = []
        dataA = []
        rowsA = []
        columnsA = []
        for EID, elem in elemDict.items():
            #Select the element
            #tempElem = elemDict[i]
            # Get the NIDs reference by the element
            tempNodes = elem.NIDs
#            print(EID)
#            print(tempNodes)
            elnd = elem.nd
            # Update the cross-section mass
            emass = elem.mass
            m += emass
            # Update the first mass moment of ineratia about x
            xm+= emass*elem.x(0.,0.)
            # Update the first mass moment of ineratia about y
            ym+= emass*elem.y(0.,0.)
            # If the 2D element is a CQUADX

            # Create local references to the element equilibrium matricies
            #A = A + csr_matrix(elem.Ae)
            dataA += elem.Aeflat
            rowsA += [0]*6+[1]*6+[2]*6+[3]*6+[4]*6+[5]*6
            columnsA += [0,1,2,3,4,5]*6
#                Re = elem.Re
#                Ee = elem.Ee
#                Ce = elem.Ce
#                Le = elem.Le
#                Me = elem.Me
            Redat = elem.Reflat
            Eedat = elem.Eeflat
            Cedat = elem.Ceflat
            Ledat = elem.Leflat
            Medat = elem.Meflat
            # Cross-section finite element matrix assembely
            for j in range(0,len(tempNodes)):
                row = nodeMap[tempNodes[j]]
                rows_j = [3*row]*6+[3*row+1]*6+[3*row+2]*6
                columns_j = [0,1,2,3,4,5]*3
                rowsRL += rows_j
                columnsRL += columns_j
                dataR += Redat[18*j:18*j+18]
                dataL += Ledat[18*j:18*j+18]
                #dataR = dataR + list(Re[3*j:3*j+3,:].flatten())
                #dataL = dataL + list(Le[3*j:3*j+3,:].flatten())
#                    if j==0:
#                        print(len(dataR))
#                        print(len(rowsRL))
#                        print(len(columnsRL))
#                        print(dataR)
#                        print(rowsRL)
#                        print(columnsRL)
#                        raise ValueError('Test')
#                    R = R + csr_matrix((Re.flatten(), (rows_j, columns_j)), shape=(nd, 6))
#                    L = L + csr_matrix((Le.flatten(), (rows_j, columns_j)), shape=(nd, 6))
#                    R[3*row:3*row+3,:] = R[3*row:3*row+3,:] + Re[3*j:3*j+3,:]
#                    L[3*row:3*row+3,:] = L[3*row:3*row+3,:] + Le[3*j:3*j+3,:]
                for k in range(0,len(tempNodes)):
                    col = nodeMap[tempNodes[k]]
                    rows_k = [3*row]*3+[3*row+1]*3+[3*row+2]*3
                    columns_k = [3*col,3*col+1,3*col+2]*3
                    rowsECM += rows_k
                    columnsECM += columns_k
                    dataE += Eedat[3*elnd*j+3*k:3*elnd*j+3*k+3]+\
                        Eedat[3*elnd*j+3*k+elnd:3*elnd*j+3*k+3+elnd]+\
                        Eedat[3*elnd*j+3*k+2*elnd:3*elnd*j+3*k+3+2*elnd]
                    dataC += Cedat[3*elnd*j+3*k:3*elnd*j+3*k+3]+\
                        Cedat[3*elnd*j+3*k+elnd:3*elnd*j+3*k+3+elnd]+\
                        Cedat[3*elnd*j+3*k+2*elnd:3*elnd*j+3*k+3+2*elnd]
                    dataM += Medat[3*elnd*j+3*k:3*elnd*j+3*k+3]+\
                        Medat[3*elnd*j+3*k+elnd:3*elnd*j+3*k+3+elnd]+\
                        Medat[3*elnd*j+3*k+2*elnd:3*elnd*j+3*k+3+2*elnd]
        A = coo_matrix((dataA, (rowsA, columnsA)), shape=(6, 6))
        R = coo_matrix((dataR, (rowsRL, columnsRL)), shape=(nd, 6))
        L = coo_matrix((dataL, (rowsRL, columnsRL)), shape=(nd, 6))
        E = coo_matrix((dataE, (rowsECM, columnsECM)), shape=(nd, nd))
        C = coo_matrix((dataC, (rowsECM, columnsECM)), shape=(nd, nd))
        M = coo_matrix((dataM, (rowsECM, columnsECM)), shape=(nd, nd))
        # Cross-section matricies currently not saved to xsect object to save
        # memory.
        self.A = A
        self.R = R
        self.E = E
        self.C = C
        self.L = L
        self.Mx = M
        self.D = D

        # SOLVING THE EQUILIBRIUM EQUATIONS
        # Assemble state matrix for first equation
        EquiA1 = csr_matrix(vstack((hstack((E,R,D)),hstack((R.T,A,Z6)),\
                                        hstack((D.T,Z6,Z6)))))
        self.EquiA = EquiA1
        # Assemble solution vector for first equation
        Equib1 = np.vstack((np.zeros((nd,6)),Tr.T.toarray(),Z6.toarray()))
        t2 = time.time()
        print('Finished creating sub-matrices, time taken: %4.4f' %(t2-t1))
        print('Degrees of freedom: {}'.format(EquiA1.shape[0]))
        self.Equib1 = Equib1

        tolerance = tol
        maxiter = 1000
        solver='scipy'
        if solver=='pyamg':
            sol1 = spsolve(EquiA1,Equib1)
            # res1 = []
            # res2 = []
            # res3 = []
            # res4 = []
            # res5 = []
            # res6 = []
            # ml = smoothed_aggregation_solver(EquiA1,levels)
            # print(ml)
            # sol1_1 = ml.solve(Equib1[:,0], tol=tolerance,residuals=res1,maxiter=maxiter).T
            # sol1_2 = ml.solve(Equib1[:,1], tol=tolerance,residuals=res2,maxiter=maxiter).T
            # sol1_3 = ml.solve(Equib1[:,2], tol=tolerance,residuals=res3,maxiter=maxiter).T
            # sol1_4 = ml.solve(Equib1[:,3], tol=tolerance,residuals=res4,maxiter=maxiter).T
            # sol1_5 = ml.solve(Equib1[:,4], tol=tolerance,residuals=res5,maxiter=maxiter).T
            # sol1_6 = ml.solve(Equib1[:,5], tol=tolerance,residuals=res6,maxiter=maxiter).T
            # sol1 = np.vstack((sol1_1,sol1_2,sol1_3,sol1_4,sol1_5,sol1_6)).T
            # np.savetxt('xsection_residuals_1.csv',np.array([res1]).T,delimiter=',')
            # np.savetxt('xsection_residuals_2.csv',np.array([res2]).T,delimiter=',')
            # np.savetxt('xsection_residuals_3.csv',np.array([res3]).T,delimiter=',')
            # np.savetxt('xsection_residuals_4.csv',np.array([res4]).T,delimiter=',')
            # np.savetxt('xsection_residuals_5.csv',np.array([res5]).T,delimiter=',')
            # np.savetxt('xsection_residuals_6.csv',np.array([res6]).T,delimiter=',')
        else:
            sol1_1 = np.matrix(minres(EquiA1,Equib1[:,0],tol=tolerance)[0]).T
            sol1_2 = np.matrix(minres(EquiA1,Equib1[:,1],tol=tolerance)[0]).T
            sol1_3 = np.matrix(minres(EquiA1,Equib1[:,2],tol=tolerance)[0]).T
            sol1_4 = np.matrix(minres(EquiA1,Equib1[:,3],tol=tolerance)[0]).T
            sol1_5 = np.matrix(minres(EquiA1,Equib1[:,4],tol=tolerance)[0]).T
            sol1_6 = np.matrix(minres(EquiA1,Equib1[:,5],tol=tolerance)[0]).T
            sol1 = np.hstack((sol1_1,sol1_2,sol1_3,sol1_4,sol1_5,sol1_6))
        # except RuntimeError:
        #     print('The problem is ill-conditioned. Attempting solution '
        #           'using iterative approach.')
        #     sol1_1 = np.matrix(gcrotmk(EquiA1,Equib1[:,0],tol=tolerance)[0]).T
        #     sol1_2 = np.matrix(gcrotmk(EquiA1,Equib1[:,1],tol=tolerance)[0]).T
        #     sol1_3 = np.matrix(gcrotmk(EquiA1,Equib1[:,2],tol=tolerance)[0]).T
        #     sol1_4 = np.matrix(gcrotmk(EquiA1,Equib1[:,3],tol=tolerance)[0]).T
        #     sol1_5 = np.matrix(gcrotmk(EquiA1,Equib1[:,4],tol=tolerance)[0]).T
        #     sol1_6 = np.matrix(gcrotmk(EquiA1,Equib1[:,5],tol=tolerance)[0]).T
        #     sol1 = np.hstack((sol1_1,sol1_2,sol1_3,sol1_4,sol1_5,sol1_6))

        # Recover gradient of displacement as a function of force and moment
        # resutlants
        dXdz = sol1[0:nd,:]
        self.dXdz = sol1[0:nd,:]
        # Save the gradient of section strains as a function of force and
        # moment resultants
        self.dYdz = sol1[nd:nd+6,:]
        # Set up the first of two solution vectors for second equation
        Equib2_1 = vstack((hstack((-(C-C.T),L))\
            ,hstack((-L.T,Z6)),csr_matrix((6,nd+6),dtype=np.float64)))
        # Set up the second of two solution vectors for second equation
        Equib2_2 = vstack((csr_matrix((nd,6),dtype=np.float64),eye(6,6),Z6))
        Equib2 = csc_matrix(Equib2_1*csr_matrix(sol1[0:nd+6,:])+Equib2_2)
        #del Equib2_1
        #del Equib2_2
        self.Equib2 = Equib2

        if solver=='pyamg':
            # sol2_1 = ml.solve(Equib2[:,0].toarray()).T
            # sol2_2 = ml.solve(Equib2[:,1].toarray()).T
            # sol2_3 = ml.solve(Equib2[:,2].toarray()).T
            # sol2_4 = ml.solve(Equib2[:,3].toarray()).T
            # sol2_5 = ml.solve(Equib2[:,4].toarray()).T
            # sol2_6 = ml.solve(Equib2[:,5].toarray()).T
            # sol2 = np.vstack((sol2_1,sol2_2,sol2_3,sol2_4,sol2_5,sol2_6)).T
            sol2 = spsolve(EquiA1,Equib2)
        else:
            sol2_1 = np.matrix(minres(EquiA1,Equib2[:,0].toarray(),tol=tolerance)[0]).T
            sol2_2 = np.matrix(minres(EquiA1,Equib2[:,1].toarray(),tol=tolerance)[0]).T
            sol2_3 = np.matrix(minres(EquiA1,Equib2[:,2].toarray(),tol=tolerance)[0]).T
            sol2_4 = np.matrix(minres(EquiA1,Equib2[:,3].toarray(),tol=tolerance)[0]).T
            sol2_5 = np.matrix(minres(EquiA1,Equib2[:,4].toarray(),tol=tolerance)[0]).T
            sol2_6 = np.matrix(minres(EquiA1,Equib2[:,5].toarray(),tol=tolerance)[0]).T
            sol2 = np.hstack((sol2_1,sol2_2,sol2_3,sol2_4,sol2_5,sol2_6))
        # try:
        #     #sol2 = dsolve.spsolve(EquiA1, Equib2)
        #     #sol2 = sol2.todense()
        #     # sol2_1 = ml.solve(Equib2[:,0].toarray()).T
        #     # sol2_2 = ml.solve(Equib2[:,1].toarray()).T
        #     # sol2_3 = ml.solve(Equib2[:,2].toarray()).T
        #     # sol2_4 = ml.solve(Equib2[:,3].toarray()).T
        #     # sol2_5 = ml.solve(Equib2[:,4].toarray()).T
        #     # sol2_6 = ml.solve(Equib2[:,5].toarray()).T
        #     # sol2 = np.vstack((sol2_1,sol2_2,sol2_3,sol2_4,sol2_5,sol2_6)).T
        #     # raise RuntimeError('Here')

        # except RuntimeError:
        #     sol2_1 = np.matrix(gcrotmk(EquiA1,Equib2[:,0].toarray(),tol=tolerance)[0]).T
        #     sol2_2 = np.matrix(gcrotmk(EquiA1,Equib2[:,1].toarray(),tol=tolerance)[0]).T
        #     sol2_3 = np.matrix(gcrotmk(EquiA1,Equib2[:,2].toarray(),tol=tolerance)[0]).T
        #     sol2_4 = np.matrix(gcrotmk(EquiA1,Equib2[:,3].toarray(),tol=tolerance)[0]).T
        #     sol2_5 = np.matrix(gcrotmk(EquiA1,Equib2[:,4].toarray(),tol=tolerance)[0]).T
        #     sol2_6 = np.matrix(gcrotmk(EquiA1,Equib2[:,5].toarray(),tol=tolerance)[0]).T
        #     sol2 = np.hstack((sol2_1,sol2_2,sol2_3,sol2_4,sol2_5,sol2_6))

        X = sol2[0:nd,0:6]
        # Store the warping displacement as a funtion of force and moment
        # resultants
        self.X = X
        # Store the section strain as a function of force and moment resultants
        Y = sol2[nd:nd+6,0:6]
        self.Y = Y
        #Solve for the cross-section compliance
        #comp1 = np.vstack((X,dXdz,Y))
        #comp2 = np.vstack((np.hstack((E,C,R)),np.hstack((C.T,M,L)),np.hstack((R.T,L.T,A))))
        #F = np.dot(comp1.T,np.dot(comp2,comp1))
        #del comp2
        Xcompr = csr_matrix(X)
        Ycompr = csr_matrix(Y)
        dXdzcompr = csr_matrix(dXdz)
        t1 = E*Xcompr+C*dXdzcompr+R*Ycompr
        t2 = C.T*Xcompr+M*dXdzcompr+L*Ycompr
        t3 = R.T*Xcompr+L.T*dXdzcompr+A*Ycompr
        F = Xcompr.T*t1+dXdzcompr.T*t2+Ycompr.T*t3
        #print(F)
        F = F.toarray()
        t3 = time.time()
        print('Cross-sectional analysis complete. Time taken = %4.4f' %(t3-t0))
        # Store the compliance matrix taken about the xsect origin
        self.F_raw = F
        #print(F)
        self.analyzed=True
        # Store the stiffness matrix taken about the xsect origin
        self.K_raw = np.linalg.inv(F)
        # Calculate the tension center
        self.xt = (-F[2,3]*F[3,4]+F[3,3]*F[2,4])/(F[3,3]*F[4,4]-F[3,4]**2)
        self.yt = (-F[2,3]*F[4,4]+F[3,4]*F[2,4])/(F[3,3]*F[4,4]-F[3,4]**2)
        # Calculate axis about which bedning is decoupled
        if np.abs(self.K_raw[3,4])<0.1:
            self.bendAxes = np.array([[1.,0.,0.,],[0.,1.,0.]])
        else:
            trash,axes = linalg.eig(np.array([[self.K_raw[3,3],self.K_raw[3,4]],\
                        [self.K_raw[4,3],self.K_raw[4,4]]]))
            self.bendAxes = np.array([[axes[0,0],axes[1,0],0.,],[axes[0,1],axes[1,1],0.]])
        # Calculate the location of the shear center neglecting the bending
        # torsion coupling contribution:

        # An error tolerance of 1% is chosen as the difference between shear
        # center locations at the beggining and end of the non-dimensional beam
        es = 1./100
        z = 1.
        L = 1.
        xs = (-F[5,1]+F[5,3]*(L-z))/F[5,5]
        ys = (F[5,0]+F[5,4]*(L-z))/F[5,5]
        xsz0 = (-F[5,1]+F[5,3]*(L))/F[5,5]
        ysz0 = (F[5,0]+F[5,4]*(L))/F[5,5]
        eax = xs-xsz0
        eay = ys-ysz0
        if eax>dx*es or eay>dy*es:
            print('CAUTION: The shear center does not appear to be a cross-'\
                'section property, and will vary along the length of the beam.')
        self.xs = xs
        self.ys = ys
        # Calculate the mass center of the cross-section
        self.x_m = np.array([xm/m,ym/m,0.])
        self._m = m
        self.setReferenceAxis(ref_ax,override=True)
        self.analyzed=True
    def setReferenceAxis(self,ref_ax,override=False):
        """Sets the reference axis of the cross-section. This is the point
        about which loads are applied and DOF are enforced.
        """
        if not self.analyzed and not override:
            raise ValueError('A cross-section must first be analyzed before the'
                             ' reference axis can be set.')
        Ixx=0.
        Ixy=0.
        Iyy=0.
        area=0.
        nodeMap = self.nodeMap
        X = self.X
        Y = self.Y
        dXdz = self.dXdz
        m = self._m
        if ref_ax=='shearCntr':
            self.refAxis = np.array([self.xs,self.ys,0.])
            xref = -self.refAxis[0]
            yref = -self.refAxis[1]
        elif ref_ax=='massCntr':
            self.refAxis = np.array([self.x_m[0],self.x_m[1],0.])
            xref = -self.refAxis[0]
            yref = -self.refAxis[1]
        elif ref_ax=='tensionCntr':
            self.refAxis = np.array([self.xt,self.yt,0.])
            xref = -self.refAxis[0]
            yref = -self.refAxis[1]
        else:
            if len(ref_ax)==2:
                self.refAxis = np.array([ref_ax[0],ref_ax[1],0.])
                xref = -self.refAxis[0]
                yref = -self.refAxis[1]
            else:
                raise ValueError('You entered neither a supported reference axis'\
                'keyword, nor a valid length 2 array containing the x and y'\
                'beam axis reference coordinates for the cross-section.')
        # Strain reference axis transformation
        self.T1 = np.array([[1.,0.,0.,0.,0.,-yref],[0.,1.,0.,0.,0.,xref],\
            [0.,0.,1.,yref,-xref,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],\
            [0.,0.,0.,0.,0.,1.]])
        # Force reference axis transformation
        self.T2 = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],\
            [0.,0.,1.,0.,0.,0.],[0.,0.,-yref,1.,0.,0.],[0.,0.,xref,0.,1.,0.],\
            [yref,-xref,0.,0.,0.,1.]])
        self.F = np.dot(np.linalg.inv(self.T1),np.dot(self.F_raw,self.T2))
        self.K = np.dot(np.linalg.inv(self.T2),np.dot(self.K_raw,self.T1))

        #######################################################################
        # Reset all element cross-section matricies to free up memory
        for EID, elem in self.elemDict.items():
            nd = elem.nd
            #elem.clearXSectionMatricies()
            # Initialize Guass points for integration
            etas = elem.etas_int#np.array([-1,1])*np.sqrt(3)/3
            xis = elem.xis_int#np.array([-1,1])*np.sqrt(3)/3
            w_etas = elem.w_etas_int
            w_xis = elem.w_xis_int
            # Calculate the second mass moments of inertia about the reference
            # axis
            for k in range(0,np.size(xis)):
                for l in range(0,np.size(etas)):
                    Jdet, trash = elem.Jdet_inv(etas[l],xis[k])
                    #Jmat = elem._J(etas[l],xis[k])
                    #Jdet = abs(np.linalg.det(Jmat))
                    #Compute cross-section areas
                    area += Jdet*w_etas[l]*w_xis[k]*elem.quadFactor
                    # Add to the cross-section second mass moments of inertia
                    Ixx+=elem.rho*Jdet*w_etas[l]*w_xis[k]*elem.quadFactor*(elem.y(etas[l],xis[k])-self.refAxis[1])**2
                    Iyy+=elem.rho*Jdet*w_etas[l]*w_xis[k]*elem.quadFactor*(elem.x(etas[l],xis[k])-self.refAxis[0])**2
                    Ixy+=elem.rho*Jdet*w_etas[l]*w_xis[k]*elem.quadFactor*(elem.y(etas[l],xis[k])-\
                                                                           self.refAxis[1])*(elem.x(etas[l],xis[k])-self.refAxis[0])
            # Initialize the element warping vector for strain calc
            Xelem = np.zeros((nd,6))
            # Initialize the element warping grad vector for strain calc
            dXdzelem = np.zeros((nd,6))
            # For all nodes in the element
            for j in range(0,int(nd/3)):
                row = nodeMap[elem.NIDs[j]]
                # Save warping displacement
                Xelem[3*j:3*j+3,:] = X[3*row:3*row+3,:]
                # Save warping gradient
                dXdzelem[3*j:3*j+3,:] = dXdz[3*row:3*row+3,:]
            # Initialize strain vectors
            elem.f2disp = Xelem
            # Initialize Xis (strain sampling points)
            xis = elem.xis_recov
            # Initialize Etas (strain sampling points)
            etas = elem.etas_recov
            f2strn = np.zeros((6*len(xis),6))
            # Initialize stress vectors
            f2sig = np.zeros((6*len(xis),6))
            # Calculate Strain
            S = np.zeros((6,3));S[3,0]=1;S[4,1]=1;S[5,2]=1
            for j in range(0,len(xis)):
                # Initialize S:
                # Calculate Z at the corner:
                Z = elem.Z(etas[j],xis[j])
                # Calculate the Jacobian at the element corner:
                Jdet, Jmatinv  = elem.Jdet_inv(etas[j],xis[j])
                # Calculate the inverse of the Jacobian
                #Jmatinv = np.linalg.inv(tmpJ)
                # Initialize part of the strain displacement matrix
                Bxi = np.zeros((6,3))
                Bxi[0,0] = Bxi[2,1] = Bxi[3,2] = Jmatinv[0,0]
                Bxi[1,1] = Bxi[2,0] = Bxi[4,2] = Jmatinv[1,0]
                # Initialize part of the strain displacement matrix
                Beta = np.zeros((6,3))
                Beta[0,0] = Beta[2,1] = Beta[3,2] = Jmatinv[0,1]
                Beta[1,1] = Beta[2,0] = Beta[4,2] = Jmatinv[1,1]
                # Assemble the full strain displacement matrix
                BN = np.dot(Bxi,elem.dNdxi(etas[j],xis[j])) +\
                                       np.dot(Beta,elem.dNdeta(etas[j],xis[j]))
                # Initialize shape function displacement matrix
                N = elem.Nmat(etas[j],xis[j])
                # Calculate the 3D strain state
                tmpf2strn = np.dot(S,np.dot(Z,Y))+\
                    np.dot(BN,Xelem)+np.dot(S,np.dot(N,dXdzelem))
                f2strn[6*j:6*j+6,:] = tmpf2strn
                # Calculate the 3D stress state in the cross-section CSYS
                f2sig[6*j:6*j+6,:] = np.dot(elem.Q,tmpf2strn)
            # Save the displacement vector of the element nodes
            elem.f2strn = f2strn
            # Save the strain states at all 4 corners for the element
            elem.f2sig = f2sig
            # Save the forces applied to the beam nodes
        self.area = area
        # Assemble cross-section mass matrix
        self.M = np.array([[m,0.,0.,0.,0.,-m*(self.x_m[1]-self.refAxis[1])],\
                           [0.,m,0.,0.,0.,m*(self.x_m[0]-self.refAxis[0])],\
                           [0.,0.,m,m*(self.x_m[1]-self.refAxis[1]),-m*(self.x_m[0]-self.refAxis[0]),0.],\
                           [0.,0.,m*(self.x_m[1]-self.refAxis[1]),Ixx,-Ixy,0.],\
                           [0.,0.,-m*(self.x_m[0]-self.refAxis[0]),-Ixy,Iyy,0.],\
                           [-m*(self.x_m[1]-self.refAxis[1]),m*(self.x_m[0]-self.refAxis[0]),0.,0.,0.,Ixx+Iyy]])

    def resetResults(self):
        """Resets displacements, stress and strains within an xsect

        This method clears all results (both warping, stress, and strain)
        within the elements in the xsect object.

        :Args:

        - None

        :Returns:

        - None

        """
        # For all elements within the cross-section
        for EID, elem in self.elemDict.items():
            # Clear the results
            elem.resetResults()
    def calcWarpEffects(self,LCID,F,**kwargs):
        """Calculates displacements, stresses, and strains for applied forces

        The second most powerful method of the XSect class. After an analysis
        is run, the FEM class stores force and moment resultants within the
        beam element objects. From there, warping displacement, strain and
        stress can be determined within the cross-section at any given location
        within the beam using this method. This method will take a while though
        as it has to calculate 4 displacements and 24 stresses and strains for
        every element within the cross-section. Keep that in mind when you are
        surveying your beam or wing for displacements, stresses and strains.

        :Args:

        - `force (6x1 np.array[float])`: This is the internal force and moment
            resultant experienced by the cross-section.

        :Returns:

        - None

        """
        print('Loading cross-section {} with LCID {}...'.format(self.XID,LCID))
        # Initialize the applied force
        stress = kwargs.pop('stress',True)
        strain = kwargs.pop('strain',True)
        disp = kwargs.pop('disp',True)
        frc = np.reshape(np.array(F),(6,1))
        # Calculate the force applied at the origin of the cross-section
        th = np.dot(np.linalg.inv(self.T2),frc)
        if stress:
            for EID, elem in self.elemDict.items():
                #if not LCID in elem.Sig.keys() or LCID==0:
                elem.calcStress(LCID,th)
        if strain:
            for EID, elem in self.elemDict.items() or LCID==0:
                #if not LCID in elem.Eps.keys():
                elem.calcStrain(LCID,th)
        if disp:
            for EID, elem in self.elemDict.items() or LCID==0:
                #if not LCID in elem.U.keys():
                elem.calcDisp(LCID,th)
        print('Finished loading cross-section {} with LCID {}'.format(self.XID,LCID))

    def plotWarped(self,LCIDs,**kwargs):
        """Plots the warped cross-section along a beam.

        Once an analysis has been completed, this method can be utilized in
        order to plot the results anywhere along the beam.

        :Args:

        - `displScale (float)`: The scale by which all rotations and
            displacements will be mutliplied in order make it visually
            easier to detect displacements.
        - `x (1x3 np.array[float])`: The rigid location on your beam you are
            trying to plot:
        - `U (1x6 np.array[float])`: The rigid body displacements and rotations
            experienced by the cross-section.
        - `beam_axis (1x3 np.array[float])`: The vector pointing in the
            direction of your beam axis.
        - `contour (str)`: Determines what value is to be plotted during as a
            contour in the cross-section.
        - `figName (str)`: The name of the figure.
        - `wireMesh (bool)`: A boolean to determine of the wiremesh outline
            should be plotted.*
        - `contLim (1x2 array[float])`: Describes the upper and lower bounds of
            contour color scale.
        - `warpScale (float)`: The scaling factor by which all warping
            displacements in the cross-section will be multiplied.

        :Returns:

        - `(fig)`: Plots the cross-section in a mayavi figure.

        """
        #try:
        # INPUT ARGUMENT INITIALIZATION
        # Select Displacement Scale
        displScale = kwargs.pop('dispScale',1.)
        # The defomation (tranltation and rotation) of the beam node and cross-section
        U = displScale*kwargs.pop('U',np.zeros(6))
        # The rotation matrix mapping the cross-section from the local frame to
        # the global frame
        RotMat = kwargs.pop('RotMat',np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
        # Show a contour
        contour = kwargs.pop('contour','')
        # Show wire mesh?
        wireMesh = kwargs.pop('mesh',False)
        # Stress Limits

        # Establish the warping scaling factor
        warpScale = kwargs.pop('warpScale',1.)
        # Establish if the colorbar should be generated:
        self.visualization.colorbar = kwargs.pop('colorbar',True)
        coords = ()
        warpDisp = ()
        edges = ()
        surfaces = ()
        contour_data = []
        offset = 0
        for EID, elem in self.elemDict.items():
            temp_coords, temp_u_warp, temp_edges, temp_surfaces, \
                temp_contour = elem.getGlData(LCIDs,contour=contour,offset=offset)
            coords += (temp_coords)
            warpDisp += (temp_u_warp)
            edges += (temp_edges)
            surfaces += (temp_surfaces)
            contour_data += temp_contour
            offset += len(temp_coords)
        self.contour = contour_data
        contLimMin = kwargs.pop('contLimMin',np.array(contour_data).min())
        if contLimMin=='':
            contLimMin = np.array(contour_data).min()
        contLimMax = kwargs.pop('contLimMax',np.array(contour_data).max())
        if contLimMax=='':
            contLimMax = np.array(contour_data).max()
        #contLim = [contLimMin,contLimMax]
        cm = pg.ColorMap(np.linspace(contLimMin,contLimMax,6),
         [(255, 255, 255, 255),
          (0., 0., 255,255),
          (0., 255, 255, 255),
          (0., 255, 0., 255),
          (1., 255, 0., 255),
          (255, 0., 0., 255),
           ])
        if contour=='':
            color = (tuple(self.color),)
            self.colors = np.array(color*len(coords))
        else:
            self.colors = cm.map(np.array(contour_data), mode='float')
        self.colormap = cm
        #print(contour_data)
        #print(self.visualization.colors)
        self.vertices = np.array(coords)+warpScale*np.array(warpDisp)
        self.edges = np.array(edges)
        self.surfaces = np.array(surfaces)
        #self.visualization.cmap = cmap
        self.contLim = [contLimMin,contLimMax]
        #except Exception as e: print(str(e))


    def printSummary(self,refAxis=True,decimals=8,**kwargs):
        """Print characterisic information about the cross-section.

        This method prints out characteristic information about the cross-
        section objects. By default, the method will print out the location of
        the reference axis, the shear, tension, and mass center. This method
        if requested will also print the stiffness and mass matricies.

        :Args:

        - `refAxis (bool)`: Boolean to determine if the stiffness matrix
            printed should be about the reference axis (True) or about the
            local xsect origin (False).
        - `stiffMat (bool)`: Boolean to determine if the stiffness matrix
            should be printed.
        - `tensCntr (bool)`: Boolean to determine if the location of the tension
            center should be printed.
        - `shearCntr (bool)`: Boolean to determine if the location of the shear
            center should be printed.
        - `massCntr (bool)`: Boolean to determine if the location of the mass
            center should be printed.
        - `refAxisLoc (bool)`: Boolean to determine if the location of the
            reference axis should be printed.

        :Returns:

        - `(str)`: Prints out a string of information about the cross-section.

        """
        # Print xsect info:
        print('CROSS-SECTION: %d' %(self.XID))
        print('Type of cross-section is: '+self.typeXSect)
        # Print the 6x6 stiffnes matrix?
        stiffMat = kwargs.pop('stiffMat',True)
        # Print tension center?
        tensCntr = kwargs.pop('tensCntr',True)
        # Print shear center?
        shearCntr = kwargs.pop('shearCntr',True)
        # Print mass matrix?
        massMat = kwargs.pop('massMat',True)
        # Print mass center?
        massCntr = kwargs.pop('massCntr',True)
        # Print reference axis?
        refAxisLoc = kwargs.pop('refAxis',True)

        print('General Mesh Information:')
        print('Section {} contains {} xnodes, with a min and max XNIDs of {} and {} respectively.'.format(self.XID,\
              len(self.nodeDict),min(self.nodeDict.keys()),max(self.nodeDict.keys())))
        print('Section {} contains {} xelements, with a min and max XEIDs of {} and {} respectively.'.format(self.XID,\
              len(self.elemDict),min(self.elemDict.keys()),max(self.elemDict.keys())))

        if self.analyzed:
            print('Cross-sectional coordinate properties:')
            if refAxisLoc:
                print('The x,y coordinates of the reference axis are: {}, {}\n'.format(self.refAxis[0]-self.xtransl,\
                                                                                     self.refAxis[1]-self.ytransl))
            if tensCntr:
                print('The x,y coordinates of the tension center are: {}, {}\n'.format(self.xt-self.xtransl,\
                                                                                     self.yt-self.ytransl))
            if shearCntr:
                print('The x,y coordinates of the shear center are: {}, {}\n'.format(self.xs-self.xtransl,\
                                                                                     self.ys-self.ytransl))
            if massCntr:
                print('The x,y coordinates of the mass center are: {}, {}\n'.format(self.x_m[0]-self.xtransl,\
                                                                                     self.x_m[1]-self.ytransl))
            if stiffMat:
                print('Cross-section stiffness parameters:')
                if refAxis:
                    print('X-direction shear stiffness (GAKx):   {:.4e}'.format(self.K[0,0]))
                    print('Y-direction shear stiffness (GAKy):   {:.4e}'.format(self.K[1,1]))
                    print('Z-direction axial stiffness (EA):     {:.4e}'.format(self.K[2,2]))
                    print('X-direction bending stiffness (EIxx): {:.4e}'.format(self.K[3,3]))
                    print('Y-direction bending stiffness (EIyy): {:.4e}'.format(self.K[4,4]))
                    print('Z-direction torsional stiffness (GJ): {:.4e}'.format(self.K[5,5]))
                    print('Cross-sectional area (A):             {:.4e}'.format(self.area))
                    print('\n\nThe full cross-section stiffness matrix about the reference axis is:')
                    print(tabulate(np.around(self.K,decimals=decimals),tablefmt="fancy_grid"))
                else:
                    print('X-direction shear stiffness (GAKx):   {}'.format(self.K_raw[0,0]))
                    print('Y-direction shear stiffness (GAKy):   {}'.format(self.K_raw[1,1]))
                    print('Z-direction axial stiffness (EA):     {}'.format(self.K_raw[2,2]))
                    print('X-direction bending stiffness (EIxx): {}'.format(self.K_raw[3,3]))
                    print('Y-direction bending stiffness (EIyy): {}'.format(self.K_raw[4,4]))
                    print('Z-direction torsional stiffness (GJ): {}'.format(self.K_raw[5,5]))
                    print('Cross-sectional area (A):             {}'.format(self.area))
                    print('\n\nThe cross-section stiffness matrix about the xsect origin is:')
                    print(tabulate(np.around(self.K_raw,decimals=decimals),tablefmt="fancy_grid"))
            if massMat:
                print('\n\nThe cross-section mass matrix about the reference axis is:')
                print(tabulate(np.around(self.M,decimals=decimals),tablefmt="fancy_grid"))
        else:
            print('To print the cross-section properties, the section must be analyzed first.')
    def writeToFile(self,LSID):
        """Writes the object to a csv file.

        :Args:

        - None

        :Returns:

        - A string representation of the object
        """
        section_card = 'SECTIONG,{},{}'.format(self.XID,LSID)
        list_card = 'LIST,{},INT'.format(LSID)
        xeids = self.elemDict.keys()
        for xeid in xeids:
            list_card += ','+str(xeid)
        return [section_card,list_card]

class CrossSectionLibrary:

    def __init__(self):
        self.type='CrossSectionLibrary'
        self.xsectDict = {}
    def add(self,XID,mesh=None,**kwargs):
        overwrite = kwargs.pop('overwrite',False)
        if XID in self.xsectDict.keys() and not overwrite:
            raise Exception('You may not overwrite a library cross-section'+\
                ' entry without adding the optional argument overwrite=True')
        # Save material
        self.xsectDict[XID] = XSect(XID,mesh=mesh,**kwargs)
    def get(self,XID):
        if not XID in self.xsectDict.keys():
            raise KeyError('The XID provided is not linked with any cross-sections '+
                'within the supplied cross-section library.')
        return self.xsectDict[XID]
    def getIDs(self):
        return self.xsectDict.keys()
    def delete(self,XID):
        if not XID in self.xsectDict.keys():
            raise KeyError('The XID provided is not linked with any cross-sections '+
                'within the supplied cross-section library.')
        del self.xsectDict[XID]
    def printSummary(self):
        if len(self.xsectDict)==0:
            print('The cross-section library is currently empty.\n')
        else:
            print('The cross-sections are:')
            for XID, xsect in self.xsectDict.items():
                xsect.printSummary()
    def writeToFile(self,sLSID):
        """Prints summary of all cross-sections in xsecttLib

        A method used to print out tabulated summary of all of the xelements
        held within the node library object.

        :Args:

        - None

        :Returns:

        - (str): A tabulated summary of the nodes.

        """
        print_statement = []
        if len(self.xsectDict)==0:
            print('The cross-section library is currently empty.\n')
        else:
            for XID, xsect in self.xsectDict.items():
                print_statement += xsect.writeToFile(sLSID)
                sLSID += 1
        return print_statement
class TBeam:
    """Creates a Timoshenko beam finite element object.

    The primary beam finite element used by AeroComBAT, this beam element is
    similar to the Euler-Bernoulli beam finite element most are farmiliar with,
    with the exception that it has the ability to experience shear deformation
    in addition to just bending.

    :Attributes:

    - `type (str)`:String describing the type of beam element being used.
    - `U1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the first node.
    - `U2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the second node.
    - `Umode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the first node associated with the
        particular mode.
    - `Umode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the second node associated with the
        particular mode.
    - `F1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the first node.
    - `F2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the second node.
    - `Fmode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the first node associated with the
        particular mode.*
    - `Fmode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the second node associated with the
        particular mode.*
    - `xsect (obj)`: The cross-section object used to determine the beams
        stiffnesses.
    - `EID (int)`: The element ID of the beam.
    - `SBID (int)`: The associated Superbeam ID the beam object belongs to.
    - `n1 (obj)`: The first nodal object used by the beam.
    - `n2 (obj)`: The second nodal object used by the beam.
    - `Fe (12x1 np.array[float])`: The distributed force vector of the element
    - `Ke (12x12 np.array[float])`: The stiffness matrix of the beam.
    - `Keg (12x12 np.array[float])`: The geometric stiffness matrix of the
        beam. Used for beam buckling calculations.
    - `Me (12x12 np.array[float])`: The mass matrix of the beam.
    - `h (float)`: The magnitude length of the beam element.
    - `xbar (float)`: The unit vector pointing in the direction of the rigid
        beam.
    - `T (12x12 np.array[float])`:

    :Methods:

    - `printSummary`: This method prints out characteristic attributes of the
        beam finite element.
    - `plotRigidBeam`: Plots the the shape of the rigid beam element.
    - `plotDisplBeam`: Plots the deformed shape of the beam element.
    - `printInternalForce`: Prints the internal forces of the beam element for
        a given analysis set

    .. Note:: The force and moments in the Fmode1 and Fmode2 could be completely
    fictitious and be left as an artifact to fascilitate plotting of warped
    cross-sections. DO NOT rely on this information being meaningful.

    """
    def __init__(self,EID,x1,x2,xsect,nid1,nid2,chordVec=np.array([1.,0.,0.])):
        """Instantiates a timoshenko beam element.

        This method instatiates a finite element timoshenko beam element.
        Currently the beam must be oriented along the global y-axis, however
        full 3D orientation support for frames is in progress.

        :Args:

        - `x1 (1x3 np.array[float])`: The 3D coordinates of the first beam
            element node.
        - `x2 (1x3 np.array[float])`: The 3D coordinates of the second beam
            element node.
        - `xsect (obj)`: The cross-section object used to determine stiffnes
            and mass properties for the beam.
        - `EID (int)`: The integer identifier for the beam.
        - `SBID (int)`: The associated superbeam ID.
        - `nid1 (int)`: The first node ID
        - `nid2 (int)`: The second node ID

        :Returns:

        - None

        """
        # Inherit from Beam class
        self.Fe = np.zeros((12,1),dtype=float)
        self.Ke = np.zeros((12,12),dtype=float)
        self.Keg = np.zeros((12,12),dtype=float)
        self.Me = np.zeros((12,12),dtype=float)
        self.T = np.zeros((12,12),dtype=float)
        # Initialize element type
        self.type = 'Tbeam'
        self.EID=EID
        # Verify properly dimensionalized coordinates are used to create the
        # nodes.
        if (len(x1) != 3) or (len(x2) != 3):
            raise ValueError('The nodal coordinates of the beam must be 3 dimensional.')
        # Create the node objects
        self.n1 = Node(nid1,x1[0],x1[1],x1[2])
        self.n2 = Node(nid2,x2[0],x2[1],x2[2])
        self.vertices = (
                          (self.n1.x[0],self.n1.x[1],self.n1.x[2]),
                          (self.n2.x[0],self.n2.x[1],self.n2.x[2]),
                          )
        self.colors = (1.,0.,0.,1.)

        self.width = 0.05*xsect.scale
        # Solve for the length of the beam
        h = np.linalg.norm(x2-x1)
        self.h = h
        # Solve for the beam unit vector
        self.xbar = (x2-x1)/h
        # Determine the Transformation Matrix
        zVec = self.xbar
        yVec = np.cross(zVec,chordVec)/np.linalg.norm(np.cross(zVec,chordVec))
        xVec = np.cross(yVec,zVec)/np.linalg.norm(np.cross(yVec,zVec))
        Tsubmat = np.vstack((xVec,yVec,zVec))
        self.T[0:3,0:3] = Tsubmat
        self.T[3:6,3:6] = Tsubmat
        self.T[6:9,6:9] = Tsubmat
        self.T[9:12,9:12] = Tsubmat
        self.xsect = xsect
        # Create a local reference to the cross-section stiffness matrix
        K = xsect.K
        # Lines below not needed, there for visual neatness
        C11 = K[0,0];C12 = K[0,1];C13 = K[0,2];C14 = K[0,3];C15 = K[0,4];C16 = K[0,5]
        C22 = K[1,1];C23 = K[1,2];C24 = K[1,3];C25 = K[1,4];C26 = K[1,5]
        C33 = K[2,2];C34 = K[2,3];C35 = K[2,4];C36 = K[2,5]
        C44 = K[3,3];C45 = K[3,4];C46 = K[3,5]
        C55 = K[4,4];C56 = K[4,5]
        C66 = K[5,5]
        # Initialize the Element Stiffness Matrix
        self.Kel = np.array([[C11/h,C12/h,C13/h,-C12/2+C14/h,C11/2+C15/h,C16/h,-C11/h,-C12/h,-C13/h,-C12/2-C14/h,C11/2-C15/h,-C16/h],\
                          [C12/h,C22/h,C23/h,-C22/2+C24/h,C12/2+C25/h,C26/h,-C12/h,-C22/h,-C23/h,-C22/2-C24/h,C12/2-C25/h,-C26/h],\
                          [C13/h,C23/h,C33/h,-C23/2+C34/h,C13/2+C35/h,C36/h,-C13/h,-C23/h,-C33/h,-C23/2-C34/h,C13/2-C35/h,-C36/h],\
                          [-C12/2+C14/h,-C22/2+C24/h,-C23/2+C34/h,-C24+C44/h+C22*h/4,C14/2-C25/2+C45/h-C12*h/4,-C26/2+C46/h,C12/2-C14/h,C22/2-C24/h,C23/2-C34/h,-C44/h+C22*h/4,C14/2+C25/2-C45/h-C12*h/4,C26/2-C46/h],\
                          [C11/2+C15/h,C12/2+C25/h,C13/2+C35/h,C14/2-C25/2+C45/h-C12*h/4,C15+C55/h+C11*h/4,C16/2+C56/h,-C11/2-C15/h,-C12/2-C25/h,-C13/2-C35/h,-C14/2-C25/2-C45/h-C12*h/4,-C55/h+C11*h/4,-C16/2-C56/h],\
                          [C16/h,C26/h,C36/h,-C26/2+C46/h,C16/2+C56/h,C66/h,-C16/h,-C26/h,-C36/h,-C26/2-C46/h,C16/2-C56/h,-C66/h],\
                          [-C11/h,-C12/h,-C13/h,C12/2-C14/h,-C11/2-C15/h,-C16/h,C11/h,C12/h,C13/h,C12/2+C14/h,-C11/2+C15/h,C16/h],\
                          [-C12/h,-C22/h,-C23/h,C22/2-C24/h,-C12/2-C25/h,-C26/h,C12/h,C22/h,C23/h,C22/2+C24/h,-C12/2+C25/h,C26/h],\
                          [-C13/h,-C23/h,-C33/h,C23/2-C34/h,-C13/2-C35/h,-C36/h,C13/h,C23/h,C33/h,C23/2+C34/h,-C13/2+C35/h,C36/h],\
                          [-C12/2-C14/h,-C22/2-C24/h,-C23/2-C34/h,-C44/h+C22*h/4,-C14/2-C25/2-C45/h-C12*h/4,-C26/2-C46/h,C12/2+C14/h,C22/2+C24/h,C23/2+C34/h,C24+C44/h+C22*h/4,-C14/2+C25/2+C45/h-C12*h/4,C26/2+C46/h],\
                          [C11/2-C15/h,C12/2-C25/h,C13/2-C35/h,C14/2+C25/2-C45/h-C12*h/4,-C55/h+C11*h/4,C16/2-C56/h,-C11/2+C15/h,-C12/2+C25/h,-C13/2+C35/h,-C14/2+C25/2+C45/h-C12*h/4,-C15+C55/h+C11*h/4,-C16/2+C56/h],\
                          [-C16/h,-C26/h,-C36/h,C26/2-C46/h,-C16/2-C56/h,-C66/h,C16/h,C26/h,C36/h,C26/2+C46/h,-C16/2+C56/h,C66/h]])
        self.Ke = np.dot(self.T.T,np.dot(self.Kel,self.T))
        # Initialize the element distributed load vector
        self.Fe = np.zeros((12,1),dtype=float)
        # Initialize the Geometric Stiffness Matrix
        kgtmp = np.zeros((12,12),dtype=float)
        kgtmp[0,0] = kgtmp[1,1] = kgtmp[6,6] = kgtmp[7,7] = 1./h
        kgtmp[0,6] = kgtmp[1,7] = kgtmp[6,0] = kgtmp[7,1] = -1./h
        self.Kegl = kgtmp
        self.Keg = np.dot(self.T.T,np.dot(self.Kegl,self.T))
        # Initialize the mass matrix
        # Create local reference of cross-section mass matrix
        M = xsect.M
        M11 = M[0,0]
        M16 = M[0,5]
        M26 = M[1,5]
        M44 = M[3,3]
        M45 = M[3,4]
        M55 = M[4,4]
        M66 = M[5,5]
        self.Mel = np.array([[h*M11/3.,0.,0.,0.,0.,h*M16/3.,h*M11/6.,0.,0.,0.,0.,h*M16/6.],\
                            [0.,h*M11/3.,0.,0.,0.,h*M26/3.,0.,h*M11/6.,0.,0.,0.,h*M26/6.],\
                            [0.,0.,h*M11/3.,-h*M16/3.,-h*M26/3.,0.,0.,0.,h*M11/6.,-h*M16/6.,-h*M26/6.,0.],\
                            [0.,0.,-h*M16/3.,h*M44/3.,h*M45/3.,0.,0.,0.,-h*M16/6.,h*M44/6.,h*M45/6.,0.],\
                            [0.,0.,-h*M26/3.,h*M45/3.,h*M55/3.,0.,0.,0.,-h*M26/6.,h*M45/6.,h*M55/6.,0.],\
                            [h*M16/3.,h*M26/3.,0.,0.,0.,h*M66/3.,h*M16/6.,h*M26/6.,0.,0.,0.,h*M66/6.],\
                            [h*M11/6.,0.,0.,0.,0.,h*M16/6.,h*M11/3.,0.,0.,0.,0.,h*M16/6.],\
                            [0.,h*M11/6.,0.,0.,0.,h*M26/6.,0.,h*M11/3.,0.,0.,0.,h*M26/3.],\
                            [0.,0.,h*M11/6.,-h*M16/6.,-h*M26/6.,0.,0.,0.,h*M11/3.,-h*M16/3.,-h*M26/3.,0.],\
                            [0.,0.,-h*M16/6.,h*M44/6.,h*M45/6.,0.,0.,0.,-h*M16/3.,h*M44/3.,h*M45/3.,0.],\
                            [0.,0.,-h*M26/6.,h*M45/6.,h*M55/6.,0.,0.,0.,-h*M26/3.,h*M45/3.,h*M55/3.,0.],\
                            [h*M16/6.,h*M26/6.,0.,0.,0.,h*M66/6.,h*M16/3.,h*M26/3.,0.,0.,0.,h*M66/3.]])
        self.Me = np.dot(self.T.T,np.dot(self.Mel,self.T))
    def applyDistributedLoad(self,fx):
        """Applies distributed load to the element.

        Intended primarily as a private method but left public, this method,
        applies a distributed load to the finite element. Due to the nature of
        the timoshenko beam, you cannot apply a distributed moment, however you
        can apply distributed forces.

        :Args:

        - `fx (1x6 np.array[float])`: The constant distributed load applied
            over the length of the beam.

        :Returns:

        - None

        """
        h = self.h
        self.Fe = np.reshape(np.array([h*fx[0]/2,h*fx[1]/2,\
                            h*fx[2]/2,h*fx[3]/2,h*fx[4]/2,h*fx[5]/2,\
                            h*fx[0]/2,h*fx[1]/2,h*fx[2]/2,h*fx[3]/2,h*fx[4]/2,\
                            h*fx[5]/2]),(12,1))

class BeamElementLibrary:

    def __init__(self):
        self.type='BeamElementLibrary'
        self.elemDict = {}
    def addBeamElement(self,element,**kwargs):
        EID = element.EID
        overwrite = kwargs.pop('overwrite',False)
        if EID in self.elemDict.keys() and not overwrite:
            raise Exception('You may not overwrite a library beam element'+\
                ' entry without adding the optional argument overwrite=True')
        self.elemDict[EID] = element
    def getBeamElement(self,EID):
        if not EID in self.elemDict.keys():
            raise KeyError('The EID provided is not linked with any beam elements '+
                'within the beam element library.')
        return self.elemDict[EID]
    def getIDs(self):
        return self.elemDict.keys()
    def deleteCrossSection(self,EID):
        if not EID in self.elemDict.keys():
            raise KeyError('The EID provided is not linked with any beam elements '+
                'within the beam element library.')
        del self.elemDict[EID]
    def printSummary(self):
        if len(self.elemDict)==0:
            print('The beam element library is currently empty.\n')
        else:
            print('The cross-sections are:')
            for EID, elem in self.elemDict.items():
                elem.printSummary()
class Beam:
    """Create a superbeam object.

    The superbeam object is mainly to fascilitate creating a whole series of
    beam objects along  the same line.

    :Attributes:

    - `type (str)`: The object type, a 'SuperBeam'.
    - `btype (str)`: The beam element type of the elements in the superbeam.
    - `SBID (int)`: The integer identifier for the superbeam.
    - `sNID (int)`: The starting NID of the superbeam.
    - `enid (int)`: The ending NID of the superbeam.
    - `xsect (obj)`: The cross-section object referenced by the beam elements
        in the superbeam.
    - `noe (int)`: Number of elements in the beam.
    - `NIDs2EIDs (dict)`: Mapping of NIDs to beam EIDs within the superbeam
    - `x1 (1x3 np.array[float])`: The 3D coordinate of the first point on the
        superbeam.
    - `x2 (1x3 np.array[float])`: The 3D coordinate of the last point on the
        superbeam.
    - `sEID (int)`: The integer identifier for the first beam element in the
        superbeam.
    - `elems (dict)`: A dictionary of all beam elements within the superbeam.
        The keys are the EIDs and the values are the corresponding beam
        elements.
    - `xbar (1x3 np.array[float])`: The vector pointing along the axis of the
        superbeam.

    :Methods:

    - `getBeamCoord`: Returns the 3D coordinate of a point along the superbeam.
    - `printInternalForce`: Prints all internal forces and moments at every
        node in the superbeam.
    - `writeDisplacements`: Writes all displacements and rotations in the
        superbeam to a .csv
    - `getEIDatx`: Provided a non-dimensional point along the superbeam, this
        method returns the local element EID and the non-dimensional
        coordinate within that element.
    - `printSummary`: Prints all of the elements and node IDs within the beam
        as well as the coordinates of those nodes.

    """
    def __init__(self,BID,x1,x2,xsect,noe,btype='Tbeam',sNID=1,sEID=1,chordVec=np.array([1.,0.,0.])):
        """Creates a superelement object.

        This method instantiates a superelement. What it effectively does is
        mesh a line provided the starting and ending points along that line.
        Keep in mind that for now, only beams running parallel to the z-axis
        are supported.

        :Args:

        - `x1 (1x3 np.array[float])`: The starting coordinate of the beam.
        - `x2 (1x3 np.array[float])`: The ending coordinate of the beam.
        - `xsect (obj)`: The cross-section used throught the superbeam.
        - `noe (int)`: The number of elements along the beam.
        - `SBID (int)`: The integer identifier for the superbeam.
        - `btype (str)`: The beam type to be meshed. Currently only Tbeam types
            are supported.
        - `sNID (int)`: The starting NID for the superbeam.
        - `sEID (int)`: The starting EID for the superbeam.

        :Returns:

        - None

        """
        # Initialize the object type
        self.type = 'Beam'
        # Save the beam element type used within the superbeam.
        self.btype = btype
        # Save the SBID
        self.BID = BID
        self.numXSects = 5
        # Check to make sure that the superbeam length is at least 1.
        if noe<1:
            raise ValueError('The beam super-element must contain at least 1 beam element.')
        # Store the starting NID
        self.sNID = sNID
        # Store the cross-section
        self.xsect = xsect
        # Store the number of elements
        self.noe = noe
        # Store the ending node ID
        self.enid = sNID+noe
        # Initialize a dictionary with EIDs as the keys and the associated NIDs
        # as the stored values.
        self.NIDs2EIDs = coll.defaultdict(list)
        # Create an empty element dictionary
        elems = {}
        # Parameterize the non-dimensional length of the beam
        t = np.linspace(0,1,noe+1)
        # Store the SuperBeam starting coordinate
        x1 = np.array(x1)
        x2 = np.array(x2)
        self.x1 = x1
        # Store the SuperBeam ending coordinate
        self.x2 = x2
        # Determine the 'slope' of the superbeam
        self.m = x2-x1
        # Store the starting element ID
        self.sEID = sEID
        tmpsnidb = sNID
        # Check which beam type is to be used:
        if btype == 'Tbeam':
            tmpsnide = sNID+1
            # Create all the elements in the superbeam
            for i in range(0,noe):
                x0 = self.getBeamCoord(t[i])
                xi = self.getBeamCoord(t[i+1])
                # Store the element in the superbeam elem dictionary
                elems[i+sEID] = TBeam(i+sEID,x0,xi,xsect,\
                    nid1=tmpsnidb,nid2=tmpsnide,chordVec=chordVec)
                self.NIDs2EIDs[tmpsnidb] += [i+sEID]
                self.NIDs2EIDs[tmpsnide] += [i+sEID]
                tmpsnidb = tmpsnide
                tmpsnide = tmpsnidb+1
        else:
            raise TypeError('You have entered an invalid beam type.')
        self.elems = elems
        # Save the unit vector pointing along the length of the beam
        self.xbar = elems[sEID].xbar
        self.RotMat = elems[sEID].T[0:3,0:3]
#        nodes = {}
#        for i in range(0,noe+1):
#            x0 = self.getBeamCoord(t[i])
#            nodes[sNID+i] = Node(sNID+i,x0)
#        self.nodes = nodes
    def getBeamCoord(self,x_nd):
        """Determine the global coordinate along superbeam.

        Provided the non-dimensional coordinate along the beam, this method
        returns the global coordinate at that point.

        :Args:

        - `x_nd (float)`: The non-dimensional coordinate along the beam. Note
            that x_nd must be between zero and one.

        :Returns:

        - `(1x3 np.array[float])`: The global coordinate corresponding to x_nd
        """
        # Check that x_nd is between 0 and 1
        if x_nd<0. or x_nd>1.:
            raise ValueError('The non-dimensional position along the beam can'\
                'only vary between 0 and 1')
        return self.x1+x_nd*self.m
    def printInternalForce(self,**kwargs):
        """Prints the internal forces and moments in the superbeam.

        For every node within the superbeam, this method will print out the
        internal forces and moments at those nodes.

        :Args:

        - `analysis_name (str)`: The name of the analysis for which the forces
            and moments are being surveyed.

        :Returns:

        - `(str)`: Printed output expressing all forces and moments.

        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        for EID, elem in self.elems.items():
            elem.printInternalForce(analysis_name=analysis_name)
    def writeDisplacements(self,**kwargs):
        """Write internal displacements and rotations to file.

        For every node within the superbeam, this method will tabulate all of
        the displacements and rotations and then write them to a file.

        :Args:

        - `fileName (str)`: The name of the file where the data will be written.
        - `analysis_name (str)`: The name of the analysis for which the
            displacements and rotations are being surveyed.

        :Returns:

        - `fileName (file)`: This method doesn't actually return a file, rather
            it writes the data to a file named "fileName" and saves it to the
            working directory.

        """
        # Load default value for file name
        fileName = kwargs.pop('fileName','displacements.csv')
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        Return = kwargs.pop('Return',False)
        NID = np.zeros((len(self.elems)+1,1))
        nodeX = np.zeros((len(self.elems)+1,3))
        nodeDisp = np.zeros((len(self.elems)+1,6))
        i = 0
        NIDs = []
        for EID, elem in self.elems.items():
            if not elem.n1.NID in NIDs:
                NIDs+=[elem.n1.NID]
                NID[i,0] = elem.n1.NID
                nodeX[i,:] = elem.n1.x
                nodeDisp[i,:] = elem.U1[analysis_name].T
                i+=1
            if not elem.n2.NID in NIDs:
                NIDs+=[elem.n2.NID]
                NID[i,0] = elem.n2.NID
                nodeX[i,:] = elem.n2.x
                nodeDisp[i,:] = elem.U2[analysis_name].T
                i+=1
        writeData = np.hstack((NID,nodeX,nodeDisp))
        if Return:
            return writeData
        else:
            np.savetxt(fileName,writeData,delimiter=',')
    def writeForcesMoments(self,**kwargs):
        """Write internal force and moments to file.

        For every node within the superbeam, this method will tabulate all of
        the forces and moments and then write them to a file.

        :Args:

        - `fileName (str)`: The name of the file where the data will be written.
        - `analysis_name (str)`: The name of the analysis for which the
            forces and moments are being surveyed.

        :Returns:

        - `fileName (file)`: This method doesn't actually return a file, rather
            it writes the data to a file named "fileName" and saves it to the
            working directory.

        """
        fileName = kwargs.pop('fileName','forcesMoments.csv')
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        Return = kwargs.pop('Return',False)
        NID = np.zeros((len(self.elems)+1,1))
        nodeX = np.zeros((len(self.elems)+1,3))
        nodeForce = np.zeros((len(self.elems)+1,6))
        i = 0
        NIDs = []
        for EID, elem in self.elems.items():
            if not elem.n1.NID in NIDs:
                NIDs+=[elem.n1.NID]
                NID[i,0] = elem.n1.NID
                nodeX[i,:] = elem.n1.x
                nodeForce[i,:] = elem.F1[analysis_name].T
                i+=1
            if not elem.n2.NID in NIDs:
                NIDs+=[elem.n2.NID]
                NID[i,0] = elem.n2.NID
                nodeX[i,:] = elem.n2.x
                nodeForce[i,:] = elem.F2[analysis_name].T
                i+=1
        writeData = np.hstack((NID,nodeX,nodeForce))
        if Return:
            return writeData
        else:
            np.savetxt(fileName,writeData,delimiter=',')
    def getEIDatx(self,x):
        """Returns the beam EID at a non-dimensional x-location in the superbeam.

        Provided the non-dimensional coordinate along the beam, this method
        returns the global beam element EID, as well as the local non-
        dimensional coordinate within the specific beam element.

        :Args:

        - `x (float)`: The non-dimensional coordinate within the super-beam

        :Returns:

        - `EID (int)`: The EID of the element containing the non-dimensional
            coordinate provided.
        - `local_x_nd (float)`: The non-dimensional coordinate within the beam
            element associated with the provided non-dimensional coordinate
            within the beam.

        """
        '''n = len(self.elems)
        local_x_nd = 1.
        EID = max(self.elems.keys())
        for i in range(0,n):
            if x<=(float(i)/float(n)):
                EID = self.sEID+i
                local_x_nd = 1+i-n*x
                break'''
        totalLen = np.linalg.norm(self.x2-self.x1)
        xDim = x*totalLen
        for locEID, elem in self.elems.items():
            localElemDim = np.linalg.norm(np.array(np.array(elem.n2.x)-self.x1))
            if xDim<=localElemDim:
                EID = locEID
                local_x_nd = (xDim-(localElemDim-elem.h))/elem.h
                break
        return EID, local_x_nd
    def printSummary(self,decimals=8,**kwargs):
        """Prints out characteristic information about the super beam.

        This method by default prints out the EID, XID, SBID and the NIDs along
        with the nodes associated coordinates. Upon request, it can also print
        out the beam element stiffness, geometric stiffness, mass matricies and
        distributed force vector.

        :Args:

        - `nodeCoord (bool)`: A boolean to determine if the node coordinate
            information should also be printed.
        - `Ke (bool)`: A boolean to determine if the element stiffness matrix
            should be printed.
        - `Keg (bool)`: A boolean to determine if the element gemoetric
            stiffness matrix should be printed.
        - `Me (bool)`: A boolean to determine if the element mass matrix
            should be printed.
        - `Fe (bool)`: A boolean to determine if the element distributed force
            and moment vector should be printed.

        :Returns:

        - `(str)`: Printed summary of the requested attributes.

        """
        # Print the associated xsect ID
        XID = kwargs.pop('XID',False)
        # Print the number of beam elements in the superbeam
        numElements = kwargs.pop('numElements',False)
        # Determine if node coordinates should also be printed
        nodeCoord = kwargs.pop('nodeCoord',True)
        # Print the stiffness matrix
        Ke = kwargs.pop('Ke',False)
        # Print the geometric stiffness matrix
        Keg = kwargs.pop('Keg',False)
        # Print the mass matrix
        Me = kwargs.pop('Me',False)
        # Print the distributed force vector
        Fe = kwargs.pop('Fe',False)
        # Print the element summaries

        # Print the SBID
        print('Superbeam: %d' %(self.SBID))
        if XID:
            print('Cross-section: %d' %(self.XID))
        if numElements:
            print('There are %d elements in this super-beam.' %(len(self.elems)))
        for EID, elem in self.elems.items():
            elem.printSummary(nodeCoord=nodeCoord,Ke=Ke,Keg=Keg,Me=Me,Fe=Fe)

class BeamLibrary:

    def __init__(self):
        self.type='CrossSectionLibrary'
        self.beamDict = {}
        self.BeamElements = None
    def add(self,BID,x1,x2,xsect,noe,btype,sNID,sEID,chordVec,**kwargs):
        overwrite = kwargs.pop('overwrite',False)
        if BID in self.beamDict.keys() and not overwrite:
            raise Exception('You may not overwrite a library beam'+\
                ' entry without adding the optional argument overwrite=True')
        # Save material
        #try:
        self.beamDict[BID] = Beam(BID,x1,x2,xsect,noe,btype=btype,sNID=sNID,\
            sEID=sEID,chordVec=chordVec)
        print('Beam Successfully created!')
        #except Exception as e: print(str(e))

    def get(self,BID):
        if not BID in self.beamDict.keys():
            raise KeyError('The BID provided is not linked with any beam '+
                'within the beam library.')
        return self.beamDict[BID]
    def getIDs(self):
        return self.beamDict.keys()
    def delete(self,BID):
        if not BID in self.beamDict.keys():
            raise KeyError('The BID provided is not linked with any beam '+
                'within the beam library.')
        del self.beamDict[BID]
    def printSummary(self):
        if len(self.beamDict)==0:
            print('The beam library is currently empty.\n')
        else:
            print('The beams are:')
            for BID, beam in self.beamDict.items():
                beam.printSummary()
