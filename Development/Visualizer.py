from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.pgcollections import OrderedDict
import sys
import numpy as np


class VisualModel():
    def __init__(self):
        self.vertices = None
        self.edges = None
        self.surfaces = None
        self.colors = ()
        self.colors2use = ()
        self.colors2 = None
        self.cmap = None
        self.contLim = [-1.,1.]

    def display(self):
        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        background_color = pg.mkColor(255,255,255)
        w.setBackgroundColor(background_color)
        w.opts['distance'] = 20
        w.show()
        w.setWindowTitle('AeroComBAT Visualization')
        
        ax = gl.GLAxisItem()
        ax.setSize(5,5,5)
        w.addItem(ax)
        
        md = gl.MeshData(vertexes=self.vertices, faces=self.surfaces,\
                         vertexColors=self.colors)
        
        mesh = gl.GLMeshItem(meshdata=md,drawEdges=True,edgeColor=(0, 0, 0, 1))
        
        w.addItem(mesh)
#        if self.colorbar:
#            colorBar = pg.GradientLegend(size=(50, 200), offset=(15, -25))
#            colorBar.setGradient(self.cm.getGradient())
#            labels = dict([("%0.2f" % (v * (self.contLim[1]-self.contLim[0]) + self.contLim[0]), v) for v in np.linspace(0, 1, 4)])
#            colorBar.setLabels(labels)
#            w.addItem(colorBar)
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    def resetMesh(self):
        self.vertices = ()
        self.surfaces = ()
        self.colors = ()

### Start Qt event loop unless running in interactive mode.
#if __name__ == '__main__':
#    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()