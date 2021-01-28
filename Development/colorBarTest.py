import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
pg.setConfigOption("useWeave", False) # this line eliminates some strange output lines
from pyqtgraph.pgcollections import OrderedDict

Gradients = OrderedDict([
	('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
    ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
    ('summer', {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'} ),
    ('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)), (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)), (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))], 'mode': 'rgb'}),
    ('winter', {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'})
])


class Plotter:
    def __init__(self):
        pg.mkQApp()
        # self.WindowNumber = 0
        self.plot = None
        self.plots = []
        self.windows = []
    def surfMesh(self, vertexes, faces, faceValues, cmap = 'space', drawEdges=False, smooth=False):
        '''
        plots data for the triangular mesh
        Arguments:
        float numpyarray vertexes: x-y coordinates of the vertices constituting the mesh
        note: doesn't work with 3D. need some changes
        
        int numpyarray faces: each element represents numbers of verticis constituting a face
        
        float numpyarray data
        
        str cmap:
        	'space' - default cmap developed by me. kinda cool :-)
        	'thermal'
        '''
        nbplt = len(self.plots)
        # impose color scheme
        if cmap in Gradients:
        	GradiendMode = Gradients[cmap]
        else: raise TypeError('Color Map is not defined.')
        
        # compute normalized value for color calculation
        nbv = len(vertexes) # number of vertexes
        xmin = faceValues.min()
        xmax = faceValues.max()
        if xmax != xmin:
        	x = (faceValues - xmin)/(xmax - xmin)
        else: x = np.ones(len(faceValues)) * xmin
        
        # Main window containing other widgets
        win = QtGui.QWidget()
        win.setWindowTitle('isqt plot')
        self.windows.append(win)
        # leyout windget controlling layour of other widgets
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        # Widget - righthand side of the window allocated for the colorbar
        cb = pg.GraphicsLayoutWidget()
        # values axis
        ax = pg.AxisItem('left')
        if xmax != xmin:  # here we handle arrays with all values the same
        	# ax.setRange(np.round(xmin,2), np.round(xmax,2)) # to round numbers on ticks - doesn't work
        	ax.setRange(xmin, xmax)
        else:
        	ax.setRange(xmin, xmax + 0.0001)
        cb.addItem(ax)
        # Gradient Editor widget
        gw = pg.GradientEditorItem(orientation='right')
        # load predefined color gradient
        gw.restoreState(GradiendMode)
        # left side of the window for 3D data representation
        view = gl.GLViewWidget()
        
        # it's basically 
        view.setSizePolicy(cb.sizePolicy())
        # add 3D widget to the left (first column)
        layout.addWidget(view, 0, 0)
        # add colorbar to the right (second column)
        layout.addWidget(cb, 0, 1)
        # Do not allow 2nd column (colorbar) to stretch
        layout.setColumnStretch(1, 0)
        # minimal size of the colorbar
        layout.setColumnMinimumWidth(1, 120)
        # Allow 1st column (3D widget) to stretch
        layout.setColumnStretch(0, 1)
        # horizontal size set to be large to prompt colormap to a minimum size
        view.sizeHint = lambda: pg.QtCore.QSize(1700, 800)
        cb.sizeHint = lambda: pg.QtCore.QSize(100, 800)
        # this is to remove empty space between
        layout.setHorizontalSpacing (0)
        # set initial size of the window
        win.resize(800,800)
        win.show()
        
        cb.addItem(gw)
        cm = gw.colorMap()
        # colors = cm.map(x)
        colors = cm.mapToFloat(x)
        
        # create 3D array to pass it to plotting function
        vetrs3D = np.zeros([nbv,3])
        if vertexes.shape == (nbv,3): vetrs3D = vertexes
        else: 
        	print('Converting 2D to 3D')
        	vetrs3D[:,:2] = vertexes
        # compute distance to set initial camera pozition
        dst = max(vertexes[:,0].max() - vertexes[:,0].min(), vertexes[:,1].max() - vertexes[:,1].min())
        view.setCameraPosition(distance=dst)
        
        plt = gl.GLMeshItem(vertexes=vetrs3D, faces=faces, faceColors=colors, drawEdges=drawEdges, smooth=smooth)
        view.addItem(plt)
        self.plots.append(plt)

#And an example of using this stuff is:

verts = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [1, 2, 0],
    [1, 1, 1],
])
faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
])
values = np.array([0.0,2.0,3,20.0])
plt = Plotter()
plt.surfMesh(vertexes=verts, faces=faces, faceValues=values)
plt.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()