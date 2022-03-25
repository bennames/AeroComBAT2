import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.pgcollections import OrderedDict
import copy
import numpy as np
from pyqtgraph.dockarea import *
import sys
from Model import Model
import time
import ast

Gradients = OrderedDict([
    ('standard', {'ticks': [(0, (255, 255, 255, 255)),
                            (0.2, (0, 0, 255, 255)),
                            (0.4, (0, 255, 255, 255)),
                            (0.6, (0, 255, 0, 255)), 
                            (0.8, (255, 255, 0, 255)), 
                            (1, (255, 0, 0, 255))], 'mode': 'rgb'}),
])
    
class GUI:
    def __init__(self):
        self.Model = Model()
        self.Model.GUI = self
        self.printWindow = None
        self.sectionSelectionWindow = None
        
        #######################################################################
        # CREATE THE GUI
        #######################################################################
        app = QtGui.QApplication([])
        self.app = app
        # Create the window
        win = QtGui.QMainWindow()
        area = DockArea()
        win.setCentralWidget(area)
        win.resize(1000,1000)
        win.setWindowTitle('AeroComBAT')
        icon = QtGui.QIcon("AeroComBAT.jpg")
        win.setWindowIcon(icon)
        app.setWindowIcon(icon)
        
        #############
        # MENU SETUP
        ############
        #FILE
        #Load AeroComBAT file
        loadDatFile = QtGui.QAction("&Load AeroComBAT Input File",win)
        loadDatFile.triggered.connect(self.loadDat)
        #Load Nastran file
        loadNASTFile = QtGui.QAction("&Load NASTRAN Input File",win)
        loadNASTFile.triggered.connect(self.loadNASTRAN)
        #Load csv loads file
        loadCSVFile = QtGui.QAction("&Load CSV Loads File",win)
        loadCSVFile.triggered.connect(self.loadCSVLoads)
        #Print Function
        printEntity = QtGui.QAction("&Print Entity",win)
        printEntity.triggered.connect(self.printEntities)
        #Add quit functionality
        quitEntity = QtGui.QAction("&Quit",win)
        quitEntity.triggered.connect(self.AeroQuit)
        
        #CREATE OBJECTS
        createMaterial = QtGui.QAction("&Material",win)
        createMaterial.triggered.connect(self.createMaterial)
        #Create Laminate
        createLaminate = QtGui.QAction("&Laminate",win)
        createLaminate.triggered.connect(self.createLaminate)
        
        #SECTION ANALYSIS
        selectSections = QtGui.QAction("&Select Section",win)
        selectSections.triggered.connect(self.selectSection)
        
        viewCSYS = QtGui.QAction("&CSYS",win)
        viewCSYS.triggered.connect(self.showCSYS)
        viewCSYS.setStatusTip('View material coordinate systems')
        
        viewNormals = QtGui.QAction("&Element Normals",win)
        viewNormals.triggered.connect(self.showNormals)
        viewNormals.setStatusTip('View element normals')
        
        viewRefAxis = QtGui.QAction("&Reference Axis",win)
        viewRefAxis.triggered.connect(self.showRefAxis)
        viewRefAxis.setStatusTip('View the reference axis of the cross-section')
        
        viewShearCenter = QtGui.QAction("&Shear Center",win)
        viewShearCenter.triggered.connect(self.showShearCenter)
        viewShearCenter.setStatusTip('View the shear center of the cross-section')
        
        viewTensionCenter = QtGui.QAction("&Tension Center",win)
        viewTensionCenter.triggered.connect(self.showTensionCenter)
        viewTensionCenter.setStatusTip('View the tension center of the cross-section')
        
        viewMassCenter = QtGui.QAction("&Mass Center",win)
        viewMassCenter.triggered.connect(self.showMassCenter)
        viewMassCenter.setStatusTip('View the mass center of the cross-section')
        
        translateSectionsAct = QtGui.QAction("&Translate Sections",win)
        translateSectionsAct.triggered.connect(self.translateSections)
        
        analyzeSectionsAct = QtGui.QAction("&Analyze Sections",win)
        analyzeSectionsAct.triggered.connect(self.analyzeSections)
        
        exportSectionsAct = QtGui.QAction("&Export Sections Stiffness",win)
        exportSectionsAct.triggered.connect(self.exportSectionsStiffness)
        
        exportSectionsNeu = QtGui.QAction("&Export Sections FEMAP Neutral",win)
        exportSectionsNeu.triggered.connect(self.exportSectionsNeutral)
        
        exportSectionsCrit = QtGui.QAction("&Export Sections Criteria",win)
        exportSectionsCrit.triggered.connect(self.exportSectionsCriteria)
        
        mainMenu = win.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(printEntity)
        fileMenu.addAction(loadDatFile)
        fileMenu.addAction(loadNASTFile)
        fileMenu.addAction(loadCSVFile)
        fileMenu.addAction(quitEntity)
        
        createMenu = mainMenu.addMenu('&Create')
        createMenu.addAction(createMaterial)
        createMenu.addAction(createLaminate)
        
        sectionMenu = mainMenu.addMenu('&Cross-Section')
        sectionMenu.addAction(selectSections)
        viewMenu = sectionMenu.addMenu('&View')
        viewMenu.addAction(viewCSYS)
        viewMenu.addAction(viewNormals)
        viewMenu.addAction(viewRefAxis)
        viewMenu.addAction(viewShearCenter)
        viewMenu.addAction(viewTensionCenter)
        viewMenu.addAction(viewMassCenter)
        sectionMenu.addAction(translateSectionsAct)
        sectionMenu.addAction(analyzeSectionsAct)
        sectionMenu.addAction(exportSectionsAct)
        sectionMenu.addAction(exportSectionsNeu)
        sectionMenu.addAction(exportSectionsCrit)
        
        self.pop = None

        #fileMenu.addAction(extractAction)
        
        # Create dock 1 for visualizing the global model
        d1 = Dock("Global Model", size=(900,900))
        # Create dock 2 for visualizing the cross-section model
        d2 = Dock("Cross-Section Model", size=(900,900))
        # Add dock 1 above dock 2
        area.addDock(d1, 'left')
        area.addDock(d2, 'above', d1)
        
        ##############
        # MISCELLANEOUS
        ##############
        # Create a combobox for selecting different cross-sections
        sectionComboBox2 = QtGui.QComboBox()
        # Connect the selection of the button to a plotting method
        sectionComboBox2.activated.connect(self.Model.plotRigidXSect)
        sectionComboBox2.addItem('')
        self.xsectDropDown = sectionComboBox2
        
        ##############
        # DOCK 1 SETUP - GLOBAL MODEL VIEW
        ##############
        # Create the layout widget
        w1 = pg.LayoutWidget()
        w1_global = pg.LayoutWidget()
        # Create the opengl widget for vis
        openglWindow1 = gl.GLViewWidget()
        # Set background color to white
        background_color = pg.mkColor(255,255,255)
        openglWindow1.setBackgroundColor(background_color)
        self.globalModelView = openglWindow1
        # Create a coordinate system for the view
        ax1 = gl.GLAxisItem()
        self.GCSYS = ax1
        # Set it's size to have unit 5 length
        #TODO: Adjust the size of the axis based on the size of the model
        ax1.setSize(5,5,5)
        # Add the CSYS to the gl view widget
        openglWindow1.addItem(ax1)
        # Set the size of the opengl Window
        openglWindow1.sizeHint = lambda: pg.QtCore.QSize(900, 800)
        # Create the color bar
        colorBarLayout1 = pg.GraphicsLayoutWidget()
        ax1 = pg.AxisItem('left')
        xmin = 0
        xmax = 1
        ax1.setRange(xmin, xmax)
        colorBarLayout1.addItem(ax1)
        colorBarLayout1.sizeHint = lambda: pg.QtCore.QSize(100, 800)
        gw1 = pg.GradientEditorItem(orientation='right')
        gw1.restoreState(Gradients['standard'])
        ticks1 = copy.copy(gw1.ticks)
        for tick in ticks1:
            val = gw1.ticks[tick]
            #if max(gw1.ticks.values())==val or min(gw1.ticks.values())==val:
            tick.movable=False
            #else:
            #    gw1.removeTick(tick)
        colorBarLayout1.addItem(gw1)
        colorBarLayout1.setMaximumSize(QtCore.QSize(100, 800))
        openglWindow1.setSizePolicy(colorBarLayout1.sizePolicy())
        
        #Create Update Button
        globalViewUpdate = QtGui.QPushButton('Redraw Model')
        globalViewUpdate.clicked.connect(self.redrawGlobal)
        
        # Add the color bar and opengl widgets to the window (dock 1)
        w1.addWidget(openglWindow1, row=0, col=0)
        w1.addWidget(colorBarLayout1, row=0, col=1)
        w1.addWidget(w1_global,row=1,col=0)
        w1_global.addWidget(globalViewUpdate,row=0,col=0)
        d1.addWidget(w1)
        
        ##############
        # DOCK 2 SETUP - CROSS-SECTION MODEL VIEW
        ##############
        # Create a layout widget to hold the entire dock
        w2 = pg.LayoutWidget()
        # Create a second dock area to flip between analysis and loading
        sectionArea = DockArea()
        sectionAnalysis = Dock("Cross-Section Analysis", size=(1000,200))
        sectionAnalysis.setMaximumSize(QtCore.QSize(1000, 200))
        sectionLoading = Dock("Cross-Section Loading", size=(1000,400))
        sectionLoading.setMaximumSize(QtCore.QSize(1000, 400))
        sectionArea.addDock(sectionLoading,position='left')
        sectionArea.addDock(sectionAnalysis,'above',sectionLoading)
        
        
        # Create the opengl widget for the cross-section
        openglWindow2 = gl.GLViewWidget()
        # Set the background color of the window to white
        openglWindow2.setBackgroundColor(background_color)
        self.XSectView = openglWindow2
        # Create a coordinate system for the view
        ax2 = gl.GLAxisItem()
        # Set it's size to have unit 5 length
        ax2.setSize(5,5,5)
        # Add the CSYS to the gl view widget
        openglWindow2.addItem(ax2)
        openglWindow2.sizeHint = lambda: pg.QtCore.QSize(900, 800)
        # Create the color bar
        colorBarLayout2 = pg.GraphicsLayoutWidget()
        ax2 = pg.AxisItem('left')
        xmin = 0
        xmax = 1
        ax2.setRange(xmin, xmax)
        self.xsectColorRange = ax2
        colorBarLayout2.addItem(ax2)
        colorBarLayout2.sizeHint = lambda: pg.QtCore.QSize(100, 800)
        gw2 = pg.GradientEditorItem(orientation='right')
        self.xsectColorGradient = gw2
        gw2.restoreState(Gradients['standard'])
        ticks2 = copy.copy(gw2.ticks)
        ticks2.values
        for tick in ticks2:
            val = gw2.ticks[tick]
            #if max(gw2.ticks.values())==val or min(gw2.ticks.values())==val:
            tick.movable=False
            #else:
            #    gw2.removeTick(tick)
        colorBarLayout2.addItem(gw2)
        colorBarLayout2.setMaximumSize(QtCore.QSize(100, 800))
        openglWindow2.setSizePolicy(colorBarLayout2.sizePolicy())
        
        #Create a label for the active cross-section
        comboBox2Label = QtGui.QLabel('Cross-section:  ')
        comboBox2Label.setAlignment(QtCore.Qt.AlignLeft)
        self.sectionLabel = comboBox2Label
        
        w2.addWidget(openglWindow2, row=0, col=0)
        w2.addWidget(colorBarLayout2, row=0, col=1)
        w2.addWidget(comboBox2Label, row=1, col=0,colspan=2)
        w2.addWidget(sectionArea, row=2, col=0,colspan=2)
        d2.addWidget(w2)
        
        ##############
        # CROSS-SECTION ANALYSIS VIEW
        ##############
        # Create a layout widget for cross-section analysis
        xsect_analysis_layout = pg.LayoutWidget()
        # Label and drop down box for ref_axis options
        ref_ax_label = QtGui.QLabel('Refrence axis options:')
        ref_ax_Box = QtGui.QComboBox()
        ref_ax_Box.addItem('Tension Center')
        ref_ax_Box.addItem('Mass Center')
        ref_ax_Box.addItem('Shear Center')
        ref_ax_Box.addItem('Custom Axis')
        self.ref_ax_Box = ref_ax_Box
        # Button to set custom x_ref
        xref_label = QtGui.QLabel("x_ref")
        xref_field = QtGui.QLineEdit()
        xref_field.setValidator(QtGui.QDoubleValidator())
        xref_field.textChanged.connect(self.Model.setXref)
        self.xref_field = xref_field
        # Button to set custom y_ref
        yref_label = QtGui.QLabel("y_ref")
        yref_field = QtGui.QLineEdit()
        yref_field.setValidator(QtGui.QDoubleValidator())
        yref_field.textChanged.connect(self.Model.setYref)
        self.yref_field = yref_field
        # Update reference axis button
        setRefAxisButton = QtGui.QPushButton('Update Reference Axis')
        setRefAxisButton.clicked.connect(self.Model.updateSectionRefAx)
        # Establish a tolerance for cross-sectional analysis
        tol_label = QtGui.QLabel("Solver Tolerance:")
        tol_field = QtGui.QLineEdit()
        tol_field.setValidator(QtGui.QDoubleValidator())
        tol_field.setText("1e-12")
        tol_field.textChanged.connect(self.Model.setTol)
        # Analyze Section
        analyzeSectionButton = QtGui.QPushButton('Analyze Section')
        analyzeSectionButton.clicked.connect(self.Model.analyzeActiveSection)
        ### Add buttons to the Analysis Layout
        xsect_analysis_layout.addWidget(ref_ax_label, row=0, col=0)
        xsect_analysis_layout.addWidget(ref_ax_Box, row=1, col=0)
        xsect_analysis_layout.addWidget(xref_label, row=1, col=1)
        xsect_analysis_layout.addWidget(yref_label, row=1, col=2)
        xsect_analysis_layout.addWidget(tol_label, row=1, col=3)
        xsect_analysis_layout.addWidget(setRefAxisButton, row=2, col=0)
        xsect_analysis_layout.addWidget(xref_field, row=2, col=1)
        xsect_analysis_layout.addWidget(yref_field, row=2, col=2)
        xsect_analysis_layout.addWidget(tol_field, row=2, col=3)
        xsect_analysis_layout.addWidget(analyzeSectionButton, row=3, col=0)
        sectionAnalysis.addWidget(xsect_analysis_layout)
        
        ##############
        # CROSS-SECTION LOADING VIEW
        ##############
        # Create a layout widget for cross-section analysis
        xsect_loading_layout = pg.LayoutWidget()
        
        #Determine the source for applied loads
        load_source_label = QtGui.QLabel("Loading Source:")
        load_source_drop = QtGui.QComboBox()
        load_source_drop.addItem('Custom')
        load_source_drop.addItem('All')
        load_source_drop.addItem('Set')
        load_source_drop.addItem('Load ID')
        load_source_drop.activated.connect(self.updateLSIDsDrop)
        self.load_source_drop = load_source_drop
        load_source_vals = QtGui.QComboBox()
        load_source_vals.activated.connect(self.updateLoadFields)
        self.load_source_vals = load_source_vals
        
        # Create 6 Text Edit Fields and labels for loads
        # FX:
        fx_label = QtGui.QLabel("Fx (Shear)")
        fx_field = QtGui.QLineEdit()
        fx_field.setValidator(QtGui.QDoubleValidator())
        fx_field.textChanged.connect(self.Model.setFx)
        # FY:
        fy_label = QtGui.QLabel("Fy (Shear)")
        fy_field = QtGui.QLineEdit()
        fy_field.setValidator(QtGui.QDoubleValidator())
        fy_field.textChanged.connect(self.Model.setFy)
        # FZ:
        fz_label = QtGui.QLabel("Fz (Axial)")
        fz_field = QtGui.QLineEdit()
        fz_field.setValidator(QtGui.QDoubleValidator())
        fz_field.textChanged.connect(self.Model.setFz)
        # MX:
        Mx_label = QtGui.QLabel("Mx (Bending)")
        Mx_field = QtGui.QLineEdit()
        Mx_field.setValidator(QtGui.QDoubleValidator())
        Mx_field.textChanged.connect(self.Model.setMx)
        # MY:
        My_label = QtGui.QLabel("My (Bending)")
        My_field = QtGui.QLineEdit()
        My_field.setValidator(QtGui.QDoubleValidator())
        My_field.textChanged.connect(self.Model.setMy)
        # MZ:
        Mz_label = QtGui.QLabel("Mz (Torsion)")
        Mz_field = QtGui.QLineEdit()
        Mz_field.setValidator(QtGui.QDoubleValidator())
        Mz_field.textChanged.connect(self.Model.setMz)
        #Make a connection to the load fields for later use
        self.loadFields = [fx_field,fy_field,fz_field,Mx_field,My_field,Mz_field]
        # Contour Box
        contourLabel = QtGui.QLabel("Contour:")
        contourBox = QtGui.QComboBox()
        contourBox.addItem('')
        contourBox.addItem('Von Mises Stress')
        contourBox.addItem('Maximum Principle Stress')
        contourBox.addItem('Minimum Principle Stress')
        contourBox.addItem('Sigma_xx')
        contourBox.addItem('Sigma_yy')
        contourBox.addItem('Sigma_zz')
        contourBox.addItem('Sigma_yz')
        contourBox.addItem('Sigma_xz')
        contourBox.addItem('Sigma_xy')
        contourBox.addItem('Sigma_11')
        contourBox.addItem('Sigma_22')
        contourBox.addItem('Sigma_33')
        contourBox.addItem('Sigma_23')
        contourBox.addItem('Sigma_13')
        contourBox.addItem('Sigma_12')
        contourBox.addItem('Maximum Principle Strain')
        contourBox.addItem('Minimum Principle Strain')
        contourBox.addItem('Max Abs Principle Strain')
        contourBox.addItem('Eps_11')
        contourBox.addItem('Eps_22')
        contourBox.addItem('Eps_33')
        contourBox.addItem('Eps_23')
        contourBox.addItem('Eps_13')
        contourBox.addItem('Eps_12')
        contourBox.addItem('Hoff')
        self.contourBox = contourBox
        #Set up Contour Limit Combo Box
        cont_limits_label = QtGui.QLabel("Contour Limit Setting")
        contLimSetting = QtGui.QComboBox()
        contLimSetting.addItem('Automatic')
        contLimSetting.addItem('Max-Min')
        self.contLimSetting = contLimSetting
        # Upper Contour Limit Scale:
        max_cont_lim_label = QtGui.QLabel("Upper Contour Limit")
        max_cont_lim_field = QtGui.QLineEdit()
        max_cont_lim_field.setValidator(QtGui.QDoubleValidator())
        max_cont_lim_field.textChanged.connect(self.Model.setMaxContLim)
        # Lower Contour Limit Scale:
        min_cont_lim_label = QtGui.QLabel("Lower Contour Limit")
        min_cont_lim_field = QtGui.QLineEdit()
        min_cont_lim_field.setValidator(QtGui.QDoubleValidator())
        min_cont_lim_field.textChanged.connect(self.Model.setMinContLim)
        # Warping Displacement Scale:
        warpFactor_label = QtGui.QLabel("Warping Scale:")
        warpFactor_field = QtGui.QLineEdit()
        warpFactor_field.setValidator(QtGui.QDoubleValidator())
        warpFactor_field.textChanged.connect(self.Model.setWarpFactor)
        # Calculate Warp Effects
        calcWarpButton = QtGui.QPushButton('Load Cross-Section')
        calcWarpButton.clicked.connect(self.Model.loadSection)

        xsect_loading_layout.addWidget(load_source_label, row=0, col=0)
        xsect_loading_layout.addWidget(fx_label, row=0, col=1)
        xsect_loading_layout.addWidget(fx_field, row=0, col=2)
        xsect_loading_layout.addWidget(Mx_label, row=0, col=3)
        xsect_loading_layout.addWidget(Mx_field, row=0, col=4)
        xsect_loading_layout.addWidget(load_source_drop, row=1, col=0)
        xsect_loading_layout.addWidget(fy_label, row=1, col=1)
        xsect_loading_layout.addWidget(fy_field, row=1, col=2)
        xsect_loading_layout.addWidget(My_label, row=1, col=3)
        xsect_loading_layout.addWidget(My_field, row=1, col=4)
        xsect_loading_layout.addWidget(load_source_vals, row=2, col=0)
        xsect_loading_layout.addWidget(fz_label, row=2, col=1)
        xsect_loading_layout.addWidget(fz_field, row=2, col=2)
        xsect_loading_layout.addWidget(Mz_label, row=2, col=3)
        xsect_loading_layout.addWidget(Mz_field, row=2, col=4)
        xsect_loading_layout.addWidget(contourLabel, row=3, col=0)
        xsect_loading_layout.addWidget(cont_limits_label, row=3, col=1)
        xsect_loading_layout.addWidget(max_cont_lim_label, row=3, col=2)
        xsect_loading_layout.addWidget(min_cont_lim_label, row=3, col=3)
        xsect_loading_layout.addWidget(contourBox, row=4, col=0)
        xsect_loading_layout.addWidget(contLimSetting, row=4, col=1)
        xsect_loading_layout.addWidget(max_cont_lim_field, row=4, col=2)
        xsect_loading_layout.addWidget(min_cont_lim_field, row=4, col=3)
        xsect_loading_layout.addWidget(warpFactor_label, row=5, col=0)
        xsect_loading_layout.addWidget(warpFactor_field, row=5, col=1)
        xsect_loading_layout.addWidget(calcWarpButton, row=6, col=0)
        sectionLoading.addWidget(xsect_loading_layout)
        
        
        # Display GUI
        win.show()
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    def updateXSectDrop(self):
        XIDs = self.Model.sections.getIDs()
        self.xsectDropDown.clear()
        self.xsectDropDown.addItem('')
        for XID in XIDs:
            self.xsectDropDown.addItem(str(XID))
    def updateLSIDsDrop(self):
        self.load_source_vals.clear()
        if self.load_source_drop.currentText()=='Set':
            LSIDs = list(self.Model.sectionLSIDs.keys())
            for LSID in LSIDs:
                self.load_source_vals.addItem(str(LSID))
        elif self.load_source_drop.currentText()=='Load ID':
            if self.xsectDropDown.currentText()=='':
                print('Please select a cross-section so that you can see the available'\
                      ' list of Load IDs.')
                self.load_source_vals.addItem('')
            else:
                currentXID = int(self.xsectDropDown.currentText())
                if currentXID not in self.Model.sectionLoads.keys():
                    print('Cannot populate LC dropdown with LCIDs if no loads'\
                          ' have been added to section {}'.format(currentXID))
                else:
                    LCIDs = list(self.Model.sectionLoads[currentXID].keys())
                    for LCID in LCIDs:
                        self.load_source_vals.addItem(str(LCID))
    def updateLoadFields(self):
        if self.xsectDropDown.currentText()=='':
            print('Please select a cross-section so that you can see the available'\
                  ' list of Load IDs.')
        elif self.load_source_drop.currentText()=='Load ID':
            currentXID = int(self.xsectDropDown.currentText())
            LCID = int(self.load_source_vals.currentText())
            loads = self.Model.sectionLoads[currentXID][LCID]
            currentSection = self.Model.sections.get(currentXID)
            newLoads = currentSection.transformLoads(loads)
            self.loadFields[0].setText(str(newLoads[0,0]))
            self.loadFields[1].setText(str(newLoads[1,0]))
            self.loadFields[2].setText(str(newLoads[2,0]))
            self.loadFields[3].setText(str(newLoads[3,0]))
            self.loadFields[4].setText(str(newLoads[4,0]))
            self.loadFields[5].setText(str(newLoads[5,0]))
            #Transform load to the reference axis
            
    def updateColorBar(self):
        xmin = self.Model.xsectColorScale[0]
        xmax = self.Model.xsectColorScale[1]
        self.xsectColorRange.setRange(xmin, xmax)
        
    def quitVis(self):
        QtCore.QCoreApplication.instance().quit()
        
    def createMaterial(self):
        material_popup = MaterialPopUpMenu()
        material_popup.Model = self.Model
    def analyzeSections(self):
        analyze_sections_popup = BulkSectionAnalysis()
        analyze_sections_popup.Model = self.Model
    def translateSections(self):
        translate_sections_popup = BulkSectionTranslate()
        translate_sections_popup.Model = self.Model
    def exportSectionsStiffness(self):
        export_sections_popup = ExportSectionsStiffness()
        export_sections_popup.Model = self.Model
    def exportSectionsNeutral(self):
        export_sections_neu_popup = ExportSectionsNeutral()
        export_sections_neu_popup.Model = self.Model
    def exportSectionsCriteria(self):
        export_criteria_neu_popup = ExportCriteria()
        export_criteria_neu_popup.Model = self.Model
    def createLaminate(self):
        laminate_popup = LaminatePopup()
        laminate_popup.Model = self.Model
    def showCSYS(self):
        self.Model.showCSYS = not self.Model.showCSYS
        self.Model.plotRigidXSect()
    def showNormals(self):
        self.Model.showNormals = not self.Model.showNormals
        self.Model.plotRigidXSect()
    def showRefAxis(self):
        self.Model.showRefAxis = not self.Model.showRefAxis
        self.Model.plotRigidXSect()
    def showShearCenter(self):
        self.Model.showShearCenter = not self.Model.showShearCenter
        self.Model.plotRigidXSect()
    def showTensionCenter(self):
        self.Model.showTensionCenter = not self.Model.showTensionCenter
        self.Model.plotRigidXSect()
    def showMassCenter(self):
        self.Model.showMassCenter = not self.Model.showMassCenter
        self.Model.plotRigidXSect()
    def redrawGlobal(self):
        for BID, beam in self.Model.beams.beamDict.items():
            xsect = beam.xsect
            xsect.plotRigid()
            numXSects = beam.numXSects
            for i in range(0,numXSects+1):
                EID, local_x_nd = beam.getEIDatx(float(i)/float(numXSects))
                elem = beam.elems[EID]
                x1 = elem.n1.x
                x2 = elem.n2.x
                newVertices = np.dot(xsect.vertices,elem.T[0:3,0:3])
                xtmp = x1[0]*local_x_nd+x2[0]*local_x_nd
                ytmp = x1[1]*local_x_nd+x2[1]*local_x_nd
                ztmp = x1[2]*local_x_nd+x2[2]*local_x_nd
                md = gl.MeshData(vertexes=newVertices, faces=xsect.surfaces,\
                                 vertexColors=xsect.colors)
                mesh = gl.GLMeshItem(meshdata=md,drawEdges=True,edgeColor=(0, 0, 0, 1))
                self.globalModels += [mesh]
                mesh.translate(xtmp,ytmp,ztmp)
        for item in self.globalModelView.items:
            if not item==self.GCSYS:
                self.globalModelView.removeItem(item)
        for item in self.Model.globalModels:
            self.globalModelView.addItem(item)
    def printEntities(self):
        printObject = printPopup(self.Model)
        self.printWindow = printObject
    def selectSection(self):
        cross_section_selection = selectCrossSectionPopup(self.Model)
        self.sectionSelectionWindow = cross_section_selection
    def loadDat(self):
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setNameFilters(["AeroComBAT File (*.dat)"])
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            for file in filenames:
                self.Model.loadDat(file)
    def loadNASTRAN(self):
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setNameFilters(["NASTRAN File (*.dat)"])
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            nastranPopup = LoadNastranPopup(filenames)
            nastranPopup.Model = self.Model
    def loadCSVLoads(self):
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setNameFilters(["CSV File (*.csv)"])
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            csvPopup = LoadCSVPopup(filenames)
            csvPopup.Model = self.Model
    def AeroQuit(self):
        self.app.quit()
            
class MaterialPopUpMenu(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        # Create a dock for Isotropic Materials
        d1 = Dock("Isotropic Material", size=(400,1000))
        # Create a dock for Transversely Isotropic Materials
        d2 = Dock("Transversely Isotropic Material", size=(400,1000))
        # Create a dock for Orthotropic Materials
        d3 = Dock("Orthotropic Material", size=(400,1000))
        # Add dock 1 above dock 2
        self.widgets1 = []
        self.widgets2 = []
        self.widgets3 = []
        self.addDock(d1, 'top')
        self.addDock(d2, position='below', relativeTo=d1)
        self.addDock(d3, position='below', relativeTo=d2)
        
        ######################################################################
        ############### ISOTROPIC MATERIAL LAYOUT ############################
        ######################################################################
        iso_mat_layout = pg.LayoutWidget()
        # Material Name
        iso_mat_name_label = QtGui.QLabel()
        iso_mat_name_label.setText('Material Name:')
        iso_mat_name = QtGui.QLineEdit()
        self.widgets1 += [iso_mat_name]
        # Material ID
        iso_mat_id_label = QtGui.QLabel()
        iso_mat_id_label.setText('Material ID:')
        iso_mat_id = QtGui.QLineEdit()
        iso_mat_id.setValidator(QtGui.QIntValidator())
        self.widgets1 += [iso_mat_id]
        # Youngs Modulus
        iso_mat_E_label = QtGui.QLabel()
        iso_mat_E_label.setText('E:')
        iso_mat_E = QtGui.QLineEdit()
        iso_mat_E.setValidator(QtGui.QDoubleValidator())
        self.widgets1 += [iso_mat_E]
        # Poisson Ratio
        iso_mat_nu_label = QtGui.QLabel()
        iso_mat_nu_label.setText('nu:')
        iso_mat_nu = QtGui.QLineEdit()
        iso_mat_nu.setValidator(QtGui.QDoubleValidator())
        self.widgets1 += [iso_mat_nu]
        # Density
        iso_mat_rho_label = QtGui.QLabel()
        iso_mat_rho_label.setText('Density:')
        iso_mat_rho = QtGui.QLineEdit()
        iso_mat_rho.setValidator(QtGui.QDoubleValidator())
        self.widgets1 += [iso_mat_rho]
        # Material Thickness
        iso_mat_t_label = QtGui.QLabel()
        iso_mat_t_label.setText('thickness:')
        iso_mat_t = QtGui.QLineEdit()
        iso_mat_t.setValidator(QtGui.QDoubleValidator())
        self.widgets1 += [iso_mat_t]
        # Create Iso Material Button
        iso_button = QtGui.QPushButton('Create Material')
        iso_button.clicked.connect(self.pressedIso)
        self.widgets1 += [iso_button]
        
        iso_mat_layout.addWidget(iso_mat_name_label,row=0,col=0)
        iso_mat_layout.addWidget(iso_mat_name,row=1,col=0)
        iso_mat_layout.addWidget(iso_mat_id_label,row=0,col=1)
        iso_mat_layout.addWidget(iso_mat_id,row=1,col=1)
        iso_mat_layout.addWidget(iso_mat_rho_label,row=0,col=2)
        iso_mat_layout.addWidget(iso_mat_rho,row=1,col=2)
        iso_mat_layout.addWidget(iso_mat_t_label,row=0,col=3)
        iso_mat_layout.addWidget(iso_mat_t,row=1,col=3)
        iso_mat_layout.addWidget(iso_mat_E_label,row=2,col=0)
        iso_mat_layout.addWidget(iso_mat_E,row=3,col=0)
        iso_mat_layout.addWidget(iso_mat_nu_label,row=2,col=1)
        iso_mat_layout.addWidget(iso_mat_nu,row=3,col=1)
        iso_mat_layout.addWidget(iso_button,row=4,col=0)
        
        d1.addWidget(iso_mat_layout)
        
        ######################################################################
        ########## TRASVERSELY ISOTROPIC MATERIAL LAYOUT #####################
        ######################################################################
        tiso_mat_layout = pg.LayoutWidget()
        # Material Name
        tiso_mat_name_label = QtGui.QLabel()
        tiso_mat_name_label.setText('Material Name:')
        tiso_mat_name = QtGui.QLineEdit()
        self.widgets2 += [tiso_mat_name]
        # Material ID
        tiso_mat_id_label = QtGui.QLabel()
        tiso_mat_id_label.setText('Material ID:')
        tiso_mat_id = QtGui.QLineEdit()
        tiso_mat_id.setValidator(QtGui.QIntValidator())
        self.widgets2 += [tiso_mat_id]
        # E1
        tiso_mat_E1_label = QtGui.QLabel()
        tiso_mat_E1_label.setText('E1:')
        tiso_mat_E1 = QtGui.QLineEdit()
        tiso_mat_E1.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_E1]
        # E2
        tiso_mat_E2_label = QtGui.QLabel()
        tiso_mat_E2_label.setText('E2=E3:')
        tiso_mat_E2 = QtGui.QLineEdit()
        tiso_mat_E2.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_E2]
        # nu_23
        tiso_mat_nu23_label = QtGui.QLabel()
        tiso_mat_nu23_label.setText('nu_23:')
        tiso_mat_nu23 = QtGui.QLineEdit()
        tiso_mat_nu23.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_nu23]
        # nu_13/nu_12
        tiso_mat_nu12_label = QtGui.QLabel()
        tiso_mat_nu12_label.setText('nu_13=nu_12:')
        tiso_mat_nu12 = QtGui.QLineEdit()
        tiso_mat_nu12.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_nu12]
        # G_12
        tiso_mat_G12_label = QtGui.QLabel()
        tiso_mat_G12_label.setText('G_12:')
        tiso_mat_G12 = QtGui.QLineEdit()
        tiso_mat_G12.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_G12]
        # Density
        tiso_mat_rho_label = QtGui.QLabel()
        tiso_mat_rho_label.setText('Density:')
        tiso_mat_rho = QtGui.QLineEdit()
        tiso_mat_rho.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_rho]
        # Material Thickness
        tiso_mat_t_label = QtGui.QLabel()
        tiso_mat_t_label.setText('thickness (opt.):')
        tiso_mat_t = QtGui.QLineEdit()
        tiso_mat_t.setValidator(QtGui.QDoubleValidator())
        self.widgets2 += [tiso_mat_t]
        # Create Iso Material Button
        tiso_button = QtGui.QPushButton('Create Material')
        tiso_button.clicked.connect(self.pressedTiso)
        self.widgets2 += [tiso_button]
        
        tiso_mat_layout.addWidget(tiso_mat_name_label,row=0,col=0)
        tiso_mat_layout.addWidget(tiso_mat_name,row=1,col=0)
        tiso_mat_layout.addWidget(tiso_mat_id_label,row=0,col=1)
        tiso_mat_layout.addWidget(tiso_mat_id,row=1,col=1)
        tiso_mat_layout.addWidget(tiso_mat_rho_label,row=0,col=2)
        tiso_mat_layout.addWidget(tiso_mat_rho,row=1,col=2)
        tiso_mat_layout.addWidget(tiso_mat_t_label,row=0,col=3)
        tiso_mat_layout.addWidget(tiso_mat_t,row=1,col=3)
        tiso_mat_layout.addWidget(tiso_mat_E1_label,row=2,col=0)
        tiso_mat_layout.addWidget(tiso_mat_E1,row=3,col=0)
        tiso_mat_layout.addWidget(tiso_mat_E2_label,row=2,col=1)
        tiso_mat_layout.addWidget(tiso_mat_E2,row=3,col=1)
        tiso_mat_layout.addWidget(tiso_mat_nu23_label,row=2,col=2)
        tiso_mat_layout.addWidget(tiso_mat_nu23,row=3,col=2)
        tiso_mat_layout.addWidget(tiso_mat_nu12_label,row=2,col=3)
        tiso_mat_layout.addWidget(tiso_mat_nu12,row=3,col=3)
        tiso_mat_layout.addWidget(tiso_mat_G12_label,row=2,col=4)
        tiso_mat_layout.addWidget(tiso_mat_G12,row=3,col=4)
        tiso_mat_layout.addWidget(tiso_button,row=4,col=0)
        
        d2.addWidget(tiso_mat_layout)
        
        ######################################################################
        ################## ORTHOTROPIC MATERIAL LAYOUT #######################
        ######################################################################
        ortho_mat_layout = pg.LayoutWidget()
        # Material Name
        ortho_mat_name_label = QtGui.QLabel()
        ortho_mat_name_label.setText('Material Name:')
        ortho_mat_name = QtGui.QLineEdit()
        self.widgets3 += [ortho_mat_name]
        # Material ID
        ortho_mat_id_label = QtGui.QLabel()
        ortho_mat_id_label.setText('Material ID:')
        ortho_mat_id = QtGui.QLineEdit()
        ortho_mat_id.setValidator(QtGui.QIntValidator())
        self.widgets3 += [ortho_mat_id]
        # E1
        ortho_mat_E1_label = QtGui.QLabel()
        ortho_mat_E1_label.setText('E1:')
        ortho_mat_E1 = QtGui.QLineEdit()
        ortho_mat_E1.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_E1]
        # E2
        ortho_mat_E2_label = QtGui.QLabel()
        ortho_mat_E2_label.setText('E2:')
        ortho_mat_E2 = QtGui.QLineEdit()
        ortho_mat_E2.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_E2]
        # E3
        ortho_mat_E3_label = QtGui.QLabel()
        ortho_mat_E3_label.setText('E3:')
        ortho_mat_E3 = QtGui.QLineEdit()
        ortho_mat_E3.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_E3]
        # nu_23
        ortho_mat_nu23_label = QtGui.QLabel()
        ortho_mat_nu23_label.setText('nu_23:')
        ortho_mat_nu23 = QtGui.QLineEdit()
        ortho_mat_nu23.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_nu23]
        # nu_13
        ortho_mat_nu13_label = QtGui.QLabel()
        ortho_mat_nu13_label.setText('nu_13:')
        ortho_mat_nu13 = QtGui.QLineEdit()
        ortho_mat_nu13.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_nu13]
        # nu_12
        ortho_mat_nu12_label = QtGui.QLabel()
        ortho_mat_nu12_label.setText('nu_12:')
        ortho_mat_nu12 = QtGui.QLineEdit()
        ortho_mat_nu12.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_nu12]
        # G_23
        ortho_mat_G23_label = QtGui.QLabel()
        ortho_mat_G23_label.setText('G_23:')
        ortho_mat_G23 = QtGui.QLineEdit()
        ortho_mat_G23.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_G23]
        # G_13
        ortho_mat_G13_label = QtGui.QLabel()
        ortho_mat_G13_label.setText('G_13:')
        ortho_mat_G13 = QtGui.QLineEdit()
        ortho_mat_G13.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_G13]
        # G_12
        ortho_mat_G12_label = QtGui.QLabel()
        ortho_mat_G12_label.setText('G_12:')
        ortho_mat_G12 = QtGui.QLineEdit()
        ortho_mat_G12.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_G12]
        # Density
        ortho_mat_rho_label = QtGui.QLabel()
        ortho_mat_rho_label.setText('Density:')
        ortho_mat_rho = QtGui.QLineEdit()
        ortho_mat_rho.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_rho]
        # Material Thickness
        ortho_mat_t_label = QtGui.QLabel()
        ortho_mat_t_label.setText('thickness (opt.):')
        ortho_mat_t = QtGui.QLineEdit()
        ortho_mat_t.setValidator(QtGui.QDoubleValidator())
        self.widgets3 += [ortho_mat_t]
        # Create Iso Material Button
        ortho_button = QtGui.QPushButton('Create Material')
        ortho_button.clicked.connect(self.pressedOrtho)
        self.widgets3 += [ortho_button]
        
        ortho_mat_layout.addWidget(ortho_mat_name_label,row=0,col=0)
        ortho_mat_layout.addWidget(ortho_mat_name,row=1,col=0)
        ortho_mat_layout.addWidget(ortho_mat_id_label,row=0,col=1)
        ortho_mat_layout.addWidget(ortho_mat_id,row=1,col=1)
        ortho_mat_layout.addWidget(ortho_mat_rho_label,row=0,col=2)
        ortho_mat_layout.addWidget(ortho_mat_rho,row=1,col=2)
        ortho_mat_layout.addWidget(ortho_mat_t_label,row=0,col=3)
        ortho_mat_layout.addWidget(ortho_mat_t,row=1,col=3)
        ortho_mat_layout.addWidget(ortho_mat_E1_label,row=2,col=0)
        ortho_mat_layout.addWidget(ortho_mat_E1,row=3,col=0)
        ortho_mat_layout.addWidget(ortho_mat_E2_label,row=2,col=1)
        ortho_mat_layout.addWidget(ortho_mat_E2,row=3,col=1)
        ortho_mat_layout.addWidget(ortho_mat_E3_label,row=2,col=2)
        ortho_mat_layout.addWidget(ortho_mat_E3,row=3,col=2)
        ortho_mat_layout.addWidget(ortho_mat_G23_label,row=4,col=0)
        ortho_mat_layout.addWidget(ortho_mat_G23,row=5,col=0)
        ortho_mat_layout.addWidget(ortho_mat_G13_label,row=4,col=1)
        ortho_mat_layout.addWidget(ortho_mat_G13,row=5,col=1)
        ortho_mat_layout.addWidget(ortho_mat_G12_label,row=4,col=2)
        ortho_mat_layout.addWidget(ortho_mat_G12,row=5,col=2)
        ortho_mat_layout.addWidget(ortho_mat_nu23_label,row=6,col=0)
        ortho_mat_layout.addWidget(ortho_mat_nu23,row=7,col=0)
        ortho_mat_layout.addWidget(ortho_mat_nu13_label,row=6,col=1)
        ortho_mat_layout.addWidget(ortho_mat_nu13,row=7,col=1)
        ortho_mat_layout.addWidget(ortho_mat_nu12_label,row=6,col=2)
        ortho_mat_layout.addWidget(ortho_mat_nu12,row=7,col=2)
        ortho_mat_layout.addWidget(ortho_button,row=8,col=0)
        
        d3.addWidget(ortho_mat_layout)
        
        self.show()
        
    def pressedIso(self):
        print('')
        mat_name_field = self.widgets1[0]
        mat_ID_field = self.widgets1[1]
        mat_E_field = self.widgets1[2]
        mat_nu_field = self.widgets1[3]
        mat_rho_field = self.widgets1[4]
        mat_t_field = self.widgets1[5]
        temp_flag = False
        if mat_name_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material name missing...')
        if mat_ID_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material ID missing...')
        if mat_E_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E missing...')
        if mat_nu_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu missing...')
        if mat_rho_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material density missing...')
        if temp_flag:
            print('Please complete the isotropic material form...')
        else:
            MID = int(mat_ID_field.text())
            mat_name = mat_name_field.text()
            mat_type = 'ISO'
            mat_constants = [float(mat_E_field.text()),\
                             float(mat_nu_field.text()),\
                             float(mat_rho_field.text()),1.,1.,1.]
            if mat_t_field.text()=='':
               mat_t = 0.
            else:
                mat_t = float(mat_t_field.text())
            self.Model.materials.addMaterial(MID,mat_name,mat_type,\
                mat_constants,mat_t)
            print('Successfully created isotropic material!')
            self.destroy()
            
    def pressedTiso(self):
        print('')
        mat_name_field = self.widgets2[0]
        mat_ID_field = self.widgets2[1]
        mat_E1_field = self.widgets2[2]
        mat_E2_field = self.widgets2[3]
        mat_nu23_field = self.widgets2[4]
        mat_nu12_field = self.widgets2[5]
        mat_G12_field = self.widgets2[6]
        mat_rho_field = self.widgets2[7]
        mat_t_field = self.widgets2[8]
        temp_flag = False
        if mat_name_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material name missing...')
        if mat_ID_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material ID missing...')
        if mat_E1_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E1 missing...')
        if mat_E2_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E2 missing...')
        if mat_nu23_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu_23 missing...')
        if mat_nu12_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu_12 missing...')
        if mat_G12_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material G_12 missing...')
        if mat_rho_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material density missing...')
        if temp_flag:
            print('Please complete the transversely isotropic material form...')
        else:
            MID = int(mat_ID_field.text())
            mat_name = mat_name_field.text()
            mat_type = 'TISO'
            mat_constants = [float(mat_E1_field.text()),\
                             float(mat_E2_field.text()),\
                             float(mat_nu23_field.text()),\
                             float(mat_nu12_field.text()),\
                             float(mat_G12_field.text()),\
                             float(mat_rho_field.text()),1.,1.,1.,1.,1.]
            if mat_t_field.text()=='':
               mat_t = 0.
            else:
                mat_t = float(mat_t_field.text())
            self.Model.materials.addMaterial(MID,mat_name,mat_type,\
                mat_constants,mat_t)
            print('Successfully created transversely isotropic material!')
            self.destroy()
    
    def pressedOrtho(self):
        print('')
        mat_name_field = self.widgets3[0]
        mat_ID_field = self.widgets3[1]
        mat_E1_field = self.widgets3[2]
        mat_E2_field = self.widgets3[3]
        mat_E3_field = self.widgets3[4]
        mat_nu23_field = self.widgets3[5]
        mat_nu13_field = self.widgets3[6]
        mat_nu12_field = self.widgets3[7]
        mat_G23_field = self.widgets3[8]
        mat_G13_field = self.widgets3[9]
        mat_G12_field = self.widgets3[10]
        mat_rho_field = self.widgets3[11]
        mat_t_field = self.widgets3[12]
        temp_flag = False
        if mat_name_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material name missing...')
        if mat_ID_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material ID missing...')
        if mat_E1_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E1 missing...')
        if mat_E2_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E2 missing...')
        if mat_E3_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material E3 missing...')
        if mat_nu23_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu_23 missing...')
        if mat_nu13_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu_13 missing...')
        if mat_nu12_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material nu_12 missing...')
        if mat_G23_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material G_23 missing...')
        if mat_G13_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material G_13 missing...')
        if mat_G12_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material G_12 missing...')
        if mat_rho_field.text()=='' or temp_flag:
            temp_flag = True
            print('Material density missing...')
        if temp_flag:
            print('Please complete the orthotropic material form...')
        else:
            MID = int(mat_ID_field.text())
            mat_name = mat_name_field.text()
            mat_type = 'ORTHO'
            mat_constants = [float(mat_E1_field.text()),\
                             float(mat_E2_field.text()),\
                             float(mat_E3_field.text()),\
                             float(mat_nu23_field.text()),\
                             float(mat_nu13_field.text()),\
                             float(mat_nu12_field.text()),\
                             float(mat_G23_field.text()),\
                             float(mat_G13_field.text()),\
                             float(mat_G12_field.text()),\
                             float(mat_rho_field.text()),1.,1.,1.,1.,1.,1.,1.,1.,1.]
            if mat_t_field.text()=='':
               mat_t = 0.
            else:
                mat_t = float(mat_t_field.text())
            self.Model.materials.addMaterial(MID,mat_name,mat_type,\
                mat_constants,mat_t)
            print('Successfully created orthotropic material!')
            self.destroy()
            
class printPopup(DockArea):
    def __init__(self,Model):
        DockArea.__init__(self)
        self.Model = Model
        d1 = Dock("Printing Query", size=(300,300))
        self.widgets = []
        self.addDock(d1, 'top')
        # Material ID
        id_label = QtGui.QLabel()
        id_label.setText('Entity ID')
        tmp_id = QtGui.QLineEdit()
        tmp_id.setValidator(QtGui.QIntValidator())
        self.widgets += [tmp_id]
        
        # Create radio button for entities
        entity_selection = QtGui.QComboBox()
        entity_selection.addItems(['XNode','XElement','Node','Beam Element',\
                                   'Material','Laminate','Cross-Section'])
        self.widgets += [entity_selection]
        
        # Create Iso Material Button
        button = QtGui.QPushButton('Print Entity')
        button.clicked.connect(self.pressed)
        self.widgets += [button]
        
        # Set GUI Window
        print_layout = pg.LayoutWidget()
        print_layout.addWidget(id_label,row=0,col=0)
        print_layout.addWidget(tmp_id,row=0,col=1)
        print_layout.addWidget(entity_selection,row=1,col=0)
        print_layout.addWidget(button,row=1,col=1)
        
        d1.addWidget(print_layout)
        
        self.show()
    
    def pressed(self):
        print('')
        cb = self.widgets[1]
        if cb.currentText()=='Material':
            Library = self.Model.materials
        elif cb.currentText()=='XNode':
            Library = self.Model.xnodes
        elif cb.currentText()=='Node':
            Library = self.Model.nodes
        elif cb.currentText()=='XElement':
            Library = self.Model.xelements
        elif cb.currentText()=='Beam Element':
            Library = self.Model.belements
        elif cb.currentText()=='Laminate':
            Library = self.Model.laminates
        elif cb.currentText()=='Cross-Section':
            Library = self.Model.sections
        else:
            print('Please select a valid library')
        try:
            ID = int(self.widgets[0].text())
            if ID not in Library.getIDs():
                print('This ID {} does not exist within the models {}...'.format(ID,Library.type))
            else:
                if Library.type == 'MaterialLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'XNodeLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'NodeLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'XElemLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'BeamElementLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'LaminateLibrary':
                    entity = Library.get(ID)
                elif Library.type == 'CrossSectionLibrary':
                    entity = Library.get(ID)
                entity.printSummary()
        except:
            print('Please enter a valid entity ID...')
        
            
class selectCrossSectionPopup(DockArea):
    def __init__(self,Model):
        DockArea.__init__(self)
        self.Model = Model
        d1 = Dock("Select Cross-Section", size=(300,100))
        d1.setMaximumSize(QtCore.QSize(300, 100))
        d1.setMinimumSize(QtCore.QSize(300, 100))
        self.widgets = []
        self.addDock(d1, 'top')
        # Material ID
        id_label = QtGui.QLabel()
        id_label.setText('Cross-Section ID')
        
        # Set GUI Window
        section_select_layout = pg.LayoutWidget()
        section_select_layout.addWidget(id_label,row=0,col=0)
        section_select_layout.addWidget(self.Model.GUI.xsectDropDown,row=0,col=1)
        
        d1.addWidget(section_select_layout)
        
        self.show()
    def pop(self):
        self.window = Pop()

class LaminatePopup(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Laminate Editor", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        lam_layout = pg.LayoutWidget()
        #Laminate ID Field
        lam_label = QtGui.QLabel('Laminate ID:')
        lam_id = QtGui.QLineEdit()
        lam_id.setValidator(QtGui.QIntValidator())
        self.widgets += [lam_id]
        # Number of plies ID Field
        Ni_label = QtGui.QLabel('N_i:')
        Ni_field = QtGui.QLineEdit()
        Ni_field.setToolTip('Number of plies. Ex: [2,1,4,1,1]')
        self.widgets += [Ni_field]
        # Material ID for plies
        Mi_label = QtGui.QLabel('M_i:')
        Mi_field = QtGui.QLineEdit()
        Mi_field.setToolTip('Material ID for plies. Ex: [1,2,1,2,1]')
        self.widgets += [Mi_field]
        # Laminate orientation
        thi_label = QtGui.QLabel('Theta_i:')
        thi_field = QtGui.QLineEdit()
        thi_field.setToolTip('Fiber Orientation for plies. Ex: [0,0,45,0,0]')
        self.widgets += [thi_field]
        # Symmetric Flag
        sym_label = QtGui.QLabel('Symmetric:')
        sym_box = QtGui.QCheckBox()
        sym_box.setToolTip('Is this half of a symmetric laminate?')
        self.widgets += [sym_box]
        # Create Laminate button
        create_lam = QtGui.QPushButton('Create Laminate')
        create_lam.clicked.connect(self.pressed)
        self.widgets += [create_lam]
        
        lam_layout.addWidget(lam_label,row=0,col=0)
        lam_layout.addWidget(lam_id,row=0,col=1)
        lam_layout.addWidget(Ni_label,row=1,col=0)
        lam_layout.addWidget(Ni_field,row=1,col=1)
        lam_layout.addWidget(Mi_label,row=2,col=0)
        lam_layout.addWidget(Mi_field,row=2,col=1)
        lam_layout.addWidget(thi_label,row=3,col=0)
        lam_layout.addWidget(thi_field,row=3,col=1)
        lam_layout.addWidget(sym_label,row=4,col=0)
        lam_layout.addWidget(sym_box,row=4,col=1)
        lam_layout.addWidget(create_lam,row=5,col=0)
        
        d1.addWidget(lam_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        try:
            n_i = ast.literal_eval(self.widgets[1].text())
            m_i = ast.literal_eval(self.widgets[2].text())
            th_i = ast.literal_eval(self.widgets[3].text())
        except:
            print('Entered an invalid input. Please enter laminate inputs in the following form...\n \
                  N_i = [1,2,4,3,1] where each number corresponds to the number of plies for a given orientation and material\n \
                  M_i = [1,2,1,2,1] where each number corresponds to a material ID for a given orientation\n \
                  Theta_i = [0,45,90,-45,0] where each number corresponds to a fiber orientation.\n\n \
                  Note that all three vectors have to have the same number of terms.')
        if not (len(n_i)==len(m_i) and len(n_i)==len(th_i)):
            print("The number of plies, material IDs, and fiber "
            "orientations lists must all be the same length.")
        sym = self.widgets[4].isChecked()
        proceed=True
        for MID in m_i:
            if self.Model.materials.getMaterial(MID).t==0.:
                print("You cannot make a laminate using a material property"
                      " whose material thickness=0. Laminate not created.\n")
                proceed=False
        LAMID = int(self.widgets[0].text())
        if LAMID in self.Model.laminates.getIDs():
            print('Please select a different laminate ID, the current selection is already in use...')
            proceed = False
        if proceed:
            #NiLSID = max(self.Model.lists.keys())+1
            #self.Model.lists[NiLSID] = n_i
            #MiLSID = NiLSID + 1
            #self.Model.lists[MiLSID] = m_i
            #THiLSID = MiLSID + 1
            #self.Model.lists[THiLSID] = th_i
            
            self.Model.laminates.addLaminate(LAMID,n_i,m_i,self.Model.materials,th=th_i,sym=sym)
            
            laminate = self.Model.laminates.getLaminate(LAMID)
            laminate.NiLSID = n_i
            laminate.MiLSID = m_i
            laminate.THiLSID = th_i
            self.destroy()
            
            
class BulkSectionAnalysis(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Bulk Cross-Section Analysis", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        analysis_layout = pg.LayoutWidget()
        # Select Which Sections to Analyze
        sections_label = QtGui.QLabel('Cross-Section IDs:')
        section_ids = QtGui.QLineEdit()
        self.widgets += [section_ids]
        # Analyze button
        analyze_sections_button = QtGui.QPushButton('Analyze Sections')
        analyze_sections_button.clicked.connect(self.pressed)
        self.widgets += [analyze_sections_button]
        
        analysis_layout.addWidget(sections_label,row=0,col=0)
        analysis_layout.addWidget(section_ids,row=0,col=1)
        analysis_layout.addWidget(analyze_sections_button,row=1,col=0)
        
        d1.addWidget(analysis_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        #try:
        section_ids = ast.literal_eval(self.widgets[0].text())
        proceed=True
        XIDs = []
        for XID in section_ids:
            if not isinstance(XID,int):
                #proceed=False
                XIDs += [int(XID)]
            else:
                XIDs += [XID]
        if proceed:
            self.Model.analyzeSections(XIDs)
            self.destroy()
        else:
            print('At least one of the values in the following list is not an integer... \n')
            print(section_ids)

class BulkSectionTranslate(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Bulk Cross-Section Translate", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        translate_layout = pg.LayoutWidget()
        # Select Which Sections to Analyze
        sections_label = QtGui.QLabel('Cross-Section IDs:')
        section_ids = QtGui.QLineEdit()
        x_label = QtGui.QLabel('X-Translate:')
        x_value_field = QtGui.QLineEdit()
        x_value_field.setValidator(QtGui.QDoubleValidator())
        y_label = QtGui.QLabel('Y-Translate:')
        y_value_field = QtGui.QLineEdit()
        y_value_field.setValidator(QtGui.QDoubleValidator())
        self.widgets += [section_ids]
        self.widgets += [x_value_field]
        self.widgets += [y_value_field]
        # Analyze button
        analyze_sections_button = QtGui.QPushButton('Translate Sections')
        analyze_sections_button.clicked.connect(self.pressed)
        self.widgets += [analyze_sections_button]
        
        translate_layout.addWidget(sections_label,row=0,col=0)
        translate_layout.addWidget(section_ids,row=0,col=1)
        translate_layout.addWidget(x_label,row=1,col=0)
        translate_layout.addWidget(x_value_field,row=1,col=1)
        translate_layout.addWidget(y_label,row=2,col=0)
        translate_layout.addWidget(y_value_field,row=2,col=1)
        translate_layout.addWidget(analyze_sections_button,row=3,col=0)
        
        d1.addWidget(translate_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        #try:
        section_ids = ast.literal_eval(self.widgets[0].text())
        proceed=True
        XIDs = []
        for XID in section_ids:
            if not isinstance(XID,int):
                #proceed=False
                XIDs += [int(XID)]
            else:
                XIDs += [XID]
        if proceed:
            x = float(self.widgets[1].text())
            y = float(self.widgets[2].text())
            section_lib = self.Model.sections
            for XID in XIDs:
                section = section_lib.get(XID)
                section.translateSection(x,y)
            self.Model.plotRigidXSect()
            self.destroy()
        else:
            print('At least one of the values in the following list is not an integer... \n')
            print(section_ids)

class ExportSectionsStiffness(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Export Cross-Sections Stiffness Matricies", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        export_layout = pg.LayoutWidget()
        # Select Which Sections to Export
        sections_label = QtGui.QLabel('Cross-Section IDs:')
        section_ids = QtGui.QLineEdit()
        self.widgets += [section_ids]
        filename_label = QtGui.QLabel('Export filename:')
        filename_string = QtGui.QLineEdit()
        self.widgets += [filename_string]
        #Path location button
        filename_path_label = QtGui.QLabel('File Path:')
        filename_path = QtGui.QLineEdit()
        self.widgets += [filename_path]
        # Export Sections button
        analyze_sections_button = QtGui.QPushButton('Export Sections Stiffness')
        analyze_sections_button.clicked.connect(self.pressed)
        self.widgets += [analyze_sections_button]
        
        export_layout.addWidget(sections_label,row=0,col=0)
        export_layout.addWidget(section_ids,row=0,col=1)
        export_layout.addWidget(filename_label,row=1,col=0)
        export_layout.addWidget(filename_string,row=1,col=1)
        export_layout.addWidget(filename_path_label,row=2,col=0)
        export_layout.addWidget(filename_path,row=2,col=1)
        export_layout.addWidget(analyze_sections_button,row=3,col=0)
        
        d1.addWidget(export_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        #try:
        section_ids = ast.literal_eval(self.widgets[0].text())
        filename = self.widgets[1].text()
        path = self.widgets[2].text().replace("'","").replace('"','')
        proceed=True
        XIDs = []
        for XID in section_ids:
            if not isinstance(XID,int):
                #proceed=False
                XIDs += [int(XID)]
            else:
                XIDs += [XID]
        if proceed:
            self.Model.exportSections(XIDs,filename,path=path)
            self.destroy()
        else:
            print('At least one of the values in the following list is not an integer... \n')
            print(section_ids)
            
class ExportSectionsNeutral(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Export Cross-Sections FEMAP Neutral", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        export_layout = pg.LayoutWidget()
        # Select Which Sections to Export
        sections_label = QtGui.QLabel('Cross-Section IDs:')
        section_ids = QtGui.QLineEdit()
        self.widgets += [section_ids]
        filename_label = QtGui.QLabel('Export filename (without .neu):')
        filename_string = QtGui.QLineEdit()
        self.widgets += [filename_string]
        #Path location button
        filename_path_label = QtGui.QLabel('File Path:')
        filename_path = QtGui.QLineEdit()
        self.widgets += [filename_path]
        # Export Sections button
        analyze_sections_button = QtGui.QPushButton('Export Sections')
        analyze_sections_button.clicked.connect(self.pressed)
        self.widgets += [analyze_sections_button]
        
        export_layout.addWidget(sections_label,row=0,col=0)
        export_layout.addWidget(section_ids,row=0,col=1)
        export_layout.addWidget(filename_label,row=1,col=0)
        export_layout.addWidget(filename_string,row=1,col=1)
        export_layout.addWidget(filename_path_label,row=2,col=0)
        export_layout.addWidget(filename_path,row=2,col=1)
        export_layout.addWidget(analyze_sections_button,row=3,col=0)
        
        d1.addWidget(export_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        #try:
        section_ids = ast.literal_eval(self.widgets[0].text())
        filename = self.widgets[1].text()
        path = self.widgets[2].text().replace("'","").replace('"','')
        proceed=True
        XIDs = []
        for XID in section_ids:
            if not isinstance(XID,int):
                #proceed=False
                XIDs += [int(XID)]
            else:
                XIDs += [XID]
        if proceed:
            self.Model.exportSectionsNeutral(XIDs,filename,path=path)
            self.destroy()
        else:
            print('At least one of the values in the following list is not an integer... \n')
            print(section_ids)
            
class ExportCriteria(DockArea):
    def __init__(self):
        DockArea.__init__(self)
        d1 = Dock("Export Cross-Sections Results FEMAP Neutral", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        
        export_layout = pg.LayoutWidget()
        # Select Which Sections to Export
        sections_label = QtGui.QLabel('Cross-Section IDs:')
        section_ids = QtGui.QLineEdit()
        self.widgets += [section_ids]
        filename_label = QtGui.QLabel('Export filename (without .neu):')
        filename_string = QtGui.QLineEdit()
        self.widgets += [filename_string]
        #Path location button
        filename_path_label = QtGui.QLabel('File Path:')
        filename_path = QtGui.QLineEdit()
        self.widgets += [filename_path]
        # Export Sections button
        analyze_sections_button = QtGui.QPushButton('Export Results')
        analyze_sections_button.clicked.connect(self.pressed)
        self.widgets += [analyze_sections_button]
        #Add criteria boxes
        crit1_label = QtGui.QCheckBox('Von Mises Stress')
        crit2_label = QtGui.QCheckBox('Maximum Principle Stress')
        crit3_label = QtGui.QCheckBox('Minimum Principle Stress')
        crit4_label = QtGui.QCheckBox('Sigma_xx')
        crit5_label = QtGui.QCheckBox('Sigma_yy')
        crit6_label = QtGui.QCheckBox('Sigma_zz')
        crit7_label = QtGui.QCheckBox('Sigma_yz')
        crit8_label = QtGui.QCheckBox('Sigma_xz')
        crit9_label = QtGui.QCheckBox('Sigma_xy')
        crit10_label = QtGui.QCheckBox('Sigma_11')
        crit11_label = QtGui.QCheckBox('Sigma_22')
        crit12_label = QtGui.QCheckBox('Sigma_33')
        crit13_label = QtGui.QCheckBox('Sigma_23')
        crit14_label = QtGui.QCheckBox('Sigma_13')
        crit15_label = QtGui.QCheckBox('Sigma_12')
        crit16_label = QtGui.QCheckBox('Maximum Principle Strain')
        crit17_label = QtGui.QCheckBox('Minimum Principle Strain')
        crit18_label = QtGui.QCheckBox('Max Abs Principle Strain')
        crit19_label = QtGui.QCheckBox('Eps_11')
        crit20_label = QtGui.QCheckBox('Eps_22')
        crit21_label = QtGui.QCheckBox('Eps_33')
        crit22_label = QtGui.QCheckBox('Eps_23')
        crit23_label = QtGui.QCheckBox('Eps_13')
        crit24_label = QtGui.QCheckBox('Eps_12')
        crit25_label = QtGui.QCheckBox('Hoff')
        
        self.critboxes = [crit1_label,crit2_label,crit3_label,crit4_label,\
                          crit5_label,crit6_label,crit7_label,crit8_label,\
                          crit9_label,crit10_label,crit11_label,crit12_label,\
                          crit13_label,crit14_label,crit15_label,crit16_label,\
                          crit17_label,crit18_label,crit19_label,crit20_label,\
                          crit21_label,crit22_label,crit23_label,crit24_label,\
                          crit25_label]
        
        export_layout.addWidget(sections_label,row=0,col=0)
        export_layout.addWidget(section_ids,row=0,col=1)
        export_layout.addWidget(filename_label,row=1,col=0)
        export_layout.addWidget(filename_string,row=1,col=1)
        export_layout.addWidget(filename_path_label,row=2,col=0)
        export_layout.addWidget(filename_path,row=2,col=1)
        export_layout.addWidget(analyze_sections_button,row=3,col=0)
        export_layout.addWidget(crit1_label,row=3,col=1)
        export_layout.addWidget(crit2_label,row=4,col=1)
        export_layout.addWidget(crit3_label,row=5,col=1)
        export_layout.addWidget(crit4_label,row=6,col=1)
        export_layout.addWidget(crit5_label,row=7,col=1)
        export_layout.addWidget(crit6_label,row=8,col=1)
        export_layout.addWidget(crit7_label,row=9,col=1)
        export_layout.addWidget(crit8_label,row=10,col=1)
        export_layout.addWidget(crit9_label,row=11,col=1)
        export_layout.addWidget(crit10_label,row=12,col=1)
        export_layout.addWidget(crit11_label,row=13,col=1)
        export_layout.addWidget(crit12_label,row=14,col=1)
        export_layout.addWidget(crit13_label,row=15,col=1)
        export_layout.addWidget(crit14_label,row=16,col=1)
        export_layout.addWidget(crit15_label,row=17,col=1)
        export_layout.addWidget(crit16_label,row=18,col=1)
        export_layout.addWidget(crit17_label,row=19,col=1)
        export_layout.addWidget(crit18_label,row=20,col=1)
        export_layout.addWidget(crit19_label,row=21,col=1)
        export_layout.addWidget(crit20_label,row=22,col=1)
        export_layout.addWidget(crit21_label,row=23,col=1)
        export_layout.addWidget(crit22_label,row=24,col=1)
        export_layout.addWidget(crit23_label,row=25,col=1)
        export_layout.addWidget(crit24_label,row=26,col=1)
        export_layout.addWidget(crit25_label,row=27,col=1)
        
        
        d1.addWidget(export_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        
        #try:
        section_ids = ast.literal_eval(self.widgets[0].text())
        filename = self.widgets[1].text()
        path = self.widgets[2].text().replace("'","").replace('"','')
        proceed=True
        XIDs = []
        for XID in section_ids:
            if not isinstance(XID,int):
                #proceed=False
                XIDs += [int(XID)]
            else:
                XIDs += [XID]
        criteria = []
        for box in self.critboxes:
            if box.isChecked():
                criteria += [box.text()]
        if proceed:
            self.Model.exportSectionsContour(XIDs,filename,criteria,path=path)
            self.destroy()
        else:
            print('At least one of the values in the following list is not an integer... \n')
            print(section_ids)
            
class LoadNastranPopup(DockArea):
    def __init__(self,filenames):
        DockArea.__init__(self)
        d1 = Dock("Cross-Section Directions", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        self.proceed=False
        self.xdir=None
        self.ydir=None
        self.filenames = filenames
        
        export_layout = pg.LayoutWidget()
        # Select Which Sections to Export
        xdir_label = QtGui.QLabel('Cross-Section x-direction:')
        xdir = QtGui.QComboBox()
        xdir.addItem('+x')
        xdir.addItem('-x')
        xdir.addItem('+y')
        xdir.addItem('-y')
        xdir.addItem('+z')
        xdir.addItem('-z')
        self.widgets += [xdir]
        ydir_label = QtGui.QLabel('Cross-Section y-direction:')
        ydir = QtGui.QComboBox()
        ydir.addItem('+x')
        ydir.addItem('-x')
        ydir.addItem('+y')
        ydir.addItem('-y')
        ydir.addItem('+z')
        ydir.addItem('-z')
        self.widgets += [ydir]
        #Add XID
        XID_label = QtGui.QLabel('Cross-Section ID (Optional):')
        XID_field = QtGui.QLineEdit()
        XID_field.setValidator(QtGui.QIntValidator())
        # Export Sections button
        self.widgets += [XID_field]
        loadNastran = QtGui.QPushButton('Load Nastran File')
        loadNastran.clicked.connect(self.pressed)
        self.widgets += [loadNastran]
        
        export_layout.addWidget(xdir_label,row=0,col=0)
        export_layout.addWidget(xdir,row=0,col=1)
        export_layout.addWidget(ydir_label,row=1,col=0)
        export_layout.addWidget(ydir,row=1,col=1)
        export_layout.addWidget(XID_label,row=2,col=0)
        export_layout.addWidget(XID_field,row=2,col=1)
        export_layout.addWidget(loadNastran,row=3,col=0)
        
        d1.addWidget(export_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        xdir = self.widgets[0].currentText()
        ydir = self.widgets[1].currentText()
        XID = self.widgets[2].text()
        validDirs = ['+x','-x','+y','-y','+z','-z']
        if (xdir in validDirs) and (ydir in validDirs):
            if xdir[-1]==ydir[-1]:
                print('The x and y global directions cannot be the same. Both'\
                      ' are currently set to {}.'.format(xdir))
            else:
                if xdir=='+x':
                    xdir=1
                elif xdir=='-x':
                    xdir=-1
                elif xdir=='+y':
                    xdir=2
                elif xdir=='-y':
                    xdir=-2
                elif xdir=='+z':
                    xdir=3
                else:
                    xdir=-3
                if ydir=='+x':
                    ydir=1
                elif ydir=='-x':
                    ydir=-1
                elif ydir=='+y':
                    ydir=2
                elif ydir=='-y':
                    ydir=-2
                elif ydir=='+z':
                    ydir=3
                else:
                    ydir=-3
                
                if XID=="":
                    self.Model.translateNastranDat(self.filenames,xdir,ydir)
                else:
                    self.Model.translateNastranDat(self.filenames,xdir,ydir,XID=int(XID))
                self.destroy()
            
        else:
            print('Please enter a valid x and/or y cross-direction. Must be "+x", "-x", "+y", "-y", "+z", or "-z".')
            
class LoadCSVPopup(DockArea):
    def __init__(self,filenames):
        DockArea.__init__(self)
        d1 = Dock("Loading Directions", size=(500,800))
        self.widgets = []
        self.addDock(d1, 'top')
        self.proceed=False
        self.xdir=None
        self.ydir=None
        self.filenames = filenames
        
        export_layout = pg.LayoutWidget()
        # Select Which Sections to Export
        xdir_label = QtGui.QLabel('Loading x-direction:')
        xdir = QtGui.QComboBox()
        xdir.addItem('+x')
        xdir.addItem('-x')
        xdir.addItem('+y')
        xdir.addItem('-y')
        xdir.addItem('+z')
        xdir.addItem('-z')
        self.widgets += [xdir]
        ydir_label = QtGui.QLabel('Loading y-direction:')
        ydir = QtGui.QComboBox()
        ydir.addItem('+x')
        ydir.addItem('-x')
        ydir.addItem('+y')
        ydir.addItem('-y')
        ydir.addItem('+z')
        ydir.addItem('-z')
        self.widgets += [ydir]
        #Add LF
        LF_label = QtGui.QLabel('Load Factor:')
        LF_field = QtGui.QLineEdit()
        LF_field.setValidator(QtGui.QDoubleValidator())
        LF_field.setText('1.00')
        self.widgets += [LF_field]
        # Export Sections button
        loadNastran = QtGui.QPushButton('Load CSV File')
        loadNastran.clicked.connect(self.pressed)
        self.widgets += [loadNastran]
        
        export_layout.addWidget(xdir_label,row=0,col=0)
        export_layout.addWidget(xdir,row=0,col=1)
        export_layout.addWidget(ydir_label,row=1,col=0)
        export_layout.addWidget(ydir,row=1,col=1)
        export_layout.addWidget(LF_label,row=2,col=0)
        export_layout.addWidget(LF_field,row=2,col=1)
        export_layout.addWidget(loadNastran,row=3,col=0)
        
        d1.addWidget(export_layout)
        
        self.resize(500,150)
        
        self.show()
    
    def pressed(self):
        xdir = self.widgets[0].currentText()
        ydir = self.widgets[1].currentText()
        LF = float(self.widgets[2].text())
        validDirs = ['+x','-x','+y','-y','+z','-z']
        if (xdir in validDirs) and (ydir in validDirs):
            if xdir[-1]==ydir[-1]:
                print('The x and y global directions cannot be the same. Both'\
                      ' are currently set to {}.'.format(xdir))
            elif LF==0:
                print('Please enter a load factor that is not 0.')
            else:
                if xdir=='+x':
                    xdir=1
                elif xdir=='-x':
                    xdir=-1
                elif xdir=='+y':
                    xdir=2
                elif xdir=='-y':
                    xdir=-2
                elif xdir=='+z':
                    xdir=3
                else:
                    xdir=-3
                if ydir=='+x':
                    ydir=1
                elif ydir=='-x':
                    ydir=-1
                elif ydir=='+y':
                    ydir=2
                elif ydir=='-y':
                    ydir=-2
                elif ydir=='+z':
                    ydir=3
                else:
                    ydir=-3
                
                self.Model.importSectionLoads(self.filenames,xdir,ydir,LF=LF)
                self.destroy()
            
        else:
            print('Please enter a valid x and/or y cross-direction. Must be "x", "y", or "z".')
        

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    GUI = GUI()
    
