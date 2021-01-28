=======================
AeroComBAT Introduction
=======================

AeroComBAT (Aeroelastic Composite Beam Analysis Tool) is a python Tool intended
to allow users to efficiently models composite beam structures.

Since it's inception, many of the V1.0 features have been stripped away in favor
of adding a better user experience VIA a GUI. In it's present form, it serves
as a means of quickly iterating on complex cross-sectional designs, giving users
the means of predicting cross-sectional characteristics (such as beam stiffnesses
EI, EA, GJ) as well as detailed predictions of cross-sectional stress/strain.

:Authors: 
    Ben Names

:Version: 2.0 of 2021/28/01

Version 2.0 
===========

**Capabilities**

- Analyze multi material beam cross-sections:
   1. Import capabilities
      + Read cross-sectional meshes using the AeromComBAT input file format.
	  + Read in cross-sectional meshes from a NASTRAN run deck.
	     * Note that the capabilities for this functionality are limited to elements that use isotropic materials only
   2. Compute beam cross-section characteristics:
      + 6DOF beam stiffnesses (EA, EI, GAK, GJ)
	  + Shear center and tension center
   3. Calculate detailed stress/strain throught the cross-section
      + Loads can be applied one at a time, or can be loaded via a .csv file
   4. Export capabilities
      + Bulk export cross-sectional stiffness parameters
	  + Export cross-sections as a FEMAP neutral file
	  + Export cross-section stress/strain results as a femap neutral for more robust post processing


Installation Instructions
=========================

First of all it is strongly recomended that the user first install the Anaconda
python distribution from Continuum analytics `here <https://www.continuum.io/>`_.

By installing Anaconda, you will automatically get 2 of the AeroComBAT
dependencies: Numpy and Scipy. For visualization, AeroComBAT relies on
pyqtgraph and opengl, which can be installed using pip:

.. code-block:: python

   pip install pyopengl
   pip install pyqtgraph

In order to use the NASTRAN import capabilities, the python package pyNastran must also
be installed:

.. code-block:: python

   pip install pynastran

Finally, in order to install AeromComBAT, download the zip file from GitHub.
Once downloaded, unzip the package and navigate to the folder where you
downloaded AeroComBAT using the command prompt tool for windows or terminal for
Mac. Once in the root AeroComBAT folder, run:

.. code-block:: python

   python setup.py install

AeroComBAT should now be installed!