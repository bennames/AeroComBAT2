from setuptools import setup
import AeroComBAT

setup(
    name='AeroComBAT',
    description=AeroComBAT.__doc__,
    author='Ben Names',
    packages=['AeroComBAT'],
    version='2.00',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'pyopengl'
        'pyqtgraph'],
    classifiers=[
    'Development Status  3 - Alpha',

    'Intended Audience  Aerospace Stress Analysts',

     'License  OSI Approved  MIT License',

    'Programming Language  Python  3.8',],
    )