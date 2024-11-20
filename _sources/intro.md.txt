# Introduction

This package was written for data analysis for the timestamping mode 
of the LinoSPAD2 detector. The cornerstone of this package is the 
'unpack' module, functions from which can be used to unpack, decode, 
and neatly pack the binary-encoded data from the detector into a 3D matrix 
of pixel addresses and the timestamps themselves.

The "functions" folder holds all functions from unpacking to plotting 
numerous types of graphs (pixel population, histograms of timestamp 
differences, fits, cross-talk and dark count rate characterization, etc.)

The "params" folder holds masks (used to mask the noisiest pixels) and 
calibration data (compensating for TDC nonlinearities and offset) for 
some of the LinoSPAD2 daughterboard-motherboard combinations.

The "archive" folder is a collection of scripts for debugging, tests, 
older versions of functions, etc.

The "examples" folder is a collection of jupyter notebooks with examples 
on how to use the different functions this package provides, showcasing 
how the different parameters can be used to achieve certain results.

Some functions (mainly the plotting ones) save plots as pictures in the 
.png and .pkl format, creating a folder for the output in the same folder where 
the '.dat' data files are located. Others (such as delta_t.py for collecting 
timestamp differences in the given time window) save '.csv' and '.feather' 
files with the processed data for easier and faster plotting.

Additionally, a standalone repo with an application for real-time plotting 
of the sensor population can be found [here](https://github.com/rngKomorebi/LinoSPAD2-app).
