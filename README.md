# RNN-on-C-AFM-data
# Tested with tensorflow 1.13.1 and keras 2.2.4

Code for running a recurrent neural network on C-AFM data

# Overview
The code is devided into three parts:
  1. prepare_data.py - prepares the data for the neural net
  2. runNN.py        - runs the neural net
  3. monitor.py      - visualizes the data after running the NN
  
The code for the neural net itself is stored in NN/__init_\_.py

# 1. prepare_data.py 
Reads the XLS file containing the data
  - Performs dimensionality reduction and standardization.

# 2. runNN.py
Runs the neural network.
  - The parameters for the neural network are hard coded in this file.
  - Designed to run for an unspecified amount of time, therefore the max epochs is set abonormally high
  - Will store the weights as a file whenever the model imporoves
  - Multiple versions of runNN.py can be run simultaneously with different input parameters
  
# 3 monitor.py
Tracks the progress of neural networks as they are training
  - Reads the weights from trained neural networks and visualizes the data
  - All subfolders in a directory are monitored
  - Visualize the most recent state of the network being trained in each subfolder
    - Read the most recent weights (.hdf5 file)
    - Also read the the start state of the network (.h5 file) to get the network geometery
    - Generates multiple plots that visualize the data in different ways
    - (will pass if plots already exist)
  - After all subfolders have been evaluated:
    - Sleep for 1 hour
    - Restart
