# Graph Neural Network (GNN) Node2Node Regression

## 1. Introduction
This repository contains a regression model utilizing Graph Neural Network (GNN). GNN is a novel neural network model that can directly process graph-structured data such as proteomics, social network and natural language. Graph structured data commonly consists of nodes which contain machine-readable data (e.g. numeric vector) and edges connecting nodes either in a directed or undirected manner. The model in this repository aims to capture simulation results of SUMO (Simulation of Urban MObility), which is a highly portable, microscopic and continuous traffic simulation package <a href= "https://sumo.dlr.de/docs/index.html"> (LINK for SUMO) </a>. A "node to node" regression problem is set onto the GNN model so that traffic flows [nrVeh/hr] per each edge can be predicted based on OD (Origin-Destination) vector assigned for each node.

## 2. Prerequisites
All algorithms and models are purely written in Python. So, if you are using recent release of Anaconda, there won't be severe issues to execute them. But, you may need to install below packages for applications of the Graph Neural Network Model and reporting manoeuvres.</br>
+ Pandas & Numpy for data post processing.
+ Plotly & Dash for visualization of the results and parameters.
+ Scikit-learn for statistical methods.
+ Pytorch for general application of the machine learning model.
+ Pytorch Geometric (PyG) for application of the GNN model.
+ CUDA, if you want to use GPU for training acceleration.

## 3. Toy Network
A toy network is a simple traffic network all algorithms are concerning. Is is assumed that 4 blocks of residential area surrounded with links consist of two edges for both directions. And, each edge consists of two lanes. Also, two external links are set up for external influx and out flux of traffic flow. With this network, OD counts (i.e. Nr of Vehicle) and traffic flows (i.e. Veh/hr) are assinged for each edge of links. IMPORTANT: All edges in the network are also nodes in the GNN as actual traffic assignment is conducted on each edge.</br>

<p align="center"><img src="https://github.com/hosig0204/Graph_Neural_Network_OD2FLOW/blob/971b1b3c3b269545a261b3b61886f5cfb6947739/static/images/graphDefinition.jpg" width="800"></p>

## 3. "GNN_test_bed.ipynb": Notebook  