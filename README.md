# Graph Neural Network (GNN) Node2Node Regression

## 1. Introduction
This repository contains a regression model utilizing Graph Neural Network (GNN). GNN is a novel neural network model that can directly process graph-structured data such as proteomics, social network and natural language. Graph structured data commonly consists of nodes which contain machine-readable data (e.g. numeric vector) and edges connecting nodes either in a directed or undirected manner. The model in this repository aims to capture simulation results of SUMO (Simulation of Urban MObility), which is a highly portable, microscopic and continuous traffic simulation package <a href= "https://sumo.dlr.de/docs/index.html"> (LINK for SUMO) </a>. A "node to node" regression problem is set onto the GNN model so that traffic flows [nrVeh/hr] per each edge can be predicted based on OD (Origin-Destination) vector assigned for each node.

## 2. Prerequisites
All algorithms and models are purely written in Python. So, if you are using recent release of Anaconda, there won't be severe issues to execute them. But, you may need to install below packages for applications of the Graph Neural Network Model and reporting manoeuvres.</br>
+ Pandas & Numpy for data post processing.
+ Plotly & Dash for visualization of the results and parameters.
+ Scikit-learn for statistical methods.
+ Pytorch for general application of the machine learning model.
+ Pytorch Geometric (PyG) for application of the GNN model. <a href= "https://pytorch-geometric.readthedocs.io/en/latest/index.html"> (LINK for PyG) </a>
+ CUDA, if you want to use GPU for training acceleration.

## 3. Toy Network
A toy network is a simple traffic network all algorithms are concerning. Is is assumed that 4 blocks of residential area surrounded with links consist of two edges for both directions. And, each edge consists of two lanes. Also, two external links are set up for external influx and out flux of traffic flow. With this network, OD counts (i.e. Nr of Vehicle) and traffic flows (i.e. Veh/hr) are assinged for each edge of links. IMPORTANT: All edges in the network are also nodes in the GNN as actual traffic assignment is conducted on each edge.</br>

<p align="center"><img src="https://github.com/hosig0204/Graph_Neural_Network_OD2FLOW/blob/971b1b3c3b269545a261b3b61886f5cfb6947739/static/images/graphDefinition.jpg" width="800"></p>

## 4. Node2Node Regression Problem with GNN
The GNN model was used to set-up node to node regression problem. Input data is a set of OD vectors (sliced from the OD matrix) assigned for each node in the graph, and output data is a set of traffic flow vectors (two dimensions) also assigned for each node. The GNN model aims to predict traffic flows from the OD matrix by learning abundant simulation results. Graph Convolutional Layers (GCNs) in the GNN model provides the diffusion mechanisms (Scarselli et al. 2009) which can capture topological information of the graph in learning processes. Total 7200 simulation results from the SUMO that consists of OD matrices and corresponding traffic flows are already collected and transformed into the graph dataset ** to support PyG.</br>
** You can have raw 7200 graph files from provided zip files.

<p align="center"><img src="https://github.com/hosig0204/Graph_Neural_Network_OD2FLOW/blob/971b1b3c3b269545a261b3b61886f5cfb6947739/static/images/gnnDefinition_node2node.jpg" width="800"></p>

## 5. "GNN_test_bed.ipynb": Notebook for feature engineering
This notebook is a first place you need to visit for feature engineering of the GNN model. There are many flourishing types of GCNs, and you might configure them with different parameters pursuing the best results. Three types of GCNs are implemented in the GNN model as python class definitions: Spectoral Graph Convolutional Layers (Kipf and Welling 2016), Graph Attention Networks (Veličković et al. 2017) and  Graph Attention Networks Improved (Brody et al. 2021). Please, explore many parameters as you want.

## 6. "GNN_operation.py": Compelte execution file
Once you've found a suitable GCN layers and configurations of the GNN model, you can deploy that within this execution file. This file offers complete loops of the model training with some auxiliary functions such as reporting, parameters & checkpoint save and figure generation. The CUDA is strongly recommended to accelerate your training. 