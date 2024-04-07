import torch
import pickle
import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from circle_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='circle1', help='Circle')
parser.add_argument('--threshold',  type=int, default=0, help='Threshold to set labels as 0 or 1')
parser.add_argument('--GT', default='centrality_degree', help='Metric to use as ground truth')
args = parser.parse_args()

# first we read the data
if args.data=='circle1':
    adj_matrix = read_edges('../data/gnn_explainer/circle1/107.edges')
    feature_matrix = read_features('../data/gnn_explainer/circle1/107.feat')
elif args.data=='circle2':
    adj_matrix = read_edges('../data/gnn_explainer/circle2/0.edges')
    feature_matrix = read_features('../data/gnn_explainer/circle2/0.feat')

# we generate the ground truth
if args.GT=='centrality_degree':
    labels = calculate_degree_centrality(adj_matrix)
elif args.GT=='clustering_coeff':
    labels = calculate_clustering_coefficients(adj_matrix)
elif args.GT=='eigenvector_centrality':
    labels = calculate_eigenvector_centrality(adj_matrix)

# then we compute the data in a dictionary to match the required format of the original code
graph_dict = create_graph_dict(adj_matrix, feature_matrix, labels)

# in some cases, it might be better to adjust the threshold manually
if args.threshold is not None:
    threshold = args.threshold
else:
    threshold = np.unique(graph_dict['labels']).mean()

# labels are set to 0 and 1, 0 being "not in motif" and 1 "in motif"
binary_labels = (graph_dict['labels'] > threshold).astype(int)
graph_dict['labels'] = binary_labels

# finally we store the dictionary in a pickle file
with open(f'../data/gnn_explainer/{args.data}_{args.GT}_GT.pickle', 'wb') as file:
    pickle.dump(graph_dict, file)