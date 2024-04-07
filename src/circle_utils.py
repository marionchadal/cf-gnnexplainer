import torch
import pickle
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# to check if our circles would match the expected formats in the original code
def inspect_dictionary_dimensions(d):
    print(f"Number of keys: {len(d)}")
    for key, value in d.items():
        if isinstance(value, (list, tuple, str)):
            print(f"Key: {key}, Length of value: {len(value)}")
        elif hasattr(value, 'shape'):
            print(f"Key: {key}, Shape of value: {value.shape}")
        elif hasattr(value, 'size'):
            print(f"Key: {key}, Size of value: {value.size()}")
        else:
            print(f"Key: {key}, Type of value: {type(value)}")

# to read into the .edges files
def read_edges(file_path):
    max_node = 0
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = [int(x) for x in line.strip().split()]
            edges.append((node1, node2))
            max_node = max(max_node, node1, node2)
    
    adj_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)
    
    for node1, node2 in edges:
        adj_matrix[node1, node2] = 1
        adj_matrix[node2, node1] = 1
    
    return adj_matrix

# to filter the nodes that don't have any features
def filter_adjacency_matrix(adj_matrix, nodes_to_keep):
    return adj_matrix[nodes_to_keep][:, nodes_to_keep]

# to read the .feat files
def read_features(file_path):
    with open(file_path, 'r') as f:
        features = [line.strip().split()[1:] for line in f]
    return np.array(features, dtype=int)

# calculate degree centrality
def calculate_degree_centrality(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    degree_centrality = nx.degree_centrality(G)
    degree_centrality_list = np.array([degree_centrality[node] for node in G.nodes()])
    return degree_centrality_list

# calculate clustering coefficients
def calculate_clustering_coefficients(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    clustering_coeffs = nx.clustering(G)
    clustering_coeffs_list = np.array([clustering_coeffs[node] for node in G.nodes()])
    return clustering_coeffs_list

# calculate eigenvector centrality
def calculate_eigenvector_centrality(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    eigenvector_centrality_list = np.array([eigenvector_centrality[node] for node in G.nodes()])
    return eigenvector_centrality_list

# to create the dictionary that contains all about the graph (correct format to match the original code)
def create_graph_dict(adj_matrix, feature_matrix, labels):
    nodes_with_features = np.where(~np.all(feature_matrix == 0, axis=1))[0]
    
    filtered_feature_matrix = feature_matrix[nodes_with_features]
    filtered_feature_matrix = filtered_feature_matrix[:, :]
    filtered_labels = labels[nodes_with_features]
    filtered_adj_matrix = filter_adjacency_matrix(adj_matrix, nodes_with_features)
    
    num_nodes = filtered_feature_matrix.shape[0]
    train_ratio = 0.8
    train_size = int(train_ratio * num_nodes)
    
    graph_dict = {
        'adj': filtered_adj_matrix[np.newaxis, :, :],
        'feat': filtered_feature_matrix[np.newaxis, :, :],
        'labels': filtered_labels[np.newaxis, :],
        'train_idx': np.arange(train_size),
        'test_idx': np.arange(train_size, num_nodes),
    }
    
    return graph_dict

# some statistics about the motif in a graph
def compute_motif_statistics(adj_matrix, labels, threshold):
    G = nx.from_numpy_array(adj_matrix)
    
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    
    nodes_in_motif = np.where(labels > threshold)[0]
    num_nodes_in_motif = len(nodes_in_motif)
    
    subgraph = G.subgraph(nodes_in_motif)
    edges_in_motif = subgraph.number_of_edges()
    
    avg_degree = np.mean([deg for _, deg in G.degree()])
    
    if num_nodes_in_motif > 0:
        avg_degree_in_motif = np.mean([deg for _, deg in subgraph.degree()])
    else:
        avg_degree_in_motif = 0
    
    avg_nodes_in_motif = num_nodes_in_motif
    avg_edges_in_motif = edges_in_motif
    
    print("Graph Statistics:")
    print(f"Total Nodes: {total_nodes}, Total Edges: {total_edges}")
    print(f"Nodes in Motif: {num_nodes_in_motif}, Edges in Motif (GT): {edges_in_motif}")
    print(f"Avg Node Degree: {avg_degree:.2f}, Avg Node Degree in Motif: {avg_degree_in_motif:.2f}")
    print(f"Avg # Nodes in Motif: {avg_nodes_in_motif}, Avg # Edges in Motif: {avg_edges_in_motif}")

# graph visualisation
def visualize_graph(adj_matrix):

    G = nx.from_numpy_array(adj_matrix)
    pos = nx.kamada_kawai_layout(G)
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=100, node_color='sandybrown', edge_color='gray', font_weight='bold')
    plt.axis('off')
    plt.show()

# motif visualisation
def plot_motif_from_adjacency(adj_matrix, labels, threshold=0.02):
    G = nx.from_numpy_array(adj_matrix)

    nodes_in_motif = np.where(labels > threshold)[0]

    plt.figure(figsize=(8, 6))

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color="lightgrey")
    nx.draw_networkx_edges(G, pos, edge_color="lightgrey")

    motif_subgraph = G.subgraph(nodes_in_motif)

    nx.draw_networkx_nodes(motif_subgraph, pos, node_color="sandybrown")
    nx.draw_networkx_edges(motif_subgraph, pos, edge_color="sandybrown")

    plt.axis("off")
    plt.show()



