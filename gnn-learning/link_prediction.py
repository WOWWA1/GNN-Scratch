import numpy as np 
import pandas as pd 
import random
from simple_graph import Graph

file_path = "/Users/tanjeedalam/Downloads/facebook_large/musae_facebook_edges.csv"

df = pd.read_csv(file_path)

graph = Graph()
for index, row in df.iterrows():
    graph.add_node(row['id_1'])
    graph.add_node(row['id_2'])
    graph.add_edge(row['id_1'], row['id_2'])




def create_negative_edges(graph, num_negative_edges=graph.edges):
    negative_edges = []
    nodes = list(graph.mygraph.keys())
    existing_edges = set()

    for node in graph.mygraph.keys():
        for neighbor in graph.get_neighbors(node):
            if node < neighbor: 
                existing_edges.add((node, neighbor))
    
    num_found = 0
    while num_found < num_negative_edges:
        node1, node2 = random.sample(nodes, 2)
        if node1 != node2:
            edge1 = (min(node1, node2), max(node1, node2))
            if edge1 not in existing_edges:
                negative_edges.append([node1, node2])
                num_found += 1
                
    return negative_edges

def create_positive_edges(graph):
 
    positive_edges = []
    for node in graph.mygraph.keys():
        for neighbor in graph.get_neighbors(node):
            if node < neighbor: 
                positive_edges.append([node, neighbor])
    return positive_edges

negative_edges = create_negative_edges(graph)
positive_edges = create_positive_edges(graph)

negative_pairs = [(pair, 0) for pair in negative_edges]
positive_pairs = [(pair, 1) for pair in positive_edges]
random.shuffle(negative_pairs)
random.shuffle(positive_pairs)


def split_training_data(negative_pairs, positive_pairs, test_ratio=0.2):
    split_idx = int(len(negative_pairs) * (1 - test_ratio))
    train_negative_pairs = negative_pairs[:split_idx]
    test_negative_pairs = negative_pairs[split_idx:]
    train_positive_pairs = positive_pairs[:split_idx]
    test_positive_pairs = positive_pairs[split_idx:]
    return train_negative_pairs, test_negative_pairs, train_positive_pairs, test_positive_pairs

train_negative_pairs, test_negative_pairs, train_positive_pairs, test_positive_pairs = split_training_data(negative_pairs, positive_pairs)

train_graph = Graph()

exclude_edges = [edge for (edge, _) in test_positive_pairs]

# Convert exclude_edges to a set for O(1) lookup instead of O(n)
exclude_edges_set = set()
for edge in exclude_edges:
    exclude_edges_set.add(tuple(edge))  # Convert to tuple for faster lookup
    exclude_edges_set.add(tuple(reversed(edge)))  # Add both directions

# Build training graph more efficiently
for index, row in df.iterrows():
    edge_tuple = (row['id_1'], row['id_2'])
    if edge_tuple not in exclude_edges_set:
        train_graph.add_node(row['id_1'])
        train_graph.add_node(row['id_2'])
        train_graph.add_edge(row['id_1'], row['id_2'])




print(train_graph.edges)


