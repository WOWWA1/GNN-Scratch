import numpy as np 
import pandas as pd 
import random
from simple_graph import Graph
from NeuralNetwork import NeuralNetwork

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

exclude_edges_set = set()
for edge in exclude_edges:
    exclude_edges_set.add(tuple(edge)) 
    exclude_edges_set.add(tuple(reversed(edge)))  


for index, row in df.iterrows():
    edge_tuple = (row['id_1'], row['id_2'])
    if edge_tuple not in exclude_edges_set:
        train_graph.add_node(row['id_1'])
        train_graph.add_node(row['id_2'])
        train_graph.add_edge(row['id_1'], row['id_2'])

labelled_positive_pairs = [(pair, 1) for pair in train_positive_pairs]
labelled_negative_pairs = [(pair, 0) for pair in train_negative_pairs]
labelled_pairs = labelled_positive_pairs + labelled_negative_pairs
random.shuffle(labelled_pairs)


print(train_graph.edges)

embeddings = np.load('facebook_embeddings.npy', allow_pickle=True).item()



def pair_to_vector(pair, embeddings):
    if isinstance(pair[0], list):
        node1, node2 = pair[0][0], pair[0][1]
    else:
        node1, node2 = pair[0], pair[1]
    
    return np.concatenate([embeddings[node1], embeddings[node2]])

def evaluate_model(neural_network, test_pairs, embeddings):
    correct = 0
    total = 0
    errors = 0
    
    import random
    positive_samples = [(pair, label) for pair, label in test_pairs if label == 1]
    negative_samples = [(pair, label) for pair, label in test_pairs if label == 0]
    
    sample_positive = random.sample(positive_samples, min(10, len(positive_samples)))
    sample_negative = random.sample(negative_samples, min(10, len(negative_samples)))
    sample_test_pairs = sample_positive + sample_negative
    random.shuffle(sample_test_pairs)
    
    for i, (pair, true_label) in enumerate(sample_test_pairs):
        try:
            vector = pair_to_vector(pair, embeddings)
            predicted = neural_network.forward(vector)
            binary_prediction = 1 if predicted[0] > 0.5 else 0
            
            if binary_prediction == true_label:
                correct += 1
            total += 1
        except Exception as e:
            errors += 1
            continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def train_neural_network(labelled_pairs, embeddings, test_pairs, epochs=100):
    neural_network = NeuralNetwork(input_size=128, hidden_size=64, output_size=1, learning_rate=0.01)
    list_of_vectors = np.array([pair_to_vector(pair,embeddings) for pair, label in labelled_pairs])
    list_of_labels = np.array([label for pair, label in labelled_pairs])
    
    initial_accuracy = evaluate_model(neural_network, test_pairs, embeddings)
    print(f"Initial accuracy (no training): {initial_accuracy:.3f}")
    
    for epoch in range(epochs):
        for i in range(len(list_of_vectors)):
            idx = random.randint(0, len(list_of_vectors) - 1)
            vector = list_of_vectors[idx]
            label = list_of_labels[idx]
            predicted = neural_network.forward(vector)
            loss = neural_network.cross_entropy_loss(predicted, label)
            neural_network.backward(loss, label)
        
        test_accuracy = evaluate_model(neural_network, test_pairs, embeddings)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_accuracy:.3f}")
    
    return neural_network


smaller_pairs = labelled_pairs[:len(labelled_pairs) // 10]
test_pairs = test_positive_pairs + test_negative_pairs
neural_network = train_neural_network(smaller_pairs, embeddings, test_pairs, 10)
