from simple_graph import Graph
import os
import pandas as pd 
from gensim.models import Word2Vec
import numpy as np
import random 
file_path = "/Users/tanjeedalam/Downloads/facebook_large/musae_facebook_edges.csv"

df = pd.read_csv(file_path)

graph = Graph()
for index, row in df.iterrows():
    graph.add_node(row['id_1'])
    graph.add_node(row['id_2'])
    graph.add_edge(row['id_1'], row['id_2'])

def deepwalk(graph, walk_length=80, num_walks=10, window_size=10, embedding_dim=64):
    
    walks = []
    for _ in range(num_walks):
        for node in graph.mygraph.keys(): 
            walk = generate_random_walk(graph, node, walk_length)
            walks.append([str(node) for node in walk])
    
  
    model = Word2Vec(walks, 
                    vector_size=embedding_dim,
                    window=window_size,
                    min_count=1,
                    workers=4)
    
    # Extract embeddings
    embeddings = {}
    for node in graph.mygraph.keys(): 
        embeddings[node] = model.wv[str(node)]
    
    return embeddings

def generate_random_walk(graph, start_node, walk_length):
    walk = [start_node]
    current = start_node
    
    for _ in range(walk_length - 1):
        neighbors = graph.get_neighbors(current)  
        if not neighbors:
            break
        current = random.choice(neighbors)
        walk.append(current)
    
    return walk

if __name__ == "__main__":
    print("Loading Facebook Large dataset...")
    print(f"Graph has {len(graph.mygraph)} nodes and {graph.edges} edges")
    
    print("Generating DeepWalk embeddings...")
    embeddings = deepwalk(graph, walk_length=80, num_walks=10, window_size=10, embedding_dim=64)
    
    print(f"Generated embeddings for {len(embeddings)} nodes")
    print(f"Embedding dimension: {len(list(embeddings.values())[0])}")
    
    
    for node in list(graph.mygraph.keys())[:5]:
        print(f"Node {node}: {embeddings[node][:5]}...")
    np.save('facebook_embeddings.npy', embeddings)
    print("Embeddings saved to facebook_embeddings.npy")    
