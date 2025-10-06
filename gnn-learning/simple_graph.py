class Graph: 
    def __init__(self):
        self.mygraph = {}
        self.node_features = {}
        self.edges = 0
    def add_node(self, id):
        if (id in self.mygraph.keys()):
            print(f"Node {id} already exists") 
        else:
            self.mygraph[id] = []

    def add_edge(self,v1,v2):
        if (v1 not in self.mygraph.keys()):
            print(f"Vertex {v1} does not exist in the graph yet")
        elif (v2 not in self.mygraph.keys()):
            print(f"Vertex {v2} does not exist in graph yet")
        else:
            self.mygraph[v1].append(v2)
            self.mygraph[v2].append(v1)
            self.edges += 1
    def get_neighbors(self,id):
    
        return self.mygraph[id]
    
    def print_graph(self):
        for key in self.mygraph.keys():
            print(self.mygraph[key])
            
  


