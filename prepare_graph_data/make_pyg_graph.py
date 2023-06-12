import os
import torch
import pandas as pd
import pickle
import json

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from common import remove_string_literal

class pyg_graph_dataset(object):
    def __init__(self, name = "", vocab_dict = None):
        self.name = name
        self.vocab_dict = vocab_dict
        self.pyg_graphs = []


    def parse(self, graph, problem, filename, problem_topic, problem_tag):

        written_num  = 0
        skip_num = 0
        try:
            pyg_data = from_networkx(graph)

            pyg_data = self.vocab_dict.update_vocab(pyg_data)

            pyg_data = Data(x = pyg_data.label, 
                            edge_index= pyg_data.edge_index,
                            edge_attr = pyg_data.type,
                            topic = problem_topic,
                            tag = problem_tag,
                            problem = problem,
                            filename = filename,
                            )

            self.pyg_graphs.append(pyg_data)
            written_num +=1
        except:
            skip_num+=1 
            
        return self.vocab_dict, written_num, skip_num
        
        
    def serialize(self, filename, dest = "./"):
        # Graph data
        with open(os.path.join(dest, "graphs", filename + ".pickle"), "wb") as f:
            pickle.dump(self.pyg_graphs, f)