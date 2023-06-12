import os
from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import namedtuple
import pickle
import json

import networkx as nx
from tree_sitter import Parser, Node

from common import draw_graph, initialize_cpp_tree_sitter, add_string_literal
from make_pyg_graph import pyg_graph_dataset
from pdg_graph import make_pdg_graph

from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from vocab_dict_class import VocabDict



StringLiteral = namedtuple('StringLiteral', ['value'])
DATA_PATH = "../data/cpp/cpp/"
OUTPUT_DIR = "../data/source_code_graph/"
TAG_PATH = "../data/tags.json"
TOPIC_PATH = "../data/problem_topics.json"
FILE_NAMES = os.listdir(DATA_PATH)

print(len(FILE_NAMES))

EdgeColors = {
    "Child" : "black",
    "ReturnTo" : "yellow",
    "LastLexicalUsed" : "orange",
    "LastUse" : "red",
    "ComputedFrom" : "purple",
    "NextToken" : "blue",
    "LastWrite" : "green"
}

CPP_LANGUAGE = initialize_cpp_tree_sitter()
# CPP Parser
parser = Parser()
parser.set_language(CPP_LANGUAGE)

# Tags File
with open(TAG_PATH, "r") as infile:
    tag_dict = json.load(infile)
    infile.close()

with open(TOPIC_PATH, "r") as infile:
    topic_dict = json.load(infile)
    infile.close()
    
def add_self_loop(G, node : Node, edge_attr : str, color : str):
    G.add_edge(node, node, type = edge_attr, color = color)
    return G

def make_edge_attribute(G, f_node : Node, s_node: Node, edge_attr : str, color : str):
    G.add_edge(f_node, s_node, type = edge_attr, color = color)
    return G
    
def tree_to_graph(root):
    G = nx.MultiDiGraph()
    
    todo = [root]
    NextTokenNodes = []
    LastReadNodes = {"level" : 0}
    LastWriteNodes = {"level" : 0}
    LastLexicalUsedNodes = dict()
    ComputedFromNodes = []

    FunctionDefinitionId = None

    # Make edges based on leaf nodes
    while todo:
        node = todo.pop()
        node_text = node.text.decode("utf-8")
        #print(node_text)

        G.add_node(node.id, label = add_string_literal(node.type))

        # Return To Edges
        if node.type == "function_definition":
            FunctionDefinitionId = node.id
        if node.type == "return":
            edge_attr = "ReturnTo"
            if FunctionDefinitionId is not None:
                G = make_edge_attribute(G, node.id, FunctionDefinitionId, edge_attr= edge_attr, color = EdgeColors[edge_attr])

        # Leaf of AST Trees
        if node.children == []: #Last node in branch

            # Next Token nodes
            NextTokenNodes.append(node.id)
            
            
            # Get node text
            if "literal" in node.type or "identifier" in node.type or "type" in node.type:
                G.nodes[node.id]["label"] = add_string_literal(node.type + "_" + node_text)
            
            if "identifier" in node.type:
                # Last Lexical Used Nodes
                if node_text in LastLexicalUsedNodes.keys():
                    edge_attr = "LastLexicalUsed"
                    G = make_edge_attribute(G,LastLexicalUsedNodes[node_text], node.id, edge_attr= edge_attr, color = EdgeColors[edge_attr])
                    LastLexicalUsedNodes[node_text] = node.id
                else:
                    LastLexicalUsedNodes[node_text] = node.id

        else:
            for child in node.children:                
                # Make Child Edges
                edge_attr = "Child"
                G = make_edge_attribute(G, node.id, child.id, edge_attr= edge_attr, color = EdgeColors[edge_attr])

                # Add child node into process loop
                todo.append(child)
        
        # Make Computed From Edges
        if "assignment" in node.type:
            CFRoot = [node]
            while CFRoot:
                cfnode = CFRoot.pop()
                if "identifier" in cfnode.type:
                    ComputedFromNodes.append(cfnode.id)

                for child in cfnode.children:
                    CFRoot.append(child)

            ComputedFromNodes.append(node.id)

        if ComputedFromNodes != []:
            edge_attr = "ComputedFrom"
            f_node_id = ComputedFromNodes[-1]
            
            for idx in range(1,len(ComputedFromNodes)):
                s_node_id = ComputedFromNodes[-idx - 1]
                G = make_edge_attribute(G, f_node_id, s_node_id , edge_attr= edge_attr, color = EdgeColors[edge_attr])


            ComputedFromNodes = []
            
    # Make Next Token Edges
    edge_attr = "NextToken"
    for idx in range(len(NextTokenNodes) - 1):
        G = make_edge_attribute(G, NextTokenNodes[-idx], NextTokenNodes[-idx -1], edge_attr= edge_attr, color = EdgeColors[edge_attr])
    
    return G

def graph_transform(source_code : str):
    try:
        tree = parser.parse(bytes(source_code, "utf-8"))
        root = tree.root_node
        
        graph = tree_to_graph(root)
    except:
        graph = None

    return graph

def main():
    vocab_dict = VocabDict()
    w_num = s_num = tag_skip_num = topic_skip_num = 0

    progress_bar = tqdm(os.listdir(DATA_PATH))
    dataset = pyg_graph_dataset(name = "source_code_graph", vocab_dict=vocab_dict)
    
    save_num = 0
    for folder in progress_bar:
        if os.path.isfile(os.path.join(DATA_PATH, folder)) == False:
            for cpp_file in os.listdir(os.path.join(DATA_PATH, folder)):
                if cpp_file.endswith("cpp"):
                    save_num +=1 
                    with open(os.path.join(DATA_PATH, folder, cpp_file), "r") as infile:
                        source_code = infile.read()

                    graph = graph_transform(source_code)
                    #draw_graph(src_code_graph, "type")
                    #nx.drawing.nx_pydot.write_dot(graph, "sample.dot")
                    try:
                        problem_tag = tag_dict[folder]
                    except:
                        problem_tag = None
                        tag_skip_num +=1
                    
                    try:
                        problem_topic = topic_dict[folder]
                    except:
                        problem_topic = None
                        topic_skip_num +=1

                    vocab_dict, written_num, skip_num = dataset.parse(graph, folder, cpp_file, problem_topic, problem_tag)
                    
                    w_num += written_num
                    s_num += skip_num

                    progress_bar.set_postfix({"written" : w_num, "skip" : s_num, "tag_skip" : tag_skip_num, "topic_skip" : topic_skip_num})
                    
        dataset.serialize(filename = folder, dest = OUTPUT_DIR)
        dataset = pyg_graph_dataset(name = "source_code_graph", vocab_dict=vocab_dict)
                
        
    # Save vocab dict
    with open(os.path.join(OUTPUT_DIR + "vocab_dict.pickle"), "wb") as f:
        pickle.dump(vocab_dict, f)

if __name__ == "__main__":
    main()

