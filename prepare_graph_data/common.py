import os
import sys
from subprocess import call
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

def run_system_command(cmd: str,
                       shell: bool = False,
                       err_msg: str = None,
                       verbose: bool = True,
                       split: bool = True,
                       stdout=None,
                       stderr=None) -> int:
    """
    :param cmd: A string with the terminal command invoking an external program
    :param shell: Whether the command should be executed through the shell
    :param err_msg: Error message to print if execution fails
    :param verbose: Whether to print the command to the standard output stream
    :param split: Whether to split the tokens in the command string
    :param stdout: file pointer to redirect stdout to
    :param stderr: file pointer to redirect stderr to
    :return: Return code
    """
    if verbose:
        sys.stdout.write("System cmd: {}\n".format(cmd))
    if split:
        cmd = cmd.split()
    rc = call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    if err_msg and rc:
        sys.stderr.write(err_msg)
        exit(rc)
    return rc

def initialize_cpp_tree_sitter():
    TREE_SITTER_PATH = "./tree-sitter-cpp"
    Path(TREE_SITTER_PATH).mkdir(exist_ok= True)
    TREE_SITTER_CPP_URL = "https://github.com/tree-sitter/tree-sitter-cpp.git"


    # Intial tree sitter
    from tree_sitter import Language

    if os.path.exists(TREE_SITTER_PATH) == False:
        run_system_command("git clone {}".format(TREE_SITTER_CPP_URL), verbose = False)

    Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
        TREE_SITTER_PATH
    ]
    )

    CPP_LANGUAGE = Language('build/my-languages.so', "cpp")

    return CPP_LANGUAGE

def draw_graph(G, label=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.axis(False)
    pos = nx.nx_agraph.graphviz_layout(G, 'dot')
    if label:
        labels = nx.get_node_attributes(G, label)
    else:
        labels = None
    nx.draw_networkx(G, pos, ax=ax, font_size=8, labels=labels, node_color="white")

def add_string_literal(s : str):
    return '"' + s + '"'

def remove_string_literal(s:str):
    return s[1:len(s) - 1]
