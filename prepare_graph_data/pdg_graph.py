import os

from common import run_system_command

WORKSPACE_DIR = "./workspace/"
def make_pdg_graph(source_code):
    PDG_PATH = os.path.join(WORKSPACE_DIR, "pdg")
    SRC_PATH = os.path.join(WORKSPACE_DIR, "sample.cpp")

    with open(SRC_PATH, "w") as f:
        f.write(source_code)
        f.close()
    
    run_system_command("joern-parse {} --output {} ".format(SRC_PATH, WORKSPACE_DIR + "sample.cpg.bin"))
    run_system_command("joern-export {} --repr pdg --out {}".format(WORKSPACE_DIR + "sample.cpg.bin", PDG_PATH))
    
    for root, folder, file in os.walk(PDG_PATH):
        FILE_PATH = os.path.join(PDG_PATH, file)
        break    
        