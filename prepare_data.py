import os
import json
import numpy as np
from pathlib import Path
import pandas as pd
import shutil

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Intialize PATHs
GRAPH_PATH = "./graph_data/graphs/"
PROBLEM_TOPIC_PATH = "./graph_data/problem_topics.json"
TOPIC_PATH = "./graph_data/topics.json"
OUTPUT_DIR = Path("./data/")
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_FOLDS = 2
NUM_CLASS = 6

# Set seed for reproducing
seed = 69
np.random.RandomState(seed = seed)

# Topic file
with open(PROBLEM_TOPIC_PATH, "r") as infile:
    problem_topic_dict = json.load(infile)
    infile.close()
# Topic Name
with open(TOPIC_PATH, "r") as infile:
    topic_dict = json.load(infile)
    infile.close()

df = list()

for file in os.listdir(GRAPH_PATH):
    sample = dict()

    problem = file.split(".")[0]
    problem_dir = os.path.join(GRAPH_PATH, file)
    topic_list = problem_topic_dict[problem]

    sample["problem"] = problem
    sample["problem_dir"] = problem_dir
    sample["topic"] = list()
    for idx in range(len(topic_list)):
        if topic_list[idx] == 1:
            sample["topic"].append(list(topic_dict.keys())[idx])
    
    df.append(sample)

df = pd.DataFrame(df, index=None)
df.to_csv(OUTPUT_DIR / "full_data.csv")

unique_topics = set([topic for sublist in df['topic'] for topic in sublist])

for topic in unique_topics:
    df[topic] = df['topic'].apply(lambda x: 1 if topic in x else 0)

X, Y = df[["problem", "problem_dir"]], df[list(unique_topics)]

msss = MultilabelStratifiedKFold(n_splits= 2, random_state=seed, shuffle = True)

df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(msss.split(X,Y)):
    df.loc[val_idx, "fold"] = fold

# Create Train/Test Folds
for fold in range(NUM_FOLDS):
    fold_path = f"{OUTPUT_DIR}/fold_{fold}/"
    train_path = f"{fold_path}train/"
    val_path = f"{fold_path}val/"
    
    os.makedirs(fold_path, exist_ok= True)
    os.makedirs(train_path, exist_ok= True)
    os.makedirs(val_path, exist_ok= True)

    train_df = df[df["fold"] != fold].reset_index(drop = True)
    val_df = df[df["fold"] == fold].reset_index(drop = True)

    for index, row in train_df.iterrows():   
        src_dir = row["problem_dir"]

        destination_dir = f"{train_path}"
        os.makedirs(destination_dir, exist_ok=True)
        
        shutil.copy(src_dir, destination_dir)
    
    for index, row in val_df.iterrows():   
        src_dir = row["problem_dir"]

        destination_dir = f"{val_path}"
        os.makedirs(destination_dir, exist_ok=True)
        
        shutil.copy(src_dir, destination_dir)





         