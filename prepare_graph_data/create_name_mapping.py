import os
import json

from sklearn.preprocessing import LabelEncoder

INPUT_DIR = "../data/tags.json"
OUTPUT_DIR = "../data/name_mapping.json"
LE = LabelEncoder()

with open(INPUT_DIR, "r") as infile:
    tags_dataset = json.load(infile)

unique_tags = set()
for tag_list in tags_dataset.values():
    [unique_tags.add(tag) for tag in tag_list]


LE.fit_transform(list(unique_tags))

# Name Mapping
LE_name_mapping = {i: l for i, l in enumerate(LE.classes_)}

with open(OUTPUT_DIR, "w") as outfile:
    json.dump(LE_name_mapping, outfile)
    outfile.close()
