import os
import pickle
import itertools

def build_dataset(args, is_train = True):
    data_dir = args.data_dir if is_train else args.eval_data_dir
    dataset = []
    for root, folder, files in os.walk(data_dir):
        for file in files:
            if (file.endswith("pickle")):
                pkl_path = os.path.join(root, file)
                infile = open(pkl_path, "rb")
                graphs = pickle.load(infile)
            
                dataset.append(graphs)

    dataset = list(itertools.chain(*dataset))
    
    dataset = transform_dataset(args,dataset)
    return dataset

def transform_dataset(args,dataset : list):
    if args.add_general_sink_node:
        pass

    if args.add_self_loop:
        pass
    
    return dataset
            