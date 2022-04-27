import os 
import pickle
import faiss
import numpy as np 

from metax.models.utils import pickle_load, pickle_save 

import sys 

path_memory = sys.argv[1] # memory_cache/sbert_memory.pkl
num_shards = int(sys.argv[2])
base_path = f"{path_memory}.$i_of_{num_shards}"
save_path = f"{path_memory}"
 


def load_memory_from_path(cache_path):
    print("Loading cached index manager memory")
    cache = pickle_load(cache_path)
    return cache

all_memory = {}
for i in range(num_shards):
# for i in range(2):
    current_path = base_path.replace("$i", str(i))
    print(current_path)
    memory = load_memory_from_path(current_path)
    dim_memory = memory["dim_memory"]
    memory_index = memory["memory_index"]
    memory_examples = memory["memory_examples"]
    memory_example_ids = memory["memory_example_ids"]
    # print(memory_example_ids)
    
    if "dim_memory" not in all_memory:
        all_memory["dim_memory"] = dim_memory
    else:
        assert all_memory["dim_memory"] == dim_memory
    
    if "memory_index" not in all_memory:
        all_memory["memory_index"] = list(memory_index)
    else:
        all_memory["memory_index"] += list(memory_index)  # all vectors now.
 
    if "memory_examples" not in all_memory:
        all_memory["memory_examples"] = memory_examples
    else:
        all_memory["memory_examples"].update(memory_examples)
    
    if "memory_example_ids" not in all_memory:
        all_memory["memory_example_ids"] = memory_example_ids
    else:
        all_memory["memory_example_ids"] += memory_example_ids


print(len(all_memory["memory_index"]))
print(len(all_memory["memory_example_ids"]))

memory_index = faiss.IndexFlatL2(all_memory["dim_memory"])  
vectors = np.array(all_memory["memory_index"])
memory_index.add(vectors)

all_memory["memory_index"] = memory_index
    
pickle_save(all_memory, save_path)