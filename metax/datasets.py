from typing import List, Optional
import json

def get_formatted_data(names: Optional[List] = None, t:str='upstream', num:int=2000) -> dict:
    """
    Return a dict containing subsets with specified size and split in spcified datasets.

    Use a custom Numpy generator is strongly recommended.
    """
    import json
    import os
    formatted = {}
    if t == 'fewshot':
        filename = 'fewshot.json'
    else:
        filename = f'{t}-{num}.json'
    for name in names:
        print(f"Processing dataset {name} (using data/{name}/{filename})")
        if os.path.isfile(f'data/{name}/{filename}'):
            with open(f'data/{name}/{filename}', mode='r') as f:
                formatted[name] = json.load(f)

    return formatted

def get_all_upstream_data_from_single_file(filename: Optional[str] = None) -> dict:
    assert filename
    with open(filename, 'r') as fd:
        return json.load(fd)
