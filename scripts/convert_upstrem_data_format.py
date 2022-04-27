import json
from tqdm import tqdm

sents = []
in_fname = "data/CSR_upstream_train.json"
out_fname = "data/csr_upstream_train_lines.json"
with open(in_fname) as fd:
    lines = fd.readlines()
    for line in tqdm(lines, desc="Loading Data"):
        sents.append(json.loads(line))
formatted = []
for i, sent in enumerate(tqdm(sents, desc="Converting format")):
    instance = []
    instance.append(sent['input_text'])
    instance.append([sent['output_text'],])
    instance.append(f"Dev|DummyID|#{i}")
    formatted.append(instance)
    # formatted.append(sent)
with open(out_fname, 'w') as fd:
    fd.write(json.dumps(formatted, indent=4))
