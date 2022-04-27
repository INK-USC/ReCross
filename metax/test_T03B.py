from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

import json

with open('/home/beiwen/MetaCross/data/super_glue-copa/test-1000.json') as f:
    data = json.load(f)

for i in range(50):
    # add a for-loop here to go through the first 50 examples in our test.1000 file
    inputs = tokenizer.encode(data[i][0], return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

