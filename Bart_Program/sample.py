import random
import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
sample = float(sys.argv[3])

dataset = json.load(open(input_path))
if sample <= 1.0:
    num = int(sample*len(dataset))
else:
    num = int(sample)
dataset = random.sample(dataset, num)
json.dump(dataset, open(output_path, 'w'))