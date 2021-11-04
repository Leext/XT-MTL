import random
import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
sample = float(sys.argv[3])

dataset = json.load(open(input_path))
num = int(sample*len(dataset))
dataset = random.sample(dataset, num)
json.dump(dataset, open(output_path, 'w'))