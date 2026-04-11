import json
import pickle

dataset = 'domain'

file_path = f'data\\{dataset}\\partition.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

pretty = json.dumps(data, indent=4)

print(pretty)


