import csv
import json

from nanugpt import utils

def get_model_sizes():
    csv_filepath = utils.full_path('nanugpt/assets/model_sizes.csv')
    sizes = {}
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            for k,v in row.items():
                row[k] = int(v)
            sizes[row['params_m']] = row
    return sizes

if __name__ == '__main__':
    d = get_model_sizes()
    with open(utils.full_path('nanugpt/assets/model_sizes.json'), mode='w', encoding='utf-8') as json_file:
        json.dump(d, json_file, indent=4)

