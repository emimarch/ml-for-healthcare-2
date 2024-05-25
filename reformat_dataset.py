# This file reformates the chosen dataset by transforming to a CODES runnable dataset. 
# This is done by adding the schema, schema items, and renaming the dictionary entries


import json
import os
import argparse



def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type = str, default='valid')    
    opt = parser.parse_args()
    return opt

def reformat_dataset(dset):
    # Read the question JSON file
    with open('data/dataset/mimic_iv_cxr/{}/{}_data.json'.format(dset, dset), 'r') as mset_file:
        mset = json.load(mset_file)

    # Read the schema JSON file
    with open('data/database_schema.json', 'r') as schema_file:
        schema_data = json.load(schema_file)

    # Extract relevant information

    l = []

    schema = schema_data['schema']
    schema_items = schema['schema_items']


    for i in mset: 
            
        if dset == 'train':
            text = i['question']
            sql = i['query']

            new_mset = {
                'text': text,
                'sql': sql,
                'schema': {
                "schema_items": schema_items
                }
            }

        else: # validation and test set
            text = i['question']
            id = i['id']

            new_mset = {
                'text': text,
                'id': id,
                'schema': {
                "schema_items": schema_items
                }
            }



        l.append(new_mset)




    # Write the final JSON to a new file
    with open('data/{}_reformatted.json'.format(dset), 'w') as output_file:
        json.dump(l, output_file, indent=4)


def main(): 
    opt = parse_option()
    dset = opt.dset

    reformat_dataset(dset)


