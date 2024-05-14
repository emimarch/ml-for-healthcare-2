import json

# Read the question JSON file
with open('data/dataset/mimic_iv_cxr/train/train_data.json', 'r') as train_file:
    train = json.load(train_file)

# Read the schema JSON file
with open('data/database_schema.json', 'r') as schema_file:
    schema_data = json.load(schema_file)

# Extract relevant information

l = []

schema = schema_data['schema']
schema_items = schema['schema_items']

for i in train: 
        
    text = i['question']
    sql = i['query']

    new_train = {
        'text': text,
        'sql': sql,
        'schema': {
        "schema_items": schema_items
        }
    }

    l.append(new_train)






# Write the final JSON to a new file
with open('data/train_reformatted.json', 'w') as output_file:
    json.dump(l, output_file, indent=4)