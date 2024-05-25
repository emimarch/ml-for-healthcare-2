import argparse
import os
import torch
import json
import time
import pandas as pd
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm
import sqlite3


# FROM BASELINE FILE
def post_process_sql(query):
    """
    Post-process SQL query before execution.
    """
    current_time = "2105-12-31 23:59:00"
    precomputed_dict = {
        "temperature": (35.5, 38.1),
        "sao2": (95.0, 100.0),
        "heart rate": (60.0, 100.0),
        "respiration": (12.0, 18.0),
        "systolic bp": (90.0, 120.0),
        "diastolic bp": (60.0, 90.0),
        "mean bp": (60.0, 110.0),
    }

    # Handle current_time
    query = query.replace("current_time", f"'{current_time}'")

    # Handle vital signs
    vital_lower_match = re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query)
    vital_upper_match = re.search("[ \n]+([a-zA-Z0-9_]+_upper)", query)

    if vital_lower_match and vital_upper_match:
        vital_lower_expr = vital_lower_match.group(1)
        vital_upper_expr = vital_upper_match.group(1)
        vital_name_list = list(set(re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr) + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)))

        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, str(vital_range[0])).replace(vital_upper_expr, str(vital_range[1]))

    # Handle etc.
    query = query.replace("''", "'").replace("< =", "<=")
    query = query.replace("%y", "%Y").replace("%j", "%J")
    query = query.replace("'", "'").replace("'", "'")
    query = query.replace("\u201c", '"').replace("\u201d", '"')

    return query



# FROM BASELINE FILE

def post_process_answer(answer, round_digit=6, sorted_answer=False):
    """
    Post-process answer before evaluation.
    """
    assert isinstance(answer, list) or answer == "null"

    if answer == "null":
        return answer

    if not answer:
        assert answer == []
        return answer

    # Tuple data preprocessing
    if isinstance(answer[0], tuple):
        assert len(answer[0]) == 1  # NOTE: currently, only support single column output
        answer = [ans[0] for ans in answer]  # unpack tuple

    if isinstance(answer[0], float):
        # Float-type answer
        answer = [round(ans, round_digit) for ans in answer]  # round to specified digit
    elif isinstance(answer[0], str):
        # String-type answer
        if sorted_answer:
            answer = sorted(answer)
    # else:
    #     print(answer)

    return answer


# Returns IDs of table only samples
def get_only_table_samples(val_dataset_path):
    val_dataset = json.load(open(val_dataset_path))

    only_table_val = []
    for entry in val_dataset:
        if("study" in entry["text"] or "studies" in entry["text"] or "x-ray" in entry["text"] or "radiograph" in entry["text"]):
            pass
        else:
            only_table_val.append(entry['id'])
    return only_table_val


def execute_query(database_path, query):
    """
    Execute a SELECT SQL query on the specified SQLite database and return the results.

    Args:
        database_path (str): Path to the SQLite database file.
        query (str): The SQL SELECT query to execute.

    Returns:
        list: A list of tuples representing the rows returned by the query.
    """
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)
    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--database_path', type = str)
    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)
    parser.add_argument('--dset',type = str, default = 'valid')
    
    opt = parser.parse_args()

    return opt

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql

def text2sql_func(model, inputs, tokenizer, max_new_tokens):
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4
        )

    # print(tokenizer.decode(generate_ids[0]))
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    # print(generated_sqls)

    return generated_sqls

if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    dset = opt.dset
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        "eval",
        opt.table_num,
        opt.column_num,
        opt.sic_path
    )

    # TODO: current, we only support batch size = 1

    dataloader = DataLoader(eval_set, batch_size = 1, shuffle=False)
    device = torch.device('cuda:0')
    print('Current device: {}'.format(torch.cuda.current_device()))
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, torch_dtype = torch.float16)
    model.to(device)
    model.eval()
    start_time = time.time()
    predicted_sqls = []
    observation_sql_dict = {}  # Dictionary to store observation ID and generated SQL
    # Get IDs of items with only TABLE queries
    only_table_ids = get_only_table_samples(opt.dataset_path)
    for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
        observation_id = raw_data["id"]  # Extract the observation ID
        #print('Observation: {}'.format(observation_id))
        #print('Question: {}'.format(raw_data['text']))
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        #print('Batch data: ')
        #print(batch_data)
        generated_sqls = text2sql_func(model, batch_data, tokenizer, max_new_tokens)
        generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in generated_sqls]
        generated_sqls = [post_process_sql(generated_sql) for generated_sql in generated_sqls]

        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, opt.database_path)
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql

        # If no executable sql found, just use the first
        if final_generated_sql == None:
            final_generated_sql = generated_sqls[0]
            print('Observation: {}, not found executable query'.format(observation_id))
        

        observation_sql_dict[observation_id] = final_generated_sql  # Save to dictionary
        
        if observation_id % 500: 
            with open('data/{}_generated_queries.json'.format(dset), 'w') as fp:
                json.dump(observation_sql_dict, fp)


    end_time_valid = time.time()        
    # SAVE GENERATE QUERIES

    with open('data/{}_generated_queries.json'.format(dset), 'w') as fp:
        json.dump(observation_sql_dict, fp)

    """
    submission = []

    for observation_id, generated_sql in observation_sql_dict.items():
        # Check if ID is in table only
        if observation_id in only_table_ids: 

            answer = post_process_answer(execute_query(opt.database_path, generated_sql))
        
        else: 
            answer = None

        submission.append({"id": observation_id, "answer": answer})


    result_dir = 'data/'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "{}_prediction.json".format(dset)), "w") as f:
        json.dump(submission, f)
    """
    end_time = time.time()
    
    print("LLM name: {} | Total time: {}s | Total inference time: {}s | Example number: {} | Average time: {}s".format(
        opt.llm_path, 
        end_time - start_time,
        end_time_valid - start_time, 
        observation_id + 1,
        (end_time_valid - start_time) / (observation_id + 1)
        )
    )

