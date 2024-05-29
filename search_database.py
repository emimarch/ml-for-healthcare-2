# Give the LLM generated sql queries, this function runs the queries and saves the database results into the submission folder
# WIP
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from VQA.utils.vqa_utils import get_labels, call_vqa_model
import torch
from urllib.request import urlopen
from PIL import Image
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


import argparse
import os
import json
import time
import pandas as pd
import re
from tqdm import tqdm
import sqlite3
from PIL import Image
import random
from collections import Counter


custom_home = '/c/Users/aemil/Downloads/ML_for_healthcare/huggingface'
#model, preprocess = create_model_from_pretrained('C:/Users/aemil/Downloads/ML_for_healthcare/huggingface/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir = custom_home)
#tokenizer = get_tokenizer('C:/Users/aemil/Downloads/ML_for_healthcare/huggingface/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir = custom_home)
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
similarity_model = BertModel.from_pretrained('bert-base-uncased')

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type = str, default='valid')
    parser.add_argument('--imgset', type=str, default='all')
    opt = parser.parse_args()

    global DSET
    global IMGSET
    DSET = opt.dset
    IMGSET = opt.imgset
    return opt

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

def find_image_from_folder(study_id):
    directory_path = 'data/vqa/{}/images/{}/'.format(IMGSET, study_id)
    images = []
    # Iterate through all objects in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        try:
            image = Image.open(item_path)
            images.append(image)
        except: 
            continue

    return images
    


def vqa(image, texts, labels):

    context_length = 256
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    model.to(device)
    model.eval()

    image = torch.stack([preprocess(image)]).to(device)
    texts = tokenizer(texts, context_length=context_length).to(device)

    answer = call_vqa_model(model, image, texts, labels)
    if(answer == "True"):
        answer = 1
    elif(answer == "False"):
        answer = 0
    return answer



def find_template_option(labels):
    if labels == [0,1]: 
        return 2
    if len(labels) == 2: 
        if labels != ["male", "female"] and labels != ["AP","PA"]:
            return 3
        else: 
            return 1
    return None

def get_answer(images,texts, labels):
    if len(images) == 1:
        answer = vqa(images[0], texts, labels)
        return answer
    elif len(images) > 1:
        # Opzioni:
        # 1) Quali condizioni ci sono in questo studio => concatena output di ognuna =>  se label len(>= 2), and if two solo gender or solo view
        # 2) Questo studio ha questa condizione? => True is at least one True = > se label True or False
        # 3) Quele tra queste due condizione, pneumonia o tumore, c'é nello studio? => se len > 3 majority vote se len == 2 random, => se label len >= 2 oe non gener of view
        option = find_template_option(labels)
        if option == None:
            print('No template found')
            return ''
        if option == 1:  # concatena e set, return list
            answer = []
            for image in images:
                img_answer = vqa(image, texts, labels)
                answer.extend(img_answer)
            answer = list(set(answer))
            return answer
        elif option == 2:  # True se almeno una True, return one
            answer = False
            for image in images:
                img_answer = int(vqa(image, texts, labels))
                answer = img_answer or answer
            return answer
        elif option == 3:  # return one
            answer = []
            for image in images:
                img_answer = vqa(image, texts, labels)
                answer.append(img_answer)
            if len(answer) == 2:
                rind = random.randint(0, 1)
                answer = answer[rind]
            else:  # majority vote
                counter = Counter(answer)

                # Find the item with the most counts
                most_common_item, _ = counter.most_common(1)[0]

                answer = most_common_item

            return answer

    else:
        return ''


def func_vqa(sub_query, study_id):
    # Find image from folders
    images = find_image_from_folder(study_id) # list

    labels, admissible_answers = get_labels(sub_query, similarity_model=similarity_model, tokenizer=bert_tokenizer)


    if (len(labels) <= 2):
        prompt_template = sub_query + 'Answer: '
        # We treat the problem as a classification problem, where the prompt is given by the combination of quetion + label
        # The model will assign the highest score to the combination that matches the image the best.
        texts = [prompt_template + l for l in labels]
        answers = get_answer(images, texts, labels)

    else:
        answers = []
        for l in labels:
            bool_labels = ['False', 'True']
            prompt_template = sub_query + f' Is there {l}? Answer: '
            texts = [prompt_template + bl for bl in bool_labels]
            answer = get_answer(images, texts, bool_labels)
            #print(" L ",l," Answer ", answer)
            if(answer == 1):
                answers.append(l)
        answers = list(set(answers))
        #print(" ANSWERS MULTI ",answers)
        #print(" ADMISSIBLE VALUES ", admissible_answers)
        """ If the question is of the type: are there any x in y?
        In this case, we check for the presence of every x value, put those that are present in the list and return true if the list is not empty """
        if(admissible_answers == ["False","True"]):
            if(len(answers) > 0):
                answers = 1
            else:
                answers = 0
        else:
            #answers = ", ".join([a for a in answers])
            answers = json.dumps(answers)

    #print(" ANSWER ",answers)
    return answers




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

    # ADD FUNC VQA

    connection.create_function("func_vqa", 2, func_vqa)
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



""" main of the logic for generating queries """
def main(): 
    opt = parse_option()
    dset = opt.dset
    only_table_ids = get_only_table_samples('data/{}_reformatted.json'.format(dset))
    print(only_table_ids)
    observation_sql_dict = json.load(open('data/{}_generated_queries.json'.format(dset)))


    submission = []
    for observation_id, generated_sql in tqdm(observation_sql_dict.items()):
        #if(observation_id == '42'):
        # Check if ID is in table only
        generated_sql = post_process_sql(generated_sql)
        #if int(observation_id) in only_table_ids:
        #print('Observation ID: {}'.format(observation_id))

        try:
            answer = post_process_answer(execute_query('data/database/mimic_iv_cxr/silver/mimic_iv_cxr.db', generated_sql))
        except:
            answer = None

        submission.append({"id": observation_id, "answer": answer})


    result_dir = 'data/submissions/{}/'.format(dset)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "{}_prediction.json".format(dset)), "w") as f:
        json.dump(submission, f)



if __name__ == "__main__":
    main()