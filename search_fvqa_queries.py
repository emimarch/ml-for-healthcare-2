# This file searches for the fvqa queries from the genereted CODES queries and: 
# 1) Saves the VQA queries in data/vqa/{set}
# 2) Saves the associated x-ray scans into data/vqa/{set}/images, which each image named with the corresponding study_id




import numpy as np
import json 
import os
import argparse
#from search_database import get_only_table_samples
import re
import pandas as pd
import shutil




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

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type = str, default='valid')    
    opt = parser.parse_args()
    return opt



def return_fvqa_queries(set_path, query_path): 

    queries = json.load(open(query_path))
    all_ids = set(queries.keys())
    table_queries_ids = set(get_only_table_samples(set_path))

    vqa_queries_ids = list(all_ids - table_queries_ids)
    vqa_queries = {k: queries[k] for k in vqa_queries_ids}

    return vqa_queries



# Returns a list of study ids or patient ids from the cxr database, which are mentioned in the vqa queries
def return_study_ids_or_patient_ids(vqa_queries): 
    pattern_sid = re.compile(r'\btb_cxr\.study_id\b\s*=\s*(\d+)')
    pattern_pid = re.compile(r'\btb_cxr\.subject_id\b\s*=\s*(\d+)')
    studies_ids = []
    patient_ids = []
    for qid, query in vqa_queries.items(): 
        match_sid = pattern_sid.search(query)
        match_pid = pattern_pid.search(query)

        if match_sid:
            studies_ids.append(int(match_sid.group(1)))
        if match_pid: 
            patient_ids.append(int(match_pid.group(1)))

    
    return studies_ids, patient_ids


# This copies the files from the image folder into the vqa image folder, where each image is named using the study_id
def copy_files(source_dir, target_dir, file_names, study_ids):
    """
    Copy files from source_dir to target_dir based on a filename pattern.
    
    :param source_dir: Directory to search for files.
    :param target_dir: Directory to copy files to (created if it doesn't exist).
    :param pattern: Filename pattern to match (e.g., ".txt" for all text files).
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    
    # Copy files that match the pattern
    for i, file_name in enumerate(file_names):
        target_path = '{}{}/'.format(target_dir, study_ids[i])
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            #print(f"Created target directory: {target_dir}")

        source_path = os.path.join(source_dir, file_name + '.jpg') # collect image from images
        target_path = os.path.join(target_path ,'{}.jpg'.format(file_name)) # save image into vqa_images/set/{study_id}/ using the study_id
        #print(source_path)
        #print(target_path)
        #return
        # Copy the file
        try: 
            shutil.copy(source_path, target_path)
            print(f"Copied {file_name} to {target_dir}")
        except: 
            print(f'Not found {file_name}')


def main(): 
    opt = parse_option()
    dset = opt.dset
    set_path = 'data/{}_reformatted.json'.format(dset)
    query_path = 'data/{}_generated_queries.json'.format(dset)
    vqa_queries_filepath = 'data/vqa/{}/'.format(dset)

    image_mapping = pd.read_csv('data/mimic_cxr_metadata.csv')
    # Get vqa queries 
    vqa_queries = return_fvqa_queries(set_path, query_path)
    # Save vqa queries
    os.makedirs(vqa_queries_filepath, exist_ok=True)
    with open(os.path.join(vqa_queries_filepath, "vqa_queries.json"), "w") as f:
        json.dump(vqa_queries, f)


    # Given the vqa queries, find the study ids or patient ids from the CXR database so that you can collect the images

    studies_ids, patient_ids = return_study_ids_or_patient_ids(vqa_queries)
  

    selected_images = image_mapping[(image_mapping['subject_id'].isin(patient_ids)) | (image_mapping['study_id'].isin(studies_ids))].dicom_id.values


    # Select the study ids of the collected images

    selected_images_sids = image_mapping[image_mapping['dicom_id'].isin(selected_images)].study_id.values

    print(len(selected_images_sids))


    # Select images from image folder and save them in a separate folder

    # copy_files('/run/media/filippo/Seagate Basic/resized_ratio_short_side_768/resized_ratio_short_side_768/', 'data/vqa/{}/images/'.format(dset), selected_images, selected_images_sids)
    
    copy_files('data/images/resized_ratio_short_side_768/', 'data/vqa/{}/images/'.format(dset), selected_images, selected_images_sids)




if __name__ == "__main__":
    main()








