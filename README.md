# TEXT-TO-SQL and VQA for Electronic Health Records (EHR) and Chest X-Ray Scans

## Project description

Project competition link: https://www.codabench.org/competitions/2902/?secret_key=eac86d58-aedb-4380-8c23-08f44d26b13d

For this project, we decided to fine-tune a pre-trained codes-1b model on the MIMIC-IV dataset for EHR question answering (text-to-SQL). This fine-tuned model achieves very good performace for table questions, image questions, and table + image questions across all metrics. After creating the SQL queries, we query the database. If a question is an image or table + image question, we call BiomedCLIP-PubMedBERT to perform the VQA. 

## SET-UP

1) Create and activate Python 3.8.5 environment
2) Install dependencies
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install -r codes/requirements.txt

## TRAINING
Fine tuning of codes-1b on MIMIC-IV. 

```sh
python reformat_dataset.py --dset train
training_script.sh
```
## VALIDATION AND TESTING 

python reformat_dataset.py --dset valid OR python reformat_dataset.py --dset test
testing_script.sh (modify dataset in testing_script accordingly, --dset argument)

## QUERY DATABASE 

1) First, indentify VQA queries and save them, in addition to the necessary x-ray images
```sh
python search_fvqa_queries.py --dset valid
```
OR 
```sh
python search_fvqa_queries.py --dset test
```

2) Then, query the database 
```sh
python search_database.py --dset valid
```
OR 
```sh
python search_database.py --dset test
```




