import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


""" For computing the cosine similarity between BERT embeddings"""
def sentence_similarity(sentence1, sentence2, model, tokenizer):
    tokens1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)

    # Obtain BERT embeddings for the tokenized sentences
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

    # Extract the embeddings
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    similarity_score = cosine_similarity(embeddings1,embeddings2)
    return similarity_score

""" Function for building a prompt starting from a question. It takes as value the templates as well as admissible values for the output and placeholders in the templates"""
def build_prompt_from_nlq_zs(question, prompt_file, class_values):
    question_tmp = question
    question_tmp = question_tmp.replace("anatomical finding", "anatomicalfinding")
    question_tmp = question_tmp.replace("technical assessment", "technicalassessment")
    question_tmp = question_tmp.replace("anatomical findings", "anatomicalfinding")
    question_tmp = question_tmp.replace("technical assessments", "technicalassessment")
    template_from_q = question_tmp

    # Extracting the possible values is relevant for those templates where you have to choose a value among two
    # For example, Which anatomical location is related to ${attribute}, the ${object_1} or the ${object_2}?
    # When we have this template, it is convenient to provide to the prompt to the VQA model only the values object_1 and object_2 rather then every value in the class object


    """ 
    Iterate over all the admissible values for each class, and replace it with the corresponding class name in order to build a template starting from the question.
    It doesn't necessarily corresponds to a template from the paper.
    """
    values_from_q = {} # we store the values we substitute in the question
    # We also need to keep track of the changes in the string to actually understand when there is a replacement of values within it
    q_previous_iteration = question_tmp
    class_values = json.load(open(class_values))
    for key,value in class_values.items():
        for v in value:
            template_from_q = template_from_q.replace(v,"${"+key+"}")
            if(q_previous_iteration != template_from_q):
                q_previous_iteration = template_from_q
                try:
                    values_from_q[key].append(v)

                except:
                    values_from_q[key] = []
                    values_from_q[key].append(v)
    admissible_output_values = []
    print(question_tmp)
    print(template_from_q)
    print(values_from_q)
    """ Now get the template with the highest cosine similarity with the one extracted from the question"""
    prompt_values = json.load(open(prompt_file))
    prompt_val_idx = ""
    most_similar_template_cos_sim = -1

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Define some keyword that should influence the cosine similarity and make the results more correct
    keywords_for_sim = ['abnormal','either','both','common','anatomical','which','list',"attribute","category","object","gender"]
    for key, value in prompt_values.items():
        sim = sentence_similarity(template_from_q,value["template"], model, tokenizer)
        """ 
            We decrease or increase the similarity according to the keywords. If the template build from the question and the template from the paper have some specific 
            keywords in common, add, otherwise decrease the similarity
        """
        for k in keywords_for_sim:
            if (k in template_from_q.lower() and k in value["template"].lower()):
                sim = sim + 10
            elif(k not in template_from_q.lower() and k in value["template"].lower()):
                print(k)
                sim = sim - 10
        if(sim>most_similar_template_cos_sim):
            most_similar_template_cos_sim = sim
            prompt_val_idx = key
            print(key)
    print(prompt_val_idx, " ",prompt_values[prompt_val_idx]["value_type"])
    """ Now let's define the admissible values which we are going to pass to the model through the prompt """
    if(prompt_values[prompt_val_idx]["value_type"] == "boolean"):
        admissible_output_values = ["true","false"]
    elif("_from_q" in prompt_values[prompt_val_idx]["value_type"]):
        admissible_output_values = values_from_q[prompt_values[prompt_val_idx]["value_type"].split("_")[0]]
    else:
        admissible_output_values = class_values[prompt_values[prompt_val_idx]["value_type"]]
    list_admissible_output_values = ", ".join(admissible_output_values)
    """ Finally, let's build the prompt"""
    prompt_vqa = " Given an image of a Chest X-ray from a human patient, I need you to answer a question with a value from taken from the following list: "+list_admissible_output_values+".\nQuestion: "+question
    disclaimer = "\n\nBe careful: your answer is going to be used by professional doctors in order to extract information about their patients. Failing to answer correctly will put the life of the patients at risk."
    prompt_vqa = prompt_vqa + disclaimer

    return prompt_vqa

prompt = build_prompt_from_nlq_zs("name all anatomical locations that manifest aspiration, but not costophrenic angle blunting, given the first study of patient 12754610 on 09/this year.", "/home/filippo/PycharmProjects/ml-for-healthcare-2/VQA/prompts/prompt.json", "/home/filippo/PycharmProjects/ml-for-healthcare-2/VQA/prompts/class_values.json")
print(prompt)