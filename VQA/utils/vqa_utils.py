import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import sys


def levenshtein_distance(word1, word2):
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)

    # Initialize matrix of zeros
    distances = np.zeros((len(word1) + 1, len(word2) + 1), dtype=int)

    # Populate matrix of distances (first row and first column)
    for i in range(len(word1) + 1):
        distances[i][0] = i
    for j in range(len(word2) + 1):
        distances[0][j] = j

    # Iterate over the matrix to compute the cost
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(distances[i - 1][j], distances[i][j - 1], distances[i - 1][j - 1])

    return distances[len(word1)][len(word2)]


def is_typo(phrase, candidate, max_distance=2):
    return levenshtein_distance(phrase, candidate) <= max_distance


def generate_ngrams(sentence, max_n=5):
    # Split the sentence into words
    words = sentence.split()

    # Initialize an empty list to store the n-grams
    ngrams = []

    # Generate n-grams for n from 1 to max_n
    for n in range(1, max_n + 1):
        ngrams.extend([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])

    return ngrams

def find_potential_typos(sentence, reference_phrase, max_distance=2):
    candidates = generate_ngrams(sentence)
    typos = [candidate for candidate in candidates if is_typo(reference_phrase, candidate, max_distance)]
    return typos

""" For computing the cosine similarity between BERT embeddings"""
def sentence_similarity(sentence1, sentence2, model, tokenizer, device):
    tokens1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)
    tokens1 = {key: value.to(device) for key, value in tokens1.items()}
    tokens2 = {key: value.to(device) for key, value in tokens2.items()}

    # Obtain BERT embeddings for the tokenized sentences
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

    # Extract the embeddings
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    similarity_score = cosine_similarity(embeddings1.cpu(),embeddings2.cpu())
    return similarity_score

""" Function for building a prompt starting from a question. It takes as value the templates as well as admissible values for the output and placeholders in the templates"""
def get_labels(question, similarity_model, tokenizer, prompt_file='VQA/prompts/templates.json', class_values_file='VQA/prompts/class_values.json'):
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
    class_values = json.load(open(class_values_file))
    for key,value in class_values.items():
        for v in value:
            if(v in template_from_q):
                template_from_q = template_from_q.replace(v,"${"+key+"}")
            elif(len(v) > 4):  # check if there is a typo of the value, and let's replace it. Let's ignore values that are too small

                typos = find_potential_typos(template_from_q, v)
                #print(" V ",v, " typo ",typos)
                for t in typos:
                    template_from_q = template_from_q.replace(t,"${" + key + "}")
            if(q_previous_iteration != template_from_q):
                q_previous_iteration = template_from_q
                try:
                    values_from_q[key].append(v)

                except:
                    values_from_q[key] = []
                    values_from_q[key].append(v)
    admissible_output_values = []
    """ Now get the template with the highest cosine similarity with the one extracted from the question"""
    prompt_values = json.load(open(prompt_file))
    prompt_val_idx = ""
    most_similar_template_cos_sim = -1
    #print(" PARTIAL TEMPLATE ", template_from_q)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    similarity_model.to(device)

    # Define some keyword that should influence the cosine similarity and make the results more correct
    keywords_for_sim = ['all','abnormal','either','both','common','anatomical','which','list',"attribute","category","object","gender", 'any', ' or ',' and ']
    for key, value in prompt_values.items():
        sim = sentence_similarity(template_from_q,value["template"], similarity_model, tokenizer, device)
        """ 
            We decrease or increase the similarity according to the keywords. If the template build from the question and the template from the paper have some specific 
            keywords in common, add, otherwise decrease the similarity
        """
        for k in keywords_for_sim:
            if (k in template_from_q.lower() and k in value["template"].lower()):
                sim = sim + 10
            elif(k not in template_from_q.lower() and k in value["template"].lower()):
                sim = sim - 10
        if(sim>most_similar_template_cos_sim):
            most_similar_template_cos_sim = sim
            prompt_val_idx = key

    """ Let's get the question type """
    question_type = prompt_values[prompt_val_idx]["label_type"]

    value_types = prompt_values[prompt_val_idx]["answer_value_type"]
    template = prompt_values[prompt_val_idx]["template"]
    #print(" TEMPLATE ",template)
    """ Now let's define the admissible values which we are going to pass to the model through the prompt """
    if(value_types == "boolean"):
        admissible_answers = ["False","True"]
    elif("_from_q" in value_types):
        admissible_answers = values_from_q[value_types.split("_")[0]]
    else:
        admissible_answers = class_values[value_types]

    if(question_type == 'multi'):
        if('location' in template):
            input_values = class_values[value_types]
        elif('abnormal' in template):
            input_values = class_values['anatomicalfinding']
        else:
            input_values = [item for sublist in [class_values[category] for category in values_from_q['category']] for item in sublist]
    else:
        input_values = admissible_answers
    #list_admissible_output_values = ", ".join(admissible_output_values)
    # """ Finally, let's build the prompt"""
    #prompt_vqa = " Given an image of a Chest X-ray from a human patient, I need you to answer a question with a value from taken from the following list: "+list_admissible_output_values+".\nQuestion: "+question
    #disclaimer = "\n\nBe careful: your answer is going to be used by professional doctors in order to extract information about their patients. Failing to answer correctly will put the life of the patients at risk."
    #prompt_vqa = prompt_vqa + disclaimer

    return input_values, admissible_answers

def call_vqa_model(model, image, texts, labels):
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, texts)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        sorted_indices = sorted_indices.cpu().numpy()

    top_label = labels[sorted_indices[0][0]]
    return top_label