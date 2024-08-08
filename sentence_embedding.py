# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 07:56:54 2024

@author: sjufri
"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

# function to create embeddings
def create_embedding(text):
    '''
    This is function to create sentence embedding from a text
    text: the text to be converted into embedding
    '''
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding

# function to calculate cosine similarity
def calculate_cosine(embedding_1,embedding_2):
    '''
    This is function to calculate cosine similarity between two embeddings
    embedding_1: the first embedding
    embedding_2: the second embedding
    '''
    # Compute cosine similarity
    # Calculate the dot product
    dot_product = np.dot(embedding_1,embedding_2)
    
    # Calculate the magnitudes (norms) of the embeddings
    norm1 = np.linalg.norm(embedding_1)
    norm2 = np.linalg.norm(embedding_2)
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

### LOAD MODEL ###
model_name='./model/sentence-transformers-all-MiniLM-L6-v2/'

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
