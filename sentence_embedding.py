# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 07:56:54 2024

@author: sjufri
"""

import torch
import torch.nn.functional as F
import numpy as np

# function to create embeddings
def create_embedding(text, 
                     tokenizer, 
                     model):
    '''
    This is function to create sentence embedding from a text
    text: the text to be converted into embedding
    tokenizer: the tokenizer of the LLM model
    model: the LLM model used
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
def calculate_cosine_matrix(embedding_matrix, query_embedding):
    '''
    Calculate cosine similarity between a query_embedding and each row of embedding_matrix.
    embedding_matrix: 2D numpy array where each row is an embedding
    query_embedding: 1D numpy array representing the embedding to compare against
    '''
    # Normalize the embeddings
    norm_embeddings = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norm_query = np.linalg.norm(query_embedding)
    
    normalized_embeddings = embedding_matrix / norm_embeddings
    normalized_query = query_embedding / norm_query
    
    # Compute cosine similarity
    cosine_similarities = np.dot(normalized_embeddings, normalized_query)
    
    return cosine_similarities


def mean_pooling(model_output, attention_mask):
    '''
    The Mean Pooling is to take attention mask into account for correct averaging
    model_output: the model output
    attention_mask: the attention mask
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
