# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:42:47 2024

@author: 231829
"""

import pandas as pd
import numpy as np
import torch

from sentence_embedding import create_embedding, calculate_cosine_matrix

def find_industry_code(user_input,
                       n,
                       tokenizer, 
                       model):
    '''
    This is function to obtain the ANZSIC Industry Code for an occupation
    user_input: the industry you are searching for
    n: the top n coding you wish to display
    tokenizer: the tokenizer of the LLM model
    model: the LLM model used
    '''
    # industry data
    industry_df=pd.read_excel('./dataset/anzsic_2006_complete.xlsx')
    
    # Convert column 'Code' to integer type
    industry_df['Code'] = industry_df['Code'].astype(int)
    
    # Load tensors from file
    ind_embeddings = torch.load('./dataset/IndustryEmbedding.pt')
    
    # Load the tensor from the file
    #with open('./dataset/IndustryEmbedding.pkl', 'rb') as f:
    #    ind_embeddings = pickle.load(f)    
    
    industry_df['Embedding']=[embedding for embedding in ind_embeddings]
    
    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input, tokenizer, model)
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(industry_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    industry_df['CosineSimilarity']=cosine_sims
    sorted_industry_df=industry_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_industry_df.loc[:,['Code','Division', 'SubDivision', 'Class', 'Description',]]

