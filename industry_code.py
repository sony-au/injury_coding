# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:42:47 2024

@author: 231829
"""

import pandas as pd
import pickle

from sentence_embedding import create_embedding, calculate_cosine

def find_industry_code(user_input,
                       n=15):
    '''
    This is function to obtain the ANZSIC Industry Code for an occupation
    user_input: the industry you are searching for
    n: the top n coding you wish to display
    '''
    # industry data
    industry_df=pd.read_excel('./dataset/anzsic_2006_complete.xlsx')
    
    # Convert column 'Code' to integer type
    industry_df['Code'] = industry_df['Code'].astype(int)
    
    # Load the tensor from the file
    with open('./dataset/IndustryEmbedding.pkl', 'rb') as f:
        ind_embeddings = pickle.load(f)    
    
    industry_df['Embedding']=[embedding for embedding in ind_embeddings]
    
    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input)
    cosine_sims=[]
    for embedding in industry_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
    industry_df['CosineSimilarity']=cosine_sims
    sorted_industry_df=industry_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_industry_df.loc[:,['Code','Division', 'SubDivision', 'Class', 'Description',]]

