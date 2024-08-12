# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:48:40 2024

@author: sjufri
"""

import pandas as pd
import numpy as np
import torch

from sentence_embedding import calculate_cosine_matrix

def nature_injury_code(scenario_embedding,
                       n):
    '''
    This is function to obtain the TOOCS nature of injury code for an accident/injury
    scenario_input: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    
    # nature of injury data
    nature_df=pd.read_excel('./dataset/Proposed_TOOCS_spreadsheet_updated.xlsx', sheet_name='NatureofInjury')
    nature_df.rename(columns={'TOOCSCode':'Code'}, inplace=True)

    # Convert column 'Code' to integer type
    nature_df['Code'] = nature_df['Code'].astype(int)
    
    # Load tensors from file
    nature_embeddings = torch.load('./dataset/NatureEmbedding.pt')

    # Load the tensor from the file
    #with open('./dataset/NatureEmbedding.pkl', 'rb') as f:
    #    nature_embeddings = pickle.load(f)

    nature_df['Embedding']=[embedding for embedding in nature_embeddings]
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(nature_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    nature_df['CosineSimilarity']=cosine_sims
    sorted_nature_df=nature_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_nature_df.loc[:,['Code','BodilyLocation','Description']]

def body_injury_code(scenario_embedding,
                     n):
    '''
    This is function to obtain the TOOCS body location of injury code for an accident/injury
    scenario_embedding: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    # body of injury data
    body_df=pd.read_excel('./dataset/Proposed_TOOCS_spreadsheet_updated.xlsx', sheet_name='BodilyLocation')
    
    # Load tensors from file
    body_embeddings = torch.load('./dataset/BodyEmbedding.pt')

    # Load the tensor from the file
    #with open('./dataset/BodyEmbedding.pkl', 'rb') as f:
    #    body_embeddings = pickle.load(f)

    body_df['Embedding']=[embedding for embedding in body_embeddings]
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(body_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    body_df['CosineSimilarity']=cosine_sims
    sorted_body_df=body_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_body_df.loc[:,['Code','BodyPart','Description']]

def mechanism_injury_code(scenario_embedding,
                          n):
    '''
    This is function to obtain the TOOCS mechanism of injury code for an accident/injury
    scenario_embedding: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    # mechanism of injury data
    mech_df=pd.read_excel('./dataset/Proposed_TOOCS_spreadsheet_updated.xlsx', sheet_name='MechanismofIncident')
    
    # Load tensors from file
    mech_embeddings = torch.load('./dataset/MechanismEmbedding.pt')

    # Load the tensor from the file
    #with open('./dataset/MechanismEmbedding.pkl', 'rb') as f:
    #    mech_embeddings = pickle.load(f)

    mech_df['Embedding']=[embedding for embedding in mech_embeddings]
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(mech_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    mech_df['CosineSimilarity']=cosine_sims
    sorted_mech_df=mech_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_mech_df.loc[:,['Code','Mechanism','Description']]

def agency_injury_code(agency_embedding,
                       n):
    '''
    This is function to obtain the TOOCS agency of injury code for an accident/injury
    agency_embedding: the sentence embedding for the agency of the injury
    n: the top n coding you wish to display
    '''
    # agency of injury data
    agency_df=pd.read_excel('./dataset/Proposed_TOOCS_spreadsheet_updated.xlsx', sheet_name='Agency')
    
    # Load tensors from file
    agency_embeddings = torch.load('./dataset/AgencyEmbedding.pt')

    # Load the tensor from the file
    #with open('./dataset/AgencyEmbedding.pkl', 'rb') as f:
    #    agency_embeddings = pickle.load(f)

    agency_df['Embedding']=[embedding for embedding in agency_embeddings]
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(agency_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, agency_embedding[0])
    
    agency_df['CosineSimilarity']=cosine_sims
    sorted_agency_df=agency_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_agency_df.loc[:,['Code','Agency','Description']]

def icd_code(scenario_embedding,
             n):
    '''
    This is function to obtain the ICD injury code for an accident/injury
    scenario_input: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    
    # icd injury data
    icd_df=pd.read_csv('./dataset/valid_icd_10.csv')
    
    #import time
    #start_time = time.time()
    
    # Load tensors from file
    icd_embeddings = torch.load('./dataset/ICDEmbedding.pt')
    # Load the tensor from the file
    #with open('./dataset/ICDEmbedding.pkl', 'rb') as f:
    #    icd_embeddings = pickle.load(f)

    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {elapsed_time} seconds")
    icd_df['Embedding']=icd_embeddings
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(icd_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    icd_df['CosineSimilarity']=cosine_sims
    sorted_icd_df=icd_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_icd_df.loc[:,['Code', 'ShortDescription','LongDescription']]