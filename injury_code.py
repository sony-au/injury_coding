# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:48:40 2024

@author: sjufri
"""

import pandas as pd
import pickle

from sentence_embedding import calculate_cosine

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

    # Load the tensor from the file
    with open('./dataset/NatureEmbedding.pkl', 'rb') as f:
        nature_embeddings = pickle.load(f)

    nature_df['Embedding']=[embedding for embedding in nature_embeddings]
    # create embedding and calculate cosine similarity
    cosine_sims=[]
    for embedding in nature_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
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

    # Load the tensor from the file
    with open('./dataset/BodyEmbedding.pkl', 'rb') as f:
        body_embeddings = pickle.load(f)

    body_df['Embedding']=[embedding for embedding in body_embeddings]
    
    # create embedding and calculate cosine similarity
    cosine_sims=[]
    for embedding in body_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
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

    # Load the tensor from the file
    with open('./dataset/MechanismEmbedding.pkl', 'rb') as f:
        mech_embeddings = pickle.load(f)

    mech_df['Embedding']=[embedding for embedding in mech_embeddings]
    
    # create embedding and calculate cosine similarity
    cosine_sims=[]
    for embedding in mech_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
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

    # Load the tensor from the file
    with open('./dataset/AgencyEmbedding.pkl', 'rb') as f:
        agency_embeddings = pickle.load(f)

    agency_df['Embedding']=[embedding for embedding in agency_embeddings]
    
    # create embedding and calculate cosine similarity
    cosine_sims=[]
    for embedding in agency_df['Embedding']:
        cosine_sim=calculate_cosine(agency_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
    agency_df['CosineSimilarity']=cosine_sims
    sorted_agency_df=agency_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_agency_df.loc[:,['Code','Agency','Description']]