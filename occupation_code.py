# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:07:05 2024

@author: sjufri
"""

import streamlit as st
import pandas as pd
import pickle

from sentence_embedding import create_embedding, calculate_cosine

def find_occupation_code(user_input,
                         n=15):
    '''
    This is function to obtain the ANZSCO Occupation Code for an occupation
    user_input: the occupation you are searching for
    n: the top n coding you wish to display
    '''
    # occupation data
    occ_df=pd.read_excel('./dataset/anzsco_2022_structure_062023_complete.xlsx')

    # Convert column 'Code' to integer type
    occ_df['Code'] = occ_df['Code'].astype(int)

    # Load the tensor from the file
    with open('./dataset/OccupationEmbedding.pkl', 'rb') as f:
        occ_embeddings = pickle.load(f)

    occ_df['Embedding']=[embedding for embedding in occ_embeddings]

    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input)
    cosine_sims=[]
    for embedding in occ_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)

    occ_df['CosineSimilarity']=cosine_sims
    sorted_occ_df=occ_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_occ_df.loc[:,['Code','Occupation','UnitGroup','MinorGroup','SubMajorGroup','MajorGroup']]

def occupation_code(selected_tab):
    # Use Markdown headers to increase text size
    st.write("## ANZSCO Occupation Code")
    
    # Create a text input field with default value and placeholder
    user_input=''
    user_input = st.text_input(
        '**Enter the occupation you are looking for:**',
        placeholder="Type your input here...",
    )
    
    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        ### OCCUPATION CODE ###
        # get output
        data=find_occupation_code(user_input)
        
        # display output
        display_output(data,text_output=f"Here are the top 15 occupation code for {user_input}:",margin_top=0)