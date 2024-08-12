# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 07:56:54 2024

@author: sjufri
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_embedding import create_embedding
from occupation_code import find_occupation_code
from industry_code import find_industry_code
from injury_code import nature_injury_code, body_injury_code, mechanism_injury_code, agency_injury_code, icd_code
from display_output import display_output

# Define the options for the tabs
tabs = ["ANZSCO Occupation Code", "ANZSIC Industry Code", "TOOCS Code", "ICD Code"]

# Create a selectbox or radio button in the sidebar for tab selection
selected_tab = st.sidebar.selectbox("Select a tool:", tabs)

# Create a list of numbers from 1 to 20
options = list(range(1, 21))

### LOAD MODEL ###
model_name='./model/sentence-transformers-all-MiniLM-L6-v2/'

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Display content based on the selected tab
if selected_tab == "ANZSCO Occupation Code":
    # Use Markdown headers to increase text size
    st.write("## ANZSCO Occupation Code")
    
    # Create a text input field with default value and placeholder
    user_input=''
    user_input = st.text_input(
        '**Enter the occupation you are looking for:**',
        placeholder="Type your input here...",
    )
    
    # Selectbox for choosing the no of codes to display
    top_n_codes = st.selectbox('**The number of codes to display:**', options, index=options.index(15))
    
    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        ### OCCUPATION CODE ###
        # get output
        data=find_occupation_code(user_input, top_n_codes, tokenizer, model)
        
        # display output
        display_output(data,text_output=f"Here are the top 15 occupation code for {user_input}:",margin_top=0)
            
elif selected_tab == "ANZSIC Industry Code":
    # Use Markdown headers to increase text size
    st.write("## ANZSIC Industry Code")
    
    # Create a text input field with default value and placeholder
    user_input=''
    user_input = st.text_input(
        '**Enter the industry you are looking for:**',
        placeholder="Type your input here...",
    )
    
    # Selectbox for choosing the no of codes to display
    top_n_codes = st.selectbox('**The number of codes to display:**', options, index=options.index(15))

    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        ### INDUSTRY CODE ###
        # get output
        data=find_industry_code(user_input, top_n_codes, tokenizer, model)
        
        # display output
        display_output(data,text_output=f"Here are the top 15 industry code for {user_input}:",margin_top=0)
            
elif selected_tab == "TOOCS Code":
    # Use Markdown headers to increase text size
    st.write("## TOOCS Code")
    
    # Create a text input field with default value and placeholder
    scenario_input=''
    scenario_input = st.text_area("**Enter the accident scenario and the resulting injury:**",
                                  placeholder="Type your input here...")
    user_input=''
    user_input = st.text_input(
        '**Enter the agency of the injury:**',
        placeholder="Type your input here...",
    )
    
    # Selectbox for choosing the no of codes to display
    top_n_codes = st.selectbox('**The number of codes to display:**', options, index=options.index(15))
    
    # Display the entered text and the DataFrame with adjustable width
    if scenario_input!='':
        ### NATURE OF INJURY ###
        # get output
        scenario_embedding=create_embedding(scenario_input, tokenizer, model)
        data=nature_injury_code(scenario_embedding, top_n_codes)
        
        # display output
        display_output(data,text_output='Top 15 nature of injury codes:',margin_top=0)
            
        ### BODY LOCATION OF INJURY ###
        # get output
        data=body_injury_code(scenario_embedding, top_n_codes)
        
        # display output
        display_output(data,text_output='Top 15 body location of injury codes:')
            
        ### MECHANISM OF INJURY ###
        # get output
        data=mechanism_injury_code(scenario_embedding, top_n_codes)
        
        # display output
        display_output(data,text_output='Top 15 mechanism of injury codes:')
            
    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        ### AGENCY OF INJURY ###
        # get output
        agency_embedding=create_embedding(user_input, tokenizer, model)
        data=agency_injury_code(agency_embedding, top_n_codes)
        
        # display output
        display_output(data,text_output='Top 15 agency of injury codes:')
        
elif selected_tab == "ICD Code":
    # Use Markdown headers to increase text size
    st.write("## ICD Code")
    
    user_input=''
    user_input = st.text_input(
        '**Enter the diagnosis of the injury:**',
        placeholder="Type your input here...",
    )
    
    # Selectbox for choosing the no of codes to display
    top_n_codes = st.selectbox('**The number of codes to display:**', options, index=options.index(15))
    
    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        ### ICD INJURY CODE###
        # get output
        icd_embedding=create_embedding(user_input, tokenizer, model)
        data=icd_code(icd_embedding, top_n_codes)
        
        # display output
        display_output(data,text_output='Top 15 ICD codes:',margin_top=0)
