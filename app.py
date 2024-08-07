from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st
import numpy as np
import re
import pickle

#st.write("Imports successful!")

# CSS styling 
style = """
<style scoped>
    .dataframe-div {
      max-height: 350px;
      max-width: 1050px;
      overflow: auto;
      position: relative;
    }

    .dataframe {
      font-size: 12px; /* Adjust base font size here */
    }

    .dataframe thead th {
      position: -webkit-sticky; /* for Safari */
      position: sticky;
      top: 0;
      background: #2ca25f;
      color: white;
      font-size: 13px; /* Font size for table headers */
    }

    .dataframe tbody td, .dataframe tbody th {
      font-size: 13px; /* Font size for table body cells */
    }

    .dataframe thead th:first-child {
      left: 0;
      z-index: 1;
    }

    .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
    }

    .dataframe tbody tr th {
      position: -webkit-sticky; /* for Safari */
      position: sticky;
      left: 0;
      background: #99d8c9;
      color: white;
      vertical-align: top;
    }
</style>
"""

# function to create embeddings
def create_embedding(text):
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
    # Compute cosine similarity
    # Calculate the dot product
    dot_product = np.dot(embedding_1,embedding_2)
    
    # Calculate the magnitudes (norms) of the embeddings
    norm1 = np.linalg.norm(embedding_1)
    norm2 = np.linalg.norm(embedding_2)
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

def preprocess_text(text):
    text = re.sub(r' {2,}', ' ', text) # Remove extra spaces
    return text

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

# Define the options for the tabs
tabs = ["ANZSCO Occupation Code", "ANZSIC Industry Code", "TOOCS Code"]

# Create a selectbox or radio button in the sidebar for tab selection
selected_tab = st.sidebar.selectbox("Select a tool:", tabs)

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
    sorted_occ_df=occ_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)

    data=sorted_occ_df.loc[:,['Code','Occupation','UnitGroup','MinorGroup','SubMajorGroup','MajorGroup']]

    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        st.write(f"**Here are the top 15 occupation code for {user_input}:**")
        #st.dataframe(data, hide_index=True)  # Adjust the width as needed
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
            
elif selected_tab == "ANZSIC Industry Code":
    # Use Markdown headers to increase text size
    st.write("## ANZSIC Industry Code")
    
    # Create a text input field with default value and placeholder
    user_input=''
    user_input = st.text_input(
        '**Enter the industry you are looking for:**',
        placeholder="Type your input here...",
    )
    
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
    sorted_industry_df=industry_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)
    
    data=sorted_industry_df.loc[:,['Code','Division', 'SubDivision', 'Class', 'Description',]]

    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        st.write(f"**Here are the top 15 industry code for {user_input}:**")
        #st.dataframe(data, hide_index=True)  # Adjust the width as needed
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
            
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
    
    ### NATURE OF INJURY ###
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
    scenario_embedding=create_embedding(scenario_input)
    cosine_sims=[]
    for embedding in nature_df['Embedding']:
        cosine_sim=calculate_cosine(scenario_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
    nature_df['CosineSimilarity']=cosine_sims
    sorted_nature_df=nature_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)
    
    data=sorted_nature_df.loc[:,['Code','BodilyLocation','Description']]

    # Display the entered text and the DataFrame with adjustable width
    if scenario_input!='':
        html_content = """
            <div style='margin-top: 0px;margin-bottom: 0px;'>
                <h2 style='font-size: 16px;'>Top 15 nature of injury codes:</h2>
            </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
            
    ### BODY LOCATION OF INJURY ###
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
    sorted_body_df=body_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)
    
    data=sorted_body_df.loc[:,['Code','BodyPart','Description']]

    # Display the entered text and the DataFrame with adjustable width
    if scenario_input!='':
        html_content = """
            <div style='margin-top: 10px;margin-bottom: 0px;'>
                <h2 style='font-size: 16px;'>Top 15 body location of injury codes:</h2>
            </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
    
    ### MECHANISM OF INJURY ###
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
    sorted_mech_df=mech_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)
    
    data=sorted_mech_df.loc[:,['Code','Mechanism','Description']]

    # Display the entered text and the DataFrame with adjustable width
    if scenario_input!='':
        html_content = """
            <div style='margin-top: 10px;margin-bottom: 0px;'>
                <h2 style='font-size: 16px;'>Top 15 mechanism of injury codes:</h2>
            </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
            
    ### AGENCY OF INJURY ###
    # agency of injury data
    agency_df=pd.read_excel('./dataset/Proposed_TOOCS_spreadsheet_updated.xlsx', sheet_name='Agency')

    # Load the tensor from the file
    with open('./dataset/AgencyEmbedding.pkl', 'rb') as f:
        agency_embeddings = pickle.load(f)

    agency_df['Embedding']=[embedding for embedding in agency_embeddings]
    
    # create embedding and calculate cosine similarity
    user_embedding=create_embedding(user_input)
    cosine_sims=[]
    for embedding in agency_df['Embedding']:
        cosine_sim=calculate_cosine(user_embedding,embedding)
        cosine_sims.append(cosine_sim)
    
    agency_df['CosineSimilarity']=cosine_sims
    sorted_agency_df=agency_df.sort_values(by=['CosineSimilarity'], ascending=False).head(15)
    
    data=sorted_agency_df.loc[:,['Code','Agency','Description']]

    # Display the entered text and the DataFrame with adjustable width
    if user_input!='':
        # Display the HTML content
        html_content = """
            <div style='margin-top: 10px;margin-bottom: 0px;'>
                <h2 style='font-size: 16px;'>Top 15 agency of injury codes:</h2>
            </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        #st.write('##### ')
        #st.write("**Top 15 agency of injury codes:**")
        
        # Use custom CSS to control table width
        data_html = data.to_html(escape=False, index=False)
        data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
        st.markdown(f"""
            {data_html}
            """, unsafe_allow_html=True)
            
            


