# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:22:50 2024

@author: sjufri
"""

import streamlit as st
from load_css import load_css

# CSS styling 
css=load_css('styles.css')
style=f'<style scoped>{css}</style>'

def display_output(data,
                   text_output='',
                   margin_top=10,
                   margin_bottom=0,
                   font_size=16):
    '''
    This is function to display streamlit output
    data: the output data
    text_output: the text output to display
    margin_top: the top margin of the text
    margin_bottom: the bottom margin of the text
    font_size: the font size
    '''
    html_content = f"""
        <div style='margin-top: {margin_top}px;margin-bottom: {margin_bottom}px;'>
            <h2 style='font-size: {font_size}px;'>{text_output}</h2>
        </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
    data_html = data.to_html(escape=False, index=False)
    data_html = style+'<div class="dataframe-div">'+data_html+"\n</div>"
    st.markdown(f"""
        {data_html}
        """, unsafe_allow_html=True)