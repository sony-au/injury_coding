# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:38:10 2024

@author: sjufri
"""

# Function to load CSS from a file
def load_css(file_path):
    '''
    This is function to load CSS styling
    file_path: file path of the CSS file
    '''
    with open(file_path, 'r') as file:
        css = file.read()
    return css