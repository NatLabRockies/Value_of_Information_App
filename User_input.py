import os
import io
import pandas as pd
import numpy as np
import streamlit as st

def st_file_selector(st_placeholder, path='.', label='Please, select a file/folder...'):
    # get base path (directory)
    base_path = '.' if path == None or path == '' else path
    base_path = base_path if os.path.isdir(
        base_path) else os.path.dirname(base_path)
    base_path = '.' if base_path is None or base_path == '' else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    if base_path != '.':
        files.insert(0, '..')
    files.insert(0, '.')
    selected_file = st_placeholder.selectbox(
        label=label, options=files, key=base_path)
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file == '.':
        return selected_path
    if os.path.isdir(selected_path):
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                         path=selected_path, label=label)
    return selected_path


def Prior_probability_binary(mykey=None): #x_sample, X_train,
    """
    Function for slider of prior probability of "positive"
    Key : string
        key is given for when it get's called multiple times (e.g. demonstration, then for posterior calculations)
    """    

    
    Pr_POS = st.slider('Choose :blue[Prior probability of successful geothermal well]', float(0.00),float(1.0), float(0.1), float(0.01),key=mykey)

    return Pr_POS 

def make_value_array(count_ij, profit_drill_pos= 2e6, cost_drill_neg = -1e6):
    """
    make value_array with 
        rows= NUMBER OF decision alternatives, 1st is do nothing, 2nd drill
        columns = equal to subsurface conditions (decision variables), 1st Negative, 2nd Postive 
        number_a : int 
            number of decision alternatives
    """
    number_a = 2 # set at do nothing or drill 
    value_array = np.zeros((number_a, np.shape(count_ij)[0]))
    
    value_array[0,:] = [0, 0]
    value_array[1,:] = [cost_drill_neg, profit_drill_pos] 
    
    index_labels = ['do nothing','drill']
    value_array_df = pd.DataFrame(value_array,index=index_labels,columns=['negative','positive'])
    
    return value_array, value_array_df