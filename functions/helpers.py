import streamlit as st
import numpy as np



def code_python(string):
    st.code(string,language='python')

def print_tensor(tensor, string_custom=None, name=None, numpy=True, dim = False):
    if name is None:
        name = 'Tensor'
    else:
        name = str(name)
    if string_custom == None:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'**Tensor: {name}**')
            if numpy:
                st.write(tensor.numpy())
            else:
                st.write(tensor)
        
        with col2:
            st.write('**Size:**', tensor.size())
            if dim:
                st.write('**Quantidade de dimensões:**',tensor.dim())

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write(string_custom)
            if numpy:
                st.write(tensor.numpy())
            else:
                st.write(tensor)
        with col2:
            st.write('**Size:**', tensor.size())
            if dim:
                st.write('**Quantidade de dimensões:**',tensor.dim())

        



    
        
