import streamlit as st
import numpy as np



def code_python(string):
    st.code(string,language='python')

def print_tensor(tensor, name=None, numpy=True):
    if name is None:
        name = 'Tensor'
    else:
        name = str(name)

    if numpy:
        st.write(f'**Tensor: {name}**',tensor.numpy(),'**Size**',tensor.size())
    else:
        st.write(f'**Tensor: {name}**',tensor,'**Size**',tensor.size())


    
        
