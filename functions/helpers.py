import streamlit as st
import numpy as np
from nbconvert import HTMLExporter
import nbformat
import codecs



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

def load_notebook(path_notebook):
    # Carregando o notebook
    with codecs.open(path_notebook, 'r', 'utf-8') as notebook_file:
        notebook_content = notebook_file.read()
        notebook = nbformat.reads(notebook_content, as_version=4)

    # Convertendo o notebook para HTML
    html_exporter = HTMLExporter()
    html_body, _ = html_exporter.from_notebook_node(notebook)

    # Exibindo o conteúdo do notebook como HTML
    st.components.v1.html(html_body, height=600,scrolling=True)#width=1100, height=600, scrolling=True)
        



    
        
