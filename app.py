import streamlit as st
import torch
import pytorch_lightning as pl
from streamlit_option_menu import option_menu

# Menu lateral
with st.sidebar:
    selected = option_menu("",['0 - O que são tensores?',
                               '1 - Tensores PyTorch x Arrays NumPy'], 
                           menu_icon="", default_index=0)

def main():
    if selected == '0 - O que são tensores?':
        st.write('## O que são tensores?')
        st.write('''\nEm matemática, um tensor é um objeto geométrico 
                    arbitrariamente complexo que mapeia de maneira (multi) 
                    linear vetores geométricos, escalares e outros tensores 
                    para um tensor resultante.Mas em Deep Learning o conceito 
                    é um pouco diferente e você pode pensar no tensor como um 
                    recipiente que contém números. Um tensor é uma estrutura de 
                    dados que pode ter qualquer número de dimensões.\n''')
        st.image('imagens/tensor.png')

    elif selected == '1 - Tensores PyTorch x Arrays NumPy':
        st.write("# **Tensores PyTorch x Arrays NumPy:**\n")
        st.write('### Você pode criar tensores a partir de listas ou matrizes numpy e vice-versa.')
        st.write("Cria um tensor 2x3 a partir de uma lista Python")
        lista_python = [[1,2,3], [4,5,6]]
        t1 = torch.tensor(lista_python)
        #st.write(t1)
        
        # Mostrar o código
        with st.expander("Mostrar Código"):
            st.code("""
import torch

# Cria um tensor 2x3 a partir de uma lista Python
lista_python = [[1,2,3], [4,5,6]]
t1 = torch.tensor(lista_python)
print(t1)
            """, language="python")
            st.write(t1)

if __name__ == "__main__":
    main()
