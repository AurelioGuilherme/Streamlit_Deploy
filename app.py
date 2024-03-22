import streamlit as st
import torch
import pytorch_lightning as pl
from streamlit_option_menu import option_menu
import numpy as np

# Menu lateral
with st.sidebar:
    selected = option_menu("",['0 - O que são tensores?',
                               '1 - Tensores PyTorch x Arrays NumPy'], 
                           menu_icon="", default_index=0)

def main():
    if selected == '0 - O que são tensores?':
        st.write('# O que são tensores?')
        st.write('''\nEm matemática, um tensor é um objeto geométrico 
                    arbitrariamente complexo que mapeia de maneira (multi) 
                    linear vetores geométricos, escalares e outros tensores 
                    para um tensor resultante.Mas em Deep Learning o conceito 
                    é um pouco diferente e você pode pensar no tensor como um 
                    recipiente que contém números. Um tensor é uma estrutura de 
                    dados que pode ter qualquer número de dimensões.\n''')
        st.image('imagens/tensor.png')

    elif selected == '1 - Tensores PyTorch x Arrays NumPy':
        st.write("# Tensores PyTorch x Arrays NumPy:\n")
        st.write('### Você pode criar tensores a partir de listas ou matrizes numpy e vice-versa.')
        st.write("Cria um tensor 2x3 a partir de uma lista Python")
        lista_python = [[1,2,3], [4,5,6]]
        t1 = torch.tensor(lista_python)  

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

        st.write('Cria um tensor 2x3 a partir de um array Numpy')
        array_numpy = np.array([[9,6,1], [5,3,2]])
        t2 = torch.Tensor(array_numpy)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Cria um tensor 2x3 a partir de um array Numpy                    
array_numpy = np.array([[9,6,1], [5,3,2]])
t2 = torch.Tensor(array_numpy)
print(t2)                    
            ''', language='python')
            st.write(t2)

        st.write('# Tipos de Tensores no PyTorch')
        st.write('Cria um tensor 2 x 2 x 3 com valores randômicos')
        a = torch.rand(2,2,2)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Cria um tensor 2 x 2 x 3 com valores randômicos                   
a = torch.rand(2,2,2)
print(a)                    
            ''', language='python')
            st.write(a)

        st.write('Criando um tensor preenchido com zeros')
        b = torch.zeros(2,2,3)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando um tensor preenchido com zeros                  
b = torch.zeros(2,2,3)
print(b)                    
            ''', language='python')
            st.write(b)

        st.write('Criando um tensor semelhante a outro')
        c = torch.zeros_like(a)
        
        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando um tensor semelhante a outro              
c = torch.zeros_like(a)
print(c)                    
            ''', language='python')
            st.write(c)




if __name__ == "__main__":
    main()
