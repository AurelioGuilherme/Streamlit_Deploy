import streamlit as st
import torch
import pytorch_lightning as pl
from streamlit_option_menu import option_menu
import numpy as np

# Menu lateral
with st.sidebar:
    selected = option_menu("",['0 - O que são tensores?',
                               '1 - Tensores PyTorch x Arrays NumPy',
                               '2 - Trabalhando com as dimensões dos Tensores'], 
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
        a = torch.rand(2,2,3)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Cria um tensor 2 x 2 x 3 com valores randômicos                   
a = torch.rand(2,2,3)
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

        st.write("Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões)")
        d = torch.ones_like(a)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões)             
d = torch.ones_like(a)
print(d)                    
            ''', language='python')
            st.write(d, unsafe_allow_html=True)

        st.write('Criando tensores de diferentes tipos')
        a = np.array([[4,5,6], [7,8,9]])
        b = torch.Tensor(a)
        c = torch.FloatTensor(a)
        d = torch.LongTensor(a)


        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando tensores de diferentes tipos            
a = np.array([[4,5,6], [7,8,9]])
b = torch.Tensor(a)
c = torch.FloatTensor(a)
d = torch.LongTensor(a)
print(f'b: {b.type()}')
print(f'c: {c.type()}')
print(f'd: {d.type()}')                   
            ''', language='python')
            st.write(f" b: {b.type()}", unsafe_allow_html=True)
            st.write(f" c: {c.type()}", unsafe_allow_html=True)
            st.write(f" d: {d.type()}", unsafe_allow_html=True)
            
        st.write('Criando tensor a partir da lista de booleanos')
        e = [True, False,True, True, True, False]
        f = torch.Tensor(e)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando tensor a partir da lista de booleanos          
e = [True, False, True, True, False]
f = torch.Tensor(e)
print(f)
print(f.type())           
            ''', language='python')
            st.write(f)
            st.write(f.type())
        
        st.write('Criando tensor com valores booleanos')
        g = torch.zeros(10, dtype = torch.bool)

        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Criando tensor com valores booleanos         
g = torch.zeros(10, dtype = torch.bool)
print(g)
print(g.type())           
            ''', language='python')
            st.write(g)
            st.write(g.type())
        
        st.write('Alterando o tipo do tensor:')
        a = np.array([[4,5,6], [7,8,9]])
        c = torch.FloatTensor(a)
        valor = c
        c = c.long()
        
        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Alterando o tipo do tensor:
a = np.array([[4,5,6], [7,8,9]])
c = torch.FloatTensor(a)
print(c.type())
c = c.long()
print(c)
print(c.type())                                     
            ''', language='python')
            st.write(valor.type())
            st.write(c)
            st.write(c.type())
            
    elif selected == '2 - Trabalhando com as dimensões dos Tensores':
        st.write('# Trabalhando com as dimensões dos Tensores')
        st.write('### Size e Shape de Tensores')
        st.write('Visualizando as dimensões dos tensores com `shape` e `size()`')
        st.write('`shape`: é um atributo')
        st.write('`size()`:  é um método')
        
        # Cria um tensor com valores randômicos
        torch.manual_seed(777)
        x = torch.randint(0, 10, size = (2, 3, 4))

# Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Cria um tensor com valores randômicos
torch.manual_seed(777)
x = torch.randint(0, 10, size = (2, 3, 4))                   
print(x)
                    
# Shape - Atributo do objeto tensor
x.shape

# Size - Método do objeto tensor
x.size()                    

# Número total de elementos no Tensor                                   
torch.numel(x)

# Alterando o size do tensor (mas sem mudar o tensor original)
print(x.view(2, 2, 6))                                   
            ''', language='python')
            st.write(x)
            st.write(x.shape)
            st.write(x.size())
            st.write(torch.numel(x))


        st.write('### View')
        st.write('Alterando o size do tensor (mas sem mudar o tensor original) com `view()`')
        
        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
import torch
                                         
# Cria um tensor com valores randômicos
torch.manual_seed(777)
x = torch.randint(0, 10, size = (2, 3, 4))                   

# Alterando o size do tensor (mas sem mudar o tensor original)
print(f"Altera de '{x.size()}' para '{x.view(2, 2, 6).size()}'\n")     
print(x.view(2, 2, 6))                                                     
            ''', language='python')
            st.write(f"Altera de '{x.size()}' para '{x.view(2, 2, 6).size()}'\n")
            st.write(x.view(2, 2, 6))

        st.write('Também podemos usar o método `view()` para criar um tensor')
        t = torch.arange(60).view(3, 4, 5)
        with st.expander('Mostrar código:'):
            st.code('''
import torch

t = torch.arange(60).view(3, 4, 5)
print(t)
print(t.shape)
print(t.size())
print(torch.numel(t))
''', language= 'python')
            st.write(t) 
            st.write(t.shape) 
            st.write(t.size()) 
            st.write(torch.numel(t))


        
        def create_tensor(dimensions):
            try:
                return torch.randint(low=0, high=11, size=dimensions)
            except ValueError:
                st.error('Digite dimensões válidas para o tensor (números inteiros separados por vírgula).')

        st.write("### Slicing de Tensores")
        st.write('Criando um tensor com dimensões customizadas e com seed definida.')
        st.write('Valores entre: 0 a 10')
        torch.manual_seed(222)
        dim_input = st.text_input("Digite as dimensões do tensor separadas por vírgula (ex: 3,4,5):")

        if dim_input:
            dimensions = [int(dim) for dim in dim_input.split(',') if dim.isdigit()]
            dimensions = [max(0, min(dim, 9999999)) for dim in dimensions]
            x = create_tensor(dimensions)
            st.write(x)
            st.write('**Fazendo slice**')
            st.write(f'O tensor possui: {x.ndim} dimensões') 
            slice_input = st.text_input("Digite os valores para o slice do tensor separados por vírgula (limitado à quantidade de dimensões):")
            try:
                if slice_input:
                    slices = [int(s) for s in slice_input.split(',') if s.isdigit()]
                    result = x[tuple(slices)]  # Convertendo para uma tupla para indexação
                    st.write("Resultado do slicing:", result)
            except IndexError:
                st.error("Índices de slice inválidos.")




    
            







if __name__ == "__main__":
    main()
