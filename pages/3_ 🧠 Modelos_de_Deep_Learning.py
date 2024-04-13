import streamlit as st
import torch
from streamlit_option_menu import option_menu
from nbconvert import HTMLExporter
import nbformat
import codecs
from functions import helpers
import pickle
import numpy as np
import torchvision
import torchvision.transforms as transforms
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



PAGE_TITLE = 'Modelos de Deep Learning 🧠'
PAGE_ICON = "🧠"
MENU_LIST = ['Sobre',
             "1 - Visão computacional - ResNet"]
ICON_LIST = ["🧠","👀"]
st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON, layout="wide")

# --- LOAD MODEL ---







# --- LOAD DATA ---

with open('./Data/cifar10/dados_test', mode = 'rb') as file:
    data = pickle.load(file, encoding = 'bytes')

X = data[b'data']
y = np.array(data[b'labels'])
raw_images = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")








# --- PAGE VIEW ---

with st.sidebar:
    selected = option_menu("",
                           MENU_LIST,icons=ICON_LIST,
                           default_index=0)
    
def main():
    if selected == 'Sobre':
        st.title('Modelos de Deep Learning')

        st.write('''
            Neste espaço, estou compartilhando diversos projetos nos quais utilizei o 
            PyTorch para explorar uma variedade de problemas de Deep Learning. Desde 
            tarefas mais simples, como classificação e regressão, até abordagens mais 
            complexas, como visão computacional, previsão de séries temporais, processamento 
            de linguagem natural e reconhecimento de imagens.

            Cada projeto aqui apresentado reflete não apenas meu conhecimento técnico, 
            mas também meu compromisso em encontrar soluções eficazes e inovadoras para 
            problemas do mundo real. Espero que esses exemplos demonstrem minha habilidade 
            em lidar com dados, minha criatividade na abordagem de desafios e minha dedicação 
            em entregar resultados de alta qualidade.            
                ''')
        
    elif selected == '1 - Visão computacional - ResNet':
        st.title('ResNet (Rede Neural Residual): Classificação de imagens')

        st.write('''
                   A ResNet é uma arquitetura de rede neural convolucional (CNN) proposta por 
                   Kaiming He et al. em 2015. Ela introduziu o conceito de blocos residuais, 
                   que permitem que a rede aprenda as diferenças entre as características de 
                   entrada e de saída em vez de tentar aprender as características originais 
                   diretamente. Isso facilita o treinamento de redes muito profundas, 
                   alcançando ótimos resultados em diversas tarefas de visão computacional, 
                   incluindo classificação de imagens.
                ''')
        st.write('''
                   Para explorar o potencial da ResNet para reconhecimento de imagens, 
                   utilizarei o CIFAR-10, que é um conjunto de dados comumente utilizado 
                   para benchmarking em visão computacional. Consiste em 60.000 imagens coloridas 
                   de 32x32 pixels, divididas em 10 classes, como carros, pássaros, gatos, entre 
                   outros. O `CIFAR10DataModule`, disponível na biblioteca PyTorch Lightning Bolts, 
                   simplifica o carregamento e a preparação desses dados para treinamento de modelos.
                   ''')
        st.write('''[Link CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)''')
        st.write('''
                   ### Classes do Conjunto de Dados CIFAR-10
                   
                   | Número | Classe       |
                   |--------|--------------|
                   | 0      | Avião        |
                   | 1      | Automóvel    |
                   | 2      | Pássaro      |
                   | 3      | Gato         |
                   | 4      | Cervo        |
                   | 5      | Cachorro     |
                   | 6      | Sapo         |
                   | 7      | Cavalo       |
                   | 8      | Navio        |
                   | 9      | Caminhão     |
                ''')
        st.write('---')

        classes = {0: 'Avião',
                   1: 'Automóvel',
                   2: 'Pássaro',
                   3: 'Gato',
                   4: 'Cervo',
                   5: 'Cachorro',
                   6: 'Sapo',
                   7: 'Cavalo',
                   8: 'Navio',
                   9: 'Caminhão'}





        
        if st.button('Selecionar 10 Imagens Aleatórias'):
           # Selecionar 10 índices aleatórios
           random_indices = np.random.choice(len(raw_images), size=10, replace=False)
           
           # Exibir as imagens e as classes previstas
           columns = st.columns(10)  # Dividir a largura disponível em 10 colunas

           # Exibir as imagens selecionadas em colunas separadas       
           for i, idx in enumerate(random_indices):
                with columns[i]:  # Exibir cada imagem em uma coluna separada
                    st.image(raw_images[idx], caption=f'Imagem {idx+1}: {classes[y[idx]]}', width=100, use_column_width=False)
                    


        st.write('---')

        


        # Carregando notebook no streamlit com expander
        with st.expander('**Notebook Jupyter**'):
            
            helpers.load_notebook('Notebook/ResNet-image-classification.ipynb')
           


if __name__ == "__main__":
    main()
