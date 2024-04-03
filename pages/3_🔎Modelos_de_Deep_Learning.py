import streamlit as st
import torch

PAGE_TITLE = 'Modelos de Deep Learning 🔎'
PAGE_ICON = "🔎"
st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON)

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