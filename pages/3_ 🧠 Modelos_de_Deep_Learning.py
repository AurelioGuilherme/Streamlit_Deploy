import streamlit as st
from streamlit_option_menu import option_menu
from functions import helpers
import numpy as np
from models.image_recognition.ResNet import *
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
        
import plotly.graph_objects as go

# --- CONFIG PAGE LAYOUT ---
PAGE_TITLE = 'Modelos de Deep Learning 🧠'
PAGE_ICON = "🧠"
MENU_LIST = ['Sobre',
             "1 - Visão computacional - ResNet"]
ICON_LIST = ["🧠","👀"]

st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON, layout="wide")


# --- LOAD MODEL --- #
model = ModeloResnet()
model.load_state_dict(torch.load('./models/image_recognition/saved_models/modelo_dl.pth'))
model.eval()


# --- LOAD DATA --- #
test_data_pipeline = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization()
    ]
)

testset  = datasets.CIFAR10(root='./Data/cifar10/teste',train = False,download=False, transform=test_data_pipeline)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)



# --- PAGE VIEW --- #

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

        columns = st.columns(2)
        with columns[0]:
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
        with columns[1]:
            st.write('''
                        O modelo é construído com base na **ResNet-18** sem os pesos pré-treinados permitindo
                         que a arquitetura seja utilizada, foi adicionado 2 camadas adicionais na arquiterura original,
                        a primeira uma camada convolucional adicionando um valor de kernel com 3 canais de entrada 
                        (correspondentes aos canais RGB das imagens) e 64 canais de saída, com um kernel size de (3, 3), 
                        um stride de (1, 1) e um padding de (1, 1). 
                     
                        O kernel **(camada convolucional)** é uma pequena matriz de números que desliza sobre a imagem, multiplicando-se com os 
                        pixels correspondentes e somando os resultados para gerar um único valor na imagem de saída desta forma 
                        reduzindo a quantidade de parâmetros a serem aprendidos, aumentando a eficiência computacional e melhorando
                        a capacidade de detectar características independentemente da sua localização exata na imagem.
                     
                        A outra camada adcional é a **camada de identidade** que não realiza nenhuma operação nos dados de entrada,
                        agindo como um "pass-through" para preservar as características aprendidas pelas camadas anteriores, isso é feito
                        para manter a dimensionalidade dos dados.
                     
                        Foi utilizado 20 epocas com learning rate de 0.05, foi utilzado uma GPU Nvidia GTX 1080 para treinamento, 
                        que levou cerca de 50 minutos para conclui-lo.
                     
                     ''')
        
            st.write('[Documentação ResNet](https://pytorch.org/vision/main/models/resnet.html)')
        st.write('---')
        st.write('## Predizendo as Classes CIFAR 10 com modelo ResNet:\n')
        st.write('')

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
        
        resultado = ['❌','✅']

        # Predizendo as classes
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        tamanho = int(len(images))
        columns = st.columns(10)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) 

        def imshow(img):
            img = (img / 2) + 0.5
            img = np.clip(img, 0.0, 1.0)
            img = img.numpy()
            return np.transpose(img, (1, 2, 0))
        
         
        if st.button('Predizer 10 Imagens Aleatórias'):
                      
            for i in range(tamanho):
                foto = images[i] / 2 + 0.5
                
                with columns[i]:
                    st.image(imshow(foto), width=100)
                    st.write('**CLASSE REAL:**\n',classes[labels[i].item()], '\n','\n**PREDIÇÃO:**\n',classes[predicted[i].item()],  f'{resultado[predicted[i].item() == labels[i].item()]}')                    
                    

        st.write('---')

        st.write('### Resultado do modelo')
        st.write('''
                    O Modelo apresentou uma acurácia de **91,14%** em dados de teste.
                 
                ''')
        def show_metrics(path_metrics):
            # Carrega as métricas
            metricas = pd.read_csv(path_metrics)
            del metricas["step"]
            metricas.set_index("epoch", inplace = True)
            plt.figure(figsize=(1, 1))
            fig = sns.relplot(data = metricas, kind = "line")
            plt.show()
           
            columns = st.columns(2)
            with columns[0]:
                st.pyplot(fig)


        
     
        

        show_metrics('./models/image_recognition/saved_models/results/logs/lightning_logs/version_11/metrics.csv')

                # Carregando notebook no streamlit com expander
        with st.expander('**Notebook Jupyter**'):
            helpers.load_notebook('Notebook/ResNet-image-classification.ipynb')
           


if __name__ == "__main__":
    main()
