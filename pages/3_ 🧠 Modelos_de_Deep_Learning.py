import streamlit as st
from streamlit_option_menu import option_menu
from functions import helpers
import numpy as np
from models.image_recognition.ResNet import *

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


        dataiter = iter(testloader)
        images, labels = next(dataiter)
        tamanho = int(len(images))
        columns = st.columns(10)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) 

        def imshow(img):
            # Unnormalize the image
            img = (img / 2) + 0.5
            # Clip the image data to ensure it's within [0.0, 1.0]
            img = np.clip(img, 0.0, 1.0)
            # Convert from tensor to numpy array
            img = img.numpy()
            return np.transpose(img, (1, 2, 0))
        
         
        if st.button('Predizer 10 Imagens Aleatórias'):
                      
            for i in range(tamanho):
                foto = images[i] / 2 + 0.5
                
                with columns[i]:
                    st.image(imshow(foto), width=100)
                    st.write('**CLASSE REAL:**\n',classes[labels[i].item()], '\n','\n**PREDIÇÃO:**\n',classes[predicted[i].item()],  f'{resultado[predicted[i].item() == labels[i].item()]}')                    
                    

        st.write('---')


        # Carregando notebook no streamlit com expander
        with st.expander('**Notebook Jupyter**'):
            helpers.load_notebook('Notebook/ResNet-image-classification.ipynb')
           


if __name__ == "__main__":
    main()
