import streamlit as st
from streamlit_option_menu import option_menu
from functions import helpers
import pickle
import numpy as np




PAGE_TITLE = 'Modelos de Deep Learning 🧠'
PAGE_ICON = "🧠"
MENU_LIST = ['Sobre',
             "1 - Visão computacional - ResNet"]
ICON_LIST = ["🧠","👀"]
st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON, layout="wide")

# --- LOAD MODEL ---
# Sistema e manipulação de arquivos
import os
import warnings
warnings.filterwarnings("ignore")

# Manipulação e processamento de dados
import numpy as np
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

# Modelagem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule


from torch.optim.lr_scheduler import OneCycleLR


# Avaliação do modelo
from torchmetrics.classification import Accuracy

# Visualização de dados
import matplotlib.pyplot as plt
from torchvision import datasets
from PIL import Image


# Módulo para carregar um modelo pré-treinado de arquitetura ResNet sem os pesos (queremos somente a arquitetura)
def carrega_modelo_pretreinado():
    modelo = torchvision.models.resnet18(weights = None, num_classes = 10)
    modelo.conv1 = nn.Conv2d(3, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
    modelo.maxpool = nn.Identity()
    return modelo

# Classe com Arquitetura do Modelo
class ModeloResnet(LightningModule):
    
    # Método construtor
    def __init__(self, lr = 0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = carrega_modelo_pretreinado()

    # Método Forward
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim = 1)

    # Método de um passo de treinamento
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    # Método de avaliação
    def evaluate(self, batch, stage = None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim = 1)
        accuracy = Accuracy(task = "multiclass", num_classes = 10).to(device)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar = True)
            self.log(f"{stage}_acc", acc, prog_bar = True)

    # Método de um passo de validação
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    # Método de um passo de teste
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # Método de configuração do otimizador
    def configure_optimizers(self):
        
        # Otimização SGD
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr = self.hparams.lr, 
                                    momentum = 0.9, 
                                    weight_decay = 5e-4)
        
        # Passos por época
        steps_per_epoch = 45000 // BATCH_SIZE
        
        # Scheduler
        scheduler_dict = {
            "scheduler": OneCycleLR(optimizer,
                                    0.1,
                                    epochs = self.trainer.max_epochs,
                                    steps_per_epoch = steps_per_epoch),
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

test_data_pipeline = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization()
    ]
)

testset  = datasets.CIFAR10(root='./Data/cifar10/teste',train = False,download=False, transform=test_data_pipeline)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)
model = ModeloResnet()
model.load_state_dict(torch.load('./models/image_recognition/saved_models/modelo_dl.pth'))
model.eval()



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
        
         
        if st.button('Selecionar 10 Imagens Aleatórias'):
                      
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
