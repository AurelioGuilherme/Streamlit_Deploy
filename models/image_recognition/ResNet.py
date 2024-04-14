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

from torchvision import datasets

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
