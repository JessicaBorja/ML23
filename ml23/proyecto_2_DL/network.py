import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = 7

        # TODO: Define las capas de tu red
        self.net=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 48x48 -> 24x24
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24x24 -> 12x12
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 12x12 -> 12x12
            nn.ReLU(),
            #nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 12x12 -> 12x12
            #nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(start_dim = 1),
            nn.Linear(128*12*12,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512,7),
        )
        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        logits = self.net(x.cuda())
        proba = torch.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save( self.state_dict(),models_path )

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))
