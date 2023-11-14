from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader
from network import Network
from plot_losses import PlotLosses

def validation_step(val_loader, net, cost_function):
    '''
        Realiza un epoch completo en el conjunto de validación
        args:
        - val_loader (torch.DataLoader): dataloader para los datos de validación
        - net: instancia de red neuronal de clase Network
        - cost_function (torch.nn): Función de costo a utilizar

        returns:
        - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    '''
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_labels = batch['label'].cuda()    
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            outputs,_=net(batch_imgs)
            costo= cost_function(outputs,batch_labels)
            val_loss+=costo.item()

            preds=outputs.argmax(-1)
            preds.unsqueeze(-1)

            acc = sum(preds == batch_labels) / len(batch_labels) * 100
    # TODO: Regresa el costo promedio por minibatch
    val_loss=val_loss/len(val_loader)
    return val_loss, acc

def train():
    # Hyperparametros
    learning_rate = 1e-4
    n_epochs=100
    batch_size = 512

    # Train, validation, test loaders
    train_dataset, train_loader = \
        get_loader("train",
                    batch_size=batch_size,
                    shuffle=True)
    val_dataset, val_loader = \
        get_loader("val",
                    batch_size=batch_size,
                    shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    # Instanciamos tu red
    modelo = Network(input_dim = 48,
                     n_classes = 7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed'].cuda()
            batch_labels = batch['label'].cuda()
            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            preds, _=modelo(batch_imgs)
            costo=criterion(preds,batch_labels)
            costo.backward()
            optimizer.step()

            # TODO acumula el costo
            train_loss+=costo.item()

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader)
        val_loss, val_accuracy = validation_step(val_loader, modelo, criterion)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}, accuracy: {val_accuracy:.2f}")

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            modelo.save_model('mejor_modelo.pth')
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    plotter.on_train_end()

if __name__=="__main__":
    train()