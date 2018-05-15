#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as tor

'''
Le modèle apprend à reconnaitre à partir de données en entrée et en sortie
Le modèle apprend jusqu'à une précision K
'''

class LayerNet(tor.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LayerNet, self).__init__()
        self.linear1 = tor.nn.Linear(D_in, H)
        self.linear2 = tor.nn.Linear(H,D_out)

    #Learning 1 Step forward
    def forward(self,x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

#Specification matérielle
dtype = tor.float
device = tor.device("cpu")

#Specifications du modèle
K = 10e-5       #Précision du modèle

N = 64          #Nombre de données
D_in = 1000     #Dimension Entrée
H = 100         #Dimension Intermediaire
D_out = 10      #Dimension Sortie

x = tor.randn(N, D_in, device=device, dtype=dtype)    #Entrées Aléatoires
y = tor.randn(N, D_out, device=device, dtype=dtype)   #Sorties Aléatoires

#Modèle de réseau neuronal
model = LayerNet(D_in,H,D_out)
#Definition de la fonction de perte
loss_fn = tor.nn.MSELoss(size_average=False) #Mean Squared Error
#Definition de l'optimiseur
opt = tor.optim.SGD(model.parameters(),lr=1e-4)
                     
it = 0
while True:             #Iterations d'apprentissage
    #Forward pass
    y_pred = model(x)      #Calcul valeur finale predite

    loss = loss_fn(y_pred, y) #Calcul de la perte
    print "STEP",it,":",loss.item()

    #RaZ Gradients
    opt.zero_grad()
    #Backprop des gradients
    loss.backward()

    #Mise à jour des poids
    opt.step()

    #Point d'arret do while
    if loss < K:
        break
    #Incrément boucle
    it += 1

    
    
