#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as tor

'''
Le modèle apprend à reconnaitre à partir de données en entrée et en sortie
Le modèle apprend jusqu'à une précision K
'''

#Specification matérielle
dtype = tor.float
device = tor.device("cpu")

#Specifications du modèle
K = 10e-2       #Précision du modèle

N = 64          #Nombre de données
D_in = 1000     #Dimension Entrée
H = 100         #Dimension Intermediaire
D_out = 10      #Dimension Sortie

learn_rate = 1e-6

x = tor.randn(N, D_in, device=device, dtype=dtype)    #Entrées Aléatoires
y = tor.randn(N, D_out, device=device, dtype=dtype)   #Sorties Aléatoires

#Modèle de réseau neuronal
model = tor.nn.Sequential(
    tor.nn.Linear(D_in,H),
    tor.nn.ReLU(),
    tor.nn.Linear(H,D_out)
)
#Definition de la fonction de perte
loss_fn = tor.nn.MSELoss(size_average=False) #Mean Squared Error
#Definition de l'optimiseur
opt = tor.optim.Adam(model.parameters(),lr=learn_rate)
                     
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

    
    
