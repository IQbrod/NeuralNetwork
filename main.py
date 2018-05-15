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
K = 10e-5       #Précision du modèle

N = 64          #Nombre de données
D_in = 1000     #Dimension Entrée
H = 100         #Dimension Intermediaire
D_out = 10      #Dimension Sortie

x = tor.randn(N, D_in, device=device, dtype=dtype)    #Entrées Aléatoires
y = tor.randn(N, D_out, device=device, dtype=dtype)   #Sorties Aléatoires

w1 = tor.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)   #Poids d'entrée vers inter
w2 = tor.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)  #Poids d'inter vers sortie

learn_rate = 1e-6

it = 0
while True:             #Iterations d'apprentissage
    #Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)      #Calcul valeur finale predite

    loss = (y_pred - y).pow(2).sum() #Calcul de la perte
    print "STEP",it,":",loss.item()

    #Backprop des gradients
    loss.backward()

    #Mise à jour des poids
    with tor.no_grad():
        w1 -= learn_rate * w1.grad
        w2 -= learn_rate * w2.grad
        #RaZ des gradients
        w1.grad.zero_()
        w2.grad.zero_()

    #Point d'arret do while
    if loss < K:
        break

    #Incrément boucle
    it += 1

    
    
