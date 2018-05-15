#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

'''
Le modèle apprend à reconnaitre à partir de données en entrée et en sortie
Le modèle apprend jusqu'à une précision K
'''
K = 10e-5       #Précision du modèle

N = 64          #Nombre de données
D_in = 1000     #Dimension Entrée
H = 100         #Dimension Intermediaire
D_out = 10      #Dimension Sortie

x = np.random.randn(N, D_in)    #Entrées Aléatoires
y = np.random.randn(N, D_out)   #Sorties Aléatoires

w1 = np.random.randn(D_in, H)   #Poids d'entrée vers inter
w2 = np.random.randn(H, D_out)  #Poids d'inter vers sortie

learn_rate = 1e-6

it = 0
while True:             #Iterations d'apprentissage
    h = x.dot(w1)               #Calcul de valeur inter
    h_relu = np.maximum(h, 0)   #Apllication relu
    y_pred = h_relu.dot(w2)     #Calcul valeur finale predite

    loss = np.square(y_pred - y).sum() #Calcul de la perte
    print "STEP",it,":",loss

    #Backprop des gradients de w1 et w2
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    #Mise à jour des poids
    w1 -= learn_rate * grad_w1
    w2 -= learn_rate * grad_w2

    #Point d'arret do while
    if loss < K:
        break

    #Incrément boucle
    it += 1

    
    
