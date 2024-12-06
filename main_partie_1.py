# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:41:22 2024

@author: mateo
"""
import librairie_partie_1_test as lib
import numpy as np

#%% QUESTION 2 : LES PARAMETRES (Print)

params = lib.simulation_parameters()

print("Paramètres de simulation :")
for key, value in params.items():
    print(f"{key}: {value}")

N = params["N"]
d = params["d"]
lambda_ = params["lambda_"]
theta_s = params["theta_s"]
theta_i = params["theta_i"]
Ps = params["Ps"]
Pi = params["Pi"]
sigma = params["sigma"]
K = params["K"]
SNR_in = params["SNR_in"]
y_i = params["y_i"]
S_k = params["S_k"]
n_k = params["n_k"]

#%% QUESTION 3 VERIFICATION  SNR THEORIQUE ET NUMERIQUE

# Calculer wCBF avec la fonction get_w_cbf
w_cbf = lib.get_w_cbf(N, d, lambda_, theta_s)

# Vérification du SNR après filtrage avec wCBF
SNR_out_cbf = lib.get_SNR_out(SNR_in, w_cbf, N, d, lambda_, theta_s)

print(f"SNR_in multiplié par N : {SNR_in*N}")
print(f"SNR après filtrage avec wCBF : {SNR_out_cbf}")

#%% QUESTION 4 PLOT SINR

# Calculer la matrice de covariance C pour l'interférence
C = lib.get_C(sigma, Pi, N, d, lambda_, theta_i)

# Calculer le poids optimal adaptatif w_opt
w_opt = lib.get_w_opt(C, N, d, lambda_, theta_s)

# Calculer et tracer le SINR avec w_opt
lib.draw_SINR(SNR_in, w_opt, N, d, lambda_, theta_s)  

#%% QUESTION 7

# Erreur d'angle estimée (2 degrés)
angle_estime = theta_s + 2  # L'angle estimé est de 2° plus l'angle de la source

# Calcul de la matrice de covariance C (pour MVDR)
C = lib.get_C(sigma, Pi, N, d, lambda_, theta_i)

# Calcul de la matrice de covariance R (pour MPDR)
R = lib.get_R(Ps, theta_s, sigma, Pi, N, d, lambda_)

# Calcul des poids MPDR et MVDR
w_mvdr = lib.wMVDR(C, N, d, lambda_, angle_estime)
w_mpdr = lib.wMPDR(R, N, d, lambda_, angle_estime)


# Calcul et affichage du SINR pour MPDR et MVDR
SINR_mvdr = lib.get_SINR(SNR_in, w_mvdr, C, N, d, lambda_, theta_s)  # SINR pour MVDR
SINR_mpdr = lib.get_SINR(SNR_in, w_mpdr, C, N, d, lambda_, theta_s)  # SINR pour MPDR

print(f"SINR pour MVDR (avec erreur d'angle de 2°) : {10*np.log10(SINR_mvdr)} dB")
print(f"SINR pour MPDR (avec erreur d'angle de 2°) : {10*np.log10(SINR_mpdr)} dB")

lib.draw_SINR_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi)

#%% Question 10

S_k = params["S_k"]  # Signal source
y_i = params["y_i"]  # Signal d'interférence
n_k = params["n_k"]  # Bruit
k=1
y_k = lib.compute_y_k(params, S_k, y_i, n_k, k)

print("y_k :", y_k)

#%% Question 11

k = 0  # Choisir un indice valide
y_k = lib.compute_y_k(params, params["S_k"], params["y_i"], params["n_k"], k)

# Vérification de la taille de y_k
print(f"Dimensions de y_k: {y_k.shape}")  # Cela devrait être (N, K)

# Appel à get_R_hat
R_hat = lib.get_R_hat(params["K"], y_k)
print(f"Dimensions de R_hat: {R_hat.shape}")



