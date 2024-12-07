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
i_k = params["i_k"]

#%% QUESTION 3 VERIFICATION  SNR THEORIQUE ET NUMERIQUE

# On calcule le wCBF
w_cbf = lib.get_w_cbf(N, d, lambda_, theta_s)

# Et on vérifie le SNR après filtrage avec wCBF
SNR_out_cbf = lib.get_SNR_out(SNR_in, w_cbf, N, d, lambda_, theta_s)

print(f"SNR_in multiplié par N : {SNR_in*N}")
print(f"SNR après filtrage avec wCBF : {SNR_out_cbf}")

#%% QUESTION 4 PLOT SINR

C = lib.get_C(sigma, Pi, N, d, lambda_, theta_i) # On calcule la matrice de covariance

w_opt = lib.get_w_opt(C, N, d, lambda_, theta_s) # Et puis le poids optimal w_opt

# On calcule et on tracer le SINR avec w_opt grace à notre fonction !
lib.draw_SINR(SNR_in, w_opt, N, d, lambda_, theta_s)  

#%% QUESTION 7

angle_estime = theta_s + 2  # L'angle estimé est de 2° plus l'angle de la source

C = lib.get_C(sigma, Pi, N, d, lambda_, theta_i)      # Calcul de la matrice de C (pour MVDR)
R = lib.get_R(Ps, theta_s, sigma, Pi, N, d, lambda_)  # Calcul de la matrice de R (pour MPDR)

# On calcule les poids MPDR et MVDR
w_mvdr = lib.wMVDR(C, N, d, lambda_, angle_estime)
w_mpdr = lib.wMPDR(R, N, d, lambda_, angle_estime)

SINR_mvdr = lib.get_SINR(SNR_in, w_mvdr, C, N, d, lambda_, theta_s)  # SINR pour MVDR
SINR_mpdr = lib.get_SINR(SNR_in, w_mpdr, C, N, d, lambda_, theta_s)  # SINR pour MPDR

print(f"SINR pour MVDR (avec erreur d'angle de 2°) : {10*np.log10(SINR_mvdr)} dB")
print(f"SINR pour MPDR (avec erreur d'angle de 2°) : {10*np.log10(SINR_mpdr)} dB")

lib.draw_SINR_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi) #  On trace tout grâce à notre super fonction de tracage

#%% Question 10

S_k = params["S_k"]  # Signal source
y_i = params["y_i"]  # Signal d'interférence
n_k = params["n_k"]  # Bruit
k=1
y_k = lib.compute_y_k(params, S_k, y_i, n_k, k)  # On calcule le signal y_k

print("y_k :", y_k) # Print pour valider

#%% Question 11

angle_estime = theta_s + 2  # L'angle estimé est de 2° plus l'angle de la source
k = 0  
y_k = lib.compute_y_k(params, params["S_k"], params["y_i"], params["n_k"], k)

# On prend R_hat et C_hat pour nos prochains calculs et pour les vérifier (sur spyder on peut voir nos variables)
R_hat = lib.get_R_hat(K, y_k)
C_hat = lib.get_C_hat(K, y_i, n_k)
C = lib.get_C(sigma, Pi, N, d, lambda_, theta_i)

# On calcule les poids MPDR_hat et MVDR_hat
w_mvdr_hat = lib.wMVDR(C_hat, N, d, lambda_, angle_estime)
w_mpdr_hat = lib.wMPDR(R_hat, N, d, lambda_, angle_estime)

# Les SINR pour MPDR_hat et MVDR_hat
SINR_mvdr_hat = lib.get_SINR(SNR_in, w_mvdr_hat, C_hat, N, d, lambda_, theta_s)  # SINR pour MVDR
SINR_mpdr_hat = lib.get_SINR(SNR_in, w_mpdr_hat, C_hat, N, d, lambda_, theta_s)  # SINR pour MPDR

print(f"SINR pour MVDR_hat (avec erreur d'angle de 2°) : {10*np.log10(SINR_mvdr_hat)} dB")
print(f"SINR pour MPDR_hat (avec erreur d'angle de 2°) : {10*np.log10(SINR_mpdr_hat)} dB")

lib.draw_SINR_hat_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi, C_hat, R_hat)