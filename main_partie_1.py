# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:41:22 2024

@author: mateo
"""
import librairie_partie_1 as lib
import numpy as np

#%% QUESTION 2 : LES PARAMETRES (Print)

params = lib.simulation_parameters()

print("Paramètres de simulation :")
for key, value in params.items():
    print(f"{key}: {value}")
    

#%% QUESTION 3 VERIFICATION  SNR THEORIQUE ET NUMERIQUE

wCBF, a_theta_s = lib.compute_wCBF(params) 

SNR_theorique, SNR_numerique = lib.verify_SNR(params, wCBF, a_theta_s)

print(f"SNR théorique : {SNR_theorique:f}")
print(f"SNR numérique : {SNR_numerique:f}")

#%% QUESTION 4 PLOT SINR

C = lib.compute_covariance_matrix(params)

# Calculer le filtre wopt
wopt = lib.compute_wopt(params, C)

# Définir la plage d'angles pour θ (en radians)
theta_s = np.deg2rad(params["theta_s"])
theta_range = np.arange(theta_s - np.deg2rad(20), theta_s + np.deg2rad(20), np.deg2rad(0.1))

# Calculer le SINR pour chaque θ
SINR_log = lib.compute_SINR(params, wopt, C, theta_range)

# Tracer le SINR
lib.plot_SINR(theta_range, SINR_log)