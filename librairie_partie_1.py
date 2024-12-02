# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:47:17 2024

@author: mateo
"""

### CE CODE CONTIENT TOUTES LES FONCTIONS DE LA PARTIE 1 #######♫##############

import numpy as np
import matplotlib.pyplot as plt

def simulation_parameters():
    """
    Elle définit les paramètres de simulation et renvoit sous forme de dictionnaire.
    """
    params = {
        "N": 25,                # Le nombre d'antennes N
        "d": 50,                # La distance entre les antennes (en m)
        "lambda": 200,          # Longueur d'onde (en m)
        "theta_s": 20,          # Direction de la source (en deg)
        "Ps": 0.1,              # Puissance du signal source
        "theta_I": 30,          # Direction de l'interférence (en deg)
        "PI": 0.1,              # Puissance de l'interférence
        "sigma": 0.1            # Écart-type du bruit
    }
    return params

# Pour la question 3 ##########################################################

def compute_wCBF(params):
    """
    On commence par calculer le filtre de beamforming wCBF.

    Arguments de cette fonction : params : Dictionnaire définit juste au dessus
    Sortie : wCBF (numpy.ndarray): Le vecteur de pondération wCBF.
    """
    N = params["N"]                          
    theta_s = np.deg2rad(params["theta_s"])  # On prend la direction de la source en radians (mieux pour calculer)
    d = params["d"]
    Lambda = params["lambda"] 

    # On commence par calculer le vecteur directeur a(theta_s)
    k = 2 * np.pi / Lambda 
    a_theta_s = np.exp(1j * k * d * np.arange(N) * np.sin(theta_s))
    
    # Et puis calcul de wCBF (normalisation)
    wCBF = a_theta_s / np.linalg.norm(a_theta_s)
    return wCBF, a_theta_s

def verify_SNR(params, wCBF, a_theta_s):
    """
    On va vérifier numériquement que le SNR après filtrage correspond à notre formule théorique.

    Argumentss:
        params : Dictionnaire qu'on vient de définir
        wCBF (array): Filtre de beamforming conventionnel !

    Sortie :
        les deux SNR (théorique et calculé)
    """
    
    Ps = params["Ps"]
    sigma = params["sigma"]
    N = params["N"]
    
    SNR_in = Ps / sigma**2     # Calcul de SNR_in
    
    SNR_out_theorique = N * SNR_in     # Calcul théorique de SNR_out
    
    noise_power = sigma**2 * np.linalg.norm(wCBF)**2

    gain_directionnel = np.abs(np.dot(np.conj(wCBF), a_theta_s))**2
    signal_power = Ps * gain_directionnel             # On multiplie par le gain directionnel

    SNR_out_numerique = signal_power / noise_power    # Correction (pour le numérique)

    return SNR_out_theorique, SNR_out_numerique

# Pour la question 4 ##########################################################

def compute_wopt(params, C):
    """
    On va calculer le filtre optimal wopt.

    Arguments :
        params : Toujours le même dict
        C (array) : Matrice de covariance (interférences + bruit)

    Sortie : wopt (array) : c'est notre filtre de beamforming optimal adaptatif
    """
    N = params["N"]
    d = params["d"]
    Lambda = params["lambda"]
    theta_s = np.deg2rad(params["theta_s"])  # On le convertit en radians, c'est plus facile

    # On calcule d'abord a(theta_s)
    k = 2 * np.pi / Lambda  
    a_theta_s = np.exp(1j * k * d * np.arange(N) * np.sin(theta_s))

    # Et puis wopt
    C_inv = np.linalg.inv(C)  # On inverse la matrice C
    num = np.dot(C_inv, a_theta_s)
    den = np.dot(np.conj(a_theta_s).T, num)
    wopt = num / den

    return wopt

def compute_covariance_matrix(params):
    """
    A partir des paramètres, on calcule la matrice de covariance C 
.
    """
    N = params["N"]
    theta_I = np.deg2rad(params["theta_I"])
    d = params["d"]
    Lambda = params["lambda"]
    sigma = params["sigma"]
    PI = params["PI"]

    # On calcule le vecteur directeur a(theta_I)
    k = 2 * np.pi / Lambda 
    a_theta_I = np.exp(1j * k * d * np.arange(N) * np.sin(theta_I))

    # Et on fait la matrice C
    C = sigma**2 * np.eye(N) + PI * np.outer(a_theta_I, np.conj(a_theta_I).T)
    return C

def compute_SINR(params, w, C, theta_range):
    """
    On calcule dans cette fonction le SINR après filtrage pour un filtre donné

    Arguments :
        params 
        w (array) (par exemple wopt)
        C : la même matrice que pour wopt
        theta_range (array) : Plage d'angles (en radians)

    Sortie : SINR_log (array) : SINR en échelle logarithmique pour chaque angle
    """
    Ps = params["Ps"]
    N = params["N"]
    d = params["d"]
    Lambda = params["lambda"]

    SINR = []

    for theta in theta_range:
        # On commence par calculer le vecteur directeur a(theta)
        k = 2 * np.pi / Lambda
        a_theta = np.exp(1j * k * d * np.arange(N) * np.sin(theta))

        # Et donc pour chaque angle, on calcule le SINR
        numerator = Ps * np.abs(np.dot(np.conj(w).T, a_theta))**2
        denominator = np.dot(np.conj(w).T, np.dot(C, w))
        SINR.append(numerator / denominator)

    # On doit convertir en échelle logarithmique (dB) pour le traçage
    SINR_log = 10 * np.log10(SINR)
    
    return SINR_log

def plot_SINR(theta_range, SINR_log):
    """
    Ici, on trace simplement le SINR en fonction de theta en échelle logarithmique !
    Le faire dans une fonction séparé nous parait quand même plus clair

    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(theta_range), SINR_log, label="SINR (dB)", color="blue")
    plt.xlabel("Angle (θ en degrés)")
    plt.ylabel("SINR (dB)")
    plt.title("SINR en fonction de θ")
    plt.grid(True)
    plt.legend()
    plt.show()