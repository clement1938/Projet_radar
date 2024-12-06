# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:02:35 2024

@author: mateo
"""

import numpy as np
import matplotlib.pyplot as plt

# Fonction d'initialisation
def simulation_parameters(nb_of_antennas=25,
                    inter_distance_btw_antennas=50,
                    wavelength=200,
                    source_direction=20,
                    signal_power=0.1,
                    interference_direction=30,
                    interference_power=0.1,
                    noise_standard_deviation=0.1,
                    nb_of_snapshots=10):
    """
    Elle définit les paramètres de simulation et renvoit sous forme de dictionnaire.
    """
    N = nb_of_antennas
    d = inter_distance_btw_antennas
    lambda_ = wavelength
    theta_s = source_direction
    Ps = signal_power
    theta_i = interference_direction
    Pi = interference_power
    sigma = noise_standard_deviation
    K = nb_of_snapshots

    # Calcul de SNR_in pour la partie I
    SNR_in = Ps / (sigma ** 2)

    # Création des signaux s(k), y_I(k), n(k) basés sur une distribution gaussienne pour la partie II
    S_k = np.ones((N, K))
    i_k = np.ones((N, K))
    n_k = np.ones((N, K), dtype=complex)

    for n in range(N):
        S_k[n, :] = np.random.normal(0, Ps, K)
        i_k[n, :] = np.random.normal(0, Pi, K)
        n_k[n, :] = (np.random.normal(0, sigma**2/2, K) + 1j * np.random.normal(0, sigma**2/2, K))


    y_i = np.dot(get_a_theta(theta_i, N, d, lambda_), i_k)
    y_i = y_i.reshape(10, 1)
    
    dictionaire = {"N": N, "d": d, "lambda_": lambda_, "theta_s": theta_s, "Ps": Ps, "theta_i": theta_i,"Pi": Pi,"sigma": sigma, "K": K,"SNR_in": SNR_in,"S_k": S_k, "i_k": i_k, "n_k": n_k,"y_i": y_i}
    
    return dictionaire

def get_a_theta(angle, N, d, lambda_):
    '''
    Pour obtenir a_theta (vecteur de mise en phase)
    '''
    a_theta = np.ones(N, dtype=np.complex_)
    k = 2 * np.pi / lambda_
    
    for n in range(2, N+1):
        a_theta[n-1] = np.exp(-1j * k * d * (n-1) * np.sin(angle * np.pi/180))

    return a_theta

def get_C(sigma, Pi, N, d, lambda_, theta_i):
    """
    A partir des paramètres, on calcule la matrice de covariance C 
    """
    Id_N = np.eye(N)
    a_I = get_a_theta(theta_i, N, d, lambda_)
    
    C = Pi * a_I @ a_I.conj().T + sigma**2 * Id_N 
    
    return C


def get_R(Ps, theta_s, sigma, Pi, N, d, lambda_):
    """
    A partir des paramètres, on calcule la matrice de covariance R 
    """
    C = get_C(sigma, Pi, N, d, lambda_, theta_s)
    a_theta =  get_a_theta(theta_s, N, d, lambda_)

    R =  C + Ps * np.outer(a_theta, (a_theta).conj().T)
    return R

def get_C_hat(K, y_i, n_k):
    """
    On calcule une estimation de la matrice de covariance C 
    """
    somme = 0
    for k in range(1, K):
        somme += (y_i[k] + n_k[:, k]) @ (y_i[k] + n_k[:, k]).conj().T
    C_hat = somme / K
    return C_hat

def get_R_hat(K, y_k):
    """
    On calcule une estimation de la matrice de covariance R 
    """
    somme = 0
    for k in range(1, K):
        #somme += (y_k[:, k]) @ (y_k[:, k]).conj().T
        somme += np.outer(y_k[:, k], y_k[:, k].conj())
    R_hat = somme / K
    return R_hat

def get_w_cbf(N, d, lambda_, angle):
    """
    Filtre pour le Beamforming conventionnel 
    """
    a_theta = get_a_theta(angle, N, d, lambda_)
    w_CBF = a_theta / (a_theta.conj().T @ a_theta)
    return w_CBF


def get_w_opt(C, N, d, lambda_, theta_s):
    '''
    Pour le beamforming optimal adaptatif

    '''
    a_theta = get_a_theta(theta_s, N, d, lambda_)
    
    w_opt = np.linalg.inv(C) @ a_theta / (a_theta.conj().T @ np.linalg.inv(C) @ a_theta)
    return w_opt


def wMVDR(C, N, d, lambda_, theta_est):
    '''
    Calcule du filre MVDR
    Dans nos cas, l'angle estimé le sera avec 2 degrés de trop
    '''
    a_theta = get_a_theta(theta_est, N, d, lambda_)
    
    w_MVDR = np.linalg.inv(C) @  a_theta / (a_theta.conj().T @ np.linalg.inv(C) @ a_theta)

    return w_MVDR


def wMPDR(R, N, d, lambda_, theta_est):
    '''
    Calcule du filre MPDR
    '''
    a_theta = get_a_theta(theta_est, N, d, lambda_)
    
    w_MPDR = np.linalg.inv(R) @ a_theta / (a_theta.conj().T @ np.linalg.inv(R) @ a_theta)
    return w_MPDR


def get_SNR_out(SNR_in, w, N, d, lambda_, angle):
    SNR_out = SNR_in * np.absolute(w.conj().T @ get_a_theta(angle, N, d, lambda_))**2 / (np.linalg.norm(w)**2)
    return SNR_out

def get_SINR(SNR_in, w, C, N, d, lambda_, angle):
    Ps = 0.1
    a_theta = get_a_theta(angle, N, d, lambda_)
    SINR = np.abs(Ps * np.absolute(w.conj().T @ a_theta)**2 / (w.conj().T @ C @ w))
    return SINR

def draw_SINR(SNR_in, w_opt, N, d, lambda_, theta_s):
    '''
    On plot le SINR
    '''
    theta_array = np.linspace(theta_s-20, theta_s+20, 120)
    SINR = np.zeros(np.shape(theta_array)[0])

    for i, angle in enumerate(theta_array):
        SINR[i] = 10 * np.log10(get_SINR(SNR_in, w_opt, get_C(0.1, 0.1, N, d, lambda_, 30), N, d, lambda_, angle))

    plt.figure()
    plt.plot(theta_array, SINR, "-b")
    plt.title("SINR en fonction de l'angle")
    plt.xlabel("Angle (°)")
    plt.ylabel("SINR (dB)")
    plt.grid(True)
    plt.show()

def draw_SINR_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi):
    '''
    On plot tous les SINR
    '''
    
    C = get_C(0.1, 0.1, N, d, lambda_, 30)
    R = get_R(Ps=0.1, theta_s=20, sigma=0.1 ,Pi = 0.1, N=N, d=d, lambda_ = lambda_)
    
    theta_array = np.linspace(theta_s-20, theta_s+20, 120)
    SINR_cbf = np.zeros(np.shape(theta_array)[0])
    SINR_opt = np.zeros(np.shape(theta_array)[0])
    SINR_mvdr = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr = np.zeros(np.shape(theta_array)[0])
    MVDR = wMVDR(C, N, d, lambda_, theta_s+2)
    MPDR = wMPDR(R, N, d, lambda_, theta_s+2)
    

    for i, angle in enumerate(theta_array):
        SINR_cbf[i] = 10 * np.log10(get_SINR(SNR_in, get_w_cbf(N, d, lambda_, theta_s), C, N, d, lambda_, angle))
        SINR_opt[i] = 10 * np.log10(get_SINR(SNR_in, get_w_opt(C, N, d, lambda_, theta_s), C, N, d, lambda_, angle))
        SINR_mvdr[i] = 10 * np.log10(get_SINR(SNR_in, MVDR, C, N, d, lambda_, angle))
        SINR_mpdr[i] = 10 * np.log10(get_SINR(SNR_in, MPDR, C, N, d, lambda_, angle))
                                                                                   
    plt.figure()
    
    # Tracer les courbes avec des labels pour la légende
    plt.plot(theta_array, SINR_cbf, "-k", label="CBF (Conventional Beamforming)")
    plt.plot(theta_array, SINR_opt, "-b", label="OPT (Optimal Beamforming)")
    plt.plot(theta_array, SINR_mvdr, "-m", label="MVDR (Minimum Variance Distortionless Response)")
    plt.plot(theta_array, SINR_mpdr, "*c", label="MPDR (Minimum Power Distortionless Response)")

    # Ajouter un titre, des labels d'axes et une légende
    plt.title("Comparaison des SINR pour différentes méthodes de Beamforming")
    plt.xlabel("Angle (°)")
    plt.ylabel("SINR (dB)")
    plt.legend()  # Ajouter la légende
    plt.grid(True)  # Ajouter une grille pour améliorer la lisibilité

    # Affichage du graphique
    plt.show()
    
    
def compute_y_k(params, S_k, y_i, n_k, k):
    """
    Calcule y_k à partir des signaux S_k, du signal d'interférence y_i et du bruit n_k.
    """
    a_theta_s = get_a_theta(params["theta_s"], params["N"], params["d"], params["lambda_"])

    # Calcul de y_k en multipliant chaque colonne de S_k par a_theta_s (cela donne un résultat de forme (25, 10))
    y_k = (np.dot(a_theta_s, S_k) + y_i + n_k[k, :])

    return y_k



def draw_SINR_mpdr_mvdr_hat(simulation, w_mpdr_hat, w_mvdr_hat):
    """
    Cette fonction trace le SINR pour les filtres MPDR et MVDR, ainsi que pour leurs versions modifiées
 
    """
    # Plage d'angles (de theta_s-20° à theta_s+20° avec un pas de 0.1°)
    theta_array = np.linspace(simulation.theta_s - 20, simulation.theta_s + 20, 120)

    # Initialisation des tableaux de SINR pour chaque configuration
    SINR_mvdr = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr = np.zeros(np.shape(theta_array)[0])
    SINR_mvdr_hat = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr_hat = np.zeros(np.shape(theta_array)[0])

    # Calcul des SINR pour chaque angle
    for i, angle in enumerate(theta_array):
        # SINR pour MVDR avec estimation de la covariance
        SINR_mvdr[i] = 10 * np.log10(simulation.get_SINR(simulation.MVDR(simulation.theta_s + 2), angle))
        
        # SINR pour MPDR avec estimation de la covariance
        SINR_mpdr[i] = 10 * np.log10(simulation.get_SINR(simulation.MPDR(simulation.theta_s + 2), angle))
        
        # SINR pour MVDR avec \( \hat{C} \)
        SINR_mvdr_hat[i] = 10 * np.log10(simulation.get_SINR(w_mvdr_hat, angle))
        
        # SINR pour MPDR avec \( \hat{R} \)
        SINR_mpdr_hat[i] = 10 * np.log10(simulation.get_SINR(w_mpdr_hat, angle))

    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    plt.plot(theta_array, SINR_mpdr_hat, "*b", label="MPDR avec \( \hat{R} \)", markersize=6)
    plt.plot(theta_array, SINR_mvdr_hat, "*r", label="MVDR avec \( \hat{C} \)", markersize=6)
    plt.plot(theta_array, SINR_mvdr, "-m", label="MVDR sans estimation")
    plt.plot(theta_array, SINR_mpdr, "-c", label="MPDR sans estimation")

    # Ajouter le titre, les labels et la légende
    plt.title("Comparaison des SINR pour MPDR et MVDR avec et sans estimation")
    plt.xlabel("Angle (θ en degrés)")
    plt.ylabel("SINR (dB)")
    plt.legend()  # Affiche la légende
    plt.grid(True)  # Affiche la grille pour meilleure lisibilité

    # Affichage du graphique
    plt.show()




