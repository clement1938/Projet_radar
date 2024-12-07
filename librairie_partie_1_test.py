# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:02:35 2024

@author: mateo
"""

import numpy as np
import matplotlib.pyplot as plt

def simulation_parameters(nb_of_antennas=25,
                    inter_distance_btw_antennas=50,
                    wavelength=200,
                    source_direction=20,
                    signal_power=0.1,
                    interference_direction=30,
                    interference_power=0.1,
                    noise_standard_deviation=0.1,
                    nb_of_snapshots=10) :
    """
    Dans cette première fonction, on définit tous les paramètres de simulation et on les renvois sous forme de dictionnaire.
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

    SNR_in = Ps / (sigma ** 2)  # On calcule de SNR_in (partie a)

    # On crée les signaux s(k), y_I(k), n(k) qui sont basés sur une distribution gaussienne pour la partie b
    S_k = np.ones((N, K))
    i_k = np.ones((N, K))
    n_k = np.ones((N, K), dtype=complex)

    for n in range(N):
        S_k[n, :] = np.random.normal(0, Ps, K)   # On les regroupe dans cette même boucle car ils ont la même taille
        i_k[n, :] = np.random.normal(0, Pi, K)
        n_k[n, :] = (np.random.normal(0, sigma**2/2, K) + 1j * np.random.normal(0, sigma**2/2, K))

    a_theta = (get_a_theta(theta_i, N, d, lambda_))

    y_i = y_i = a_theta[:, np.newaxis] * i_k  # Le calcule de y_i nous sert également pour la partie b
    
    
    # On décide de fonctionner avec unn dictionnaire par pur simplicité, on pourra appeler les éléments de la simulation directement
    # Après réflexion, on aurait pu construire une classe ce qui aurait facilité les choses mais bon on a pas pu tout faire
    dictionaire = {"N": N, "d": d, "lambda_": lambda_, "theta_s": theta_s, "Ps": Ps, "theta_i": theta_i,"Pi": Pi,"sigma": sigma, "K": K,"SNR_in": SNR_in,"S_k": S_k, "i_k": i_k, "n_k": n_k,"y_i": y_i}
    
    return dictionaire

def get_a_theta(angle, N, d, lambda_):
    """
    Une des fonctions "fondatrices", qui nous
    permet d'obtenir a_theta (le vecteur de mise en phase)
    """
    a_theta = np.ones(N, dtype = np.complex_)
    k = 2 * np.pi / lambda_  # On définit k, plus simple pour la formule suivante
    
    for n in range(2, N+1):  # Création d'a_theta dans la boucle pour N éléments
        a_theta[n-1] = np.exp(-1j * k * d * (n-1) * np.sin(angle * np.pi/180))

    return a_theta

def get_C(sigma, Pi, N, d, lambda_, theta_i):
    """
    A partir des paramètres, on calcule la matrice de covariance C 
    """
    Id_N = np.eye(N)  
    a_I = get_a_theta(theta_i, N, d, lambda_)
    
    C = Pi * a_I @ a_I.conj().T + sigma**2 * Id_N   # On applique la formule qu'on a démontrée de la manière la plus évidente
    
    return C


def get_R(Ps, theta_s, sigma, Pi, N, d, lambda_):
    """
    De la même manière, on calcule la matrice de covariance R
    Nous avons eu pas mal de problème avec celle là, même si l'expression n'est pas si compliquée
    """
    C = get_C(sigma, Pi, N, d, lambda_, theta_s)  # On récupère C et a_theta
    a_theta =  get_a_theta(theta_s, N, d, lambda_)

    R =  C + Ps * np.outer(a_theta, (a_theta).conj().T)  # Et on calcule R avec la simple formule démontrée
    return R

def get_C_hat(K, y_i, n_k):
    """
    Pour la partie b :
    On calcule une estimation de la matrice de covariance C (formule du TP)
    """
    somme = 0
    for k in range(1, K):
        somme += np.outer((y_i[:, k] + n_k[:, k]), (y_i[:, k] + n_k[:, k]).conj())
    C_hat = somme / K
    return C_hat

def get_R_hat(K, y_k):
    """
    On calcule une estimation de la matrice de covariance R, toujours avec la formule du TP
    """
    somme = 0
    for k in range(1, K):
        #somme += (y_k[:, k]) @ (y_k[:, k]).conj().T
        somme += np.outer(y_k[:, k], y_k[:, k].conj())
    R_hat = somme / K
    return R_hat

def get_w_cbf(N, d, lambda_, angle):
    """
    C'est le filtre pour le beamforming conventionnel 
    """
    a_theta = get_a_theta(angle, N, d, lambda_)
    w_CBF = a_theta / (a_theta.conj().T @ a_theta) # Application de son expression
    return w_CBF


def get_w_opt(C, N, d, lambda_, theta_s):
    """
    On implémente aussi le beamforming optimal adaptatif 
    en se basant toujours sur les équations du cours
    """
    a_theta = get_a_theta(theta_s, N, d, lambda_)
    w_opt = np.linalg.inv(C) @ a_theta / (a_theta.conj().T @ np.linalg.inv(C) @ a_theta)
    return w_opt


def wMVDR(C, N, d, lambda_, theta_est):
    """
    Calcule du filre MVDR
    Dans nos cas, l'angle estimé le sera avec 2 degrés de trop 
    """
    a_theta = get_a_theta(theta_est, N, d, lambda_)
    
    w_MVDR = np.linalg.inv(C) @  a_theta / (a_theta.conj().T @ np.linalg.inv(C) @ a_theta)

    return w_MVDR


def wMPDR(R, N, d, lambda_, theta_est):
    """
    Calcul du filre MPDR avec lequel on a eu pas mal de problèmes
    Cette fonctions est EXACTEMENT la même que la précédante pour le MVDR
    Mais on a préféré la laisser pour éviter les confusions lors des appels de fonctions 
    """
    a_theta = get_a_theta(theta_est, N, d, lambda_)
    
    w_MPDR = np.linalg.inv(R) @ a_theta / (a_theta.conj().T @ np.linalg.inv(R) @ a_theta)
    return w_MPDR


def get_SNR_out(SNR_in, w, N, d, lambda_, angle):
    """
    On calcule ici SNR_out pour la question 3 de la partie a
    """
    SNR_out = SNR_in * np.absolute(w.conj().T @ get_a_theta(angle, N, d, lambda_))**2 / (np.linalg.norm(w)**2)
    return SNR_out

def get_SINR(SNR_in, w, C, N, d, lambda_, angle):
    """
    C'est sûrement l'une des fonctions les plus importantes, on calcule le SINR
    La formule retse la même pour tous les filtres
    """
    Ps = 0.1 # On a redéfini Ps même si on aurai pu le mettre en argument, c'était pour du débogage à la base
    a_theta = get_a_theta(angle, N, d, lambda_)
    SINR = np.abs(Ps * np.absolute(w.conj().T @ a_theta)**2 / (w.conj().T @ C @ w)) # La formule du SINR
    return SINR

def draw_SINR(SNR_in, w_opt, N, d, lambda_, theta_s):  
    """
    On plot le SINR pour un seul filtre, en l'occurence le w_opt pour notre exemple'
    """
    theta_array = np.linspace(theta_s-20, theta_s+20, 120) # On prend un interval de 40 degrés centré sur 20
    SINR = np.zeros(np.shape(theta_array)[0])

    for i, angle in enumerate(theta_array): # On calcule les SINR pour chaque angle (en dB !)
        SINR[i] = 10 * np.log10(get_SINR(SNR_in, w_opt, get_C(0.1, 0.1, N, d, lambda_, 30), N, d, lambda_, angle))
        
    # Et puis on plot simplement le SINR
    plt.figure()
    plt.plot(theta_array, SINR, "-b")
    plt.title("SINR en fonction de l'angle")
    plt.xlabel("Angle (°)")
    plt.ylabel("SINR (dB)")
    plt.grid(True)
    plt.show()

def draw_SINR_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi):
    """
    Cette fonction nous permet de tracer tous les SINR sur un même graph
    """
    C = get_C(0.1, 0.1, N, d, lambda_, 30) # On récupère C pour tous les calculs suivants (on a mis certains paramètres en fixe pour déboger)
    R = get_R(Ps=0.1, theta_s=20, sigma=0.1 ,Pi = 0.1, N=N, d=d, lambda_ = lambda_) # Pareil pour R
    
    theta_array = np.linspace(theta_s-20, theta_s+20, 120) # On crée notre tableau de valeur pour théta
    SINR_cbf = np.zeros(np.shape(theta_array)[0])          # On initialise tous les SINR vides
    SINR_opt = np.zeros(np.shape(theta_array)[0])
    SINR_mvdr = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr = np.zeros(np.shape(theta_array)[0])
    
    MVDR = wMVDR(C, N, d, lambda_, theta_s+2)              # On récupère les filtres (avec une erreur de 2 degrés pour ces deux là)
    MPDR = wMPDR(R, N, d, lambda_, theta_s+2)
    wOPT = get_w_opt(C, N, d, lambda_, theta_s)
    wCBF = get_w_cbf(N, d, lambda_, theta_s)
    
    for i, angle in enumerate(theta_array): # et on calcule les SINR pour chaque filtre ! (en dB !)
        SINR_cbf[i] = 10 * np.log10(get_SINR(SNR_in, wCBF, C, N, d, lambda_, angle))
        SINR_opt[i] = 10 * np.log10(get_SINR(SNR_in, wOPT, C, N, d, lambda_, angle))
        SINR_mvdr[i] = 10 * np.log10(get_SINR(SNR_in, MVDR, C, N, d, lambda_, angle))
        SINR_mpdr[i] = 10 * np.log10(get_SINR(SNR_in, MPDR, C, N, d, lambda_, angle))
                                                                                   
    plt.figure()
    
    # On peut tracer les courbes avec les labels et toutes les fioritures immaginables
    plt.plot(theta_array, SINR_cbf, "-k", label="CBF")
    plt.plot(theta_array, SINR_opt, "-b", label="OPT")
    plt.plot(theta_array, SINR_mvdr, "-m", label="MVDR")
    plt.plot(theta_array, SINR_mpdr, "*c", label="MPDR")

    plt.title("Comparaison des SINR pour différentes méthodes de Beamforming")
    plt.xlabel("Angle (°)")
    plt.ylabel("SINR (dB)")
    plt.legend()  
    plt.grid(True)

    plt.show()
    
    
def compute_y_k(params, S_k, y_i, n_k, k):
    """
    On arrive sur la partie b avec cette nouvelle fonction
    On y calcule y_k à partir des signaux S_k, du signal d'interférence y_i et du bruit n_k.
    """
    a_theta_s = get_a_theta(params["theta_s"], params["N"], params["d"], params["lambda_"])

    # Calcul de y_k en multipliant chaque colonne de S_k par a_theta_s (cela donne un résultat de forme (25, 10))
    y_k = (np.dot(a_theta_s, S_k) + y_i + n_k[k, :])

    return y_k


#################
def draw_SINR_hat_all(SNR_in, N, d, lambda_, theta_s, sigma, Pi, C_hat, R_hat):
    """
    Cette fonction nous permet de tracer tous les SINR sur un même graph
    """
    C = get_C(0.1, 0.1, N, d, lambda_, 30) # On récupère C pour tous les calculs suivants (on a mis certains paramètres en fixe pour déboger)
    R = get_R(Ps=0.1, theta_s=20, sigma=0.1 ,Pi = 0.1, N=N, d=d, lambda_ = lambda_) # Pareil pour R

    
    theta_array = np.linspace(theta_s-20, theta_s+20, 120) # On crée notre tableau de valeur pour théta

    SINR_mvdr = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr = np.zeros(np.shape(theta_array)[0])    
    SINR_mvdr_hat = np.zeros(np.shape(theta_array)[0])
    SINR_mpdr_hat = np.zeros(np.shape(theta_array)[0])
    
    MVDR = wMVDR(C, N, d, lambda_, theta_s+2)              # On récupère les filtres (avec une erreur de 2 degrés pour ces deux là)
    MPDR = wMPDR(R, N, d, lambda_, theta_s+2)
    MVDR_hat = wMVDR(C_hat, N, d, lambda_, theta_s+2)   
    MPDR_hat = wMPDR(R_hat, N, d, lambda_, theta_s+2)   
    
    for i, angle in enumerate(theta_array): # et on calcule les SINR pour chaque filtre ! (en dB !)
        SINR_mvdr[i] = 10 * np.log10(get_SINR(SNR_in, MVDR, C, N, d, lambda_, angle))
        SINR_mpdr[i] = 10 * np.log10(get_SINR(SNR_in, MPDR, C, N, d, lambda_, angle))
        SINR_mvdr_hat[i] = 10 * np.log10(get_SINR(SNR_in, MVDR_hat, C, N, d, lambda_, angle))
        SINR_mpdr_hat[i] = 10 * np.log10(get_SINR(SNR_in, MPDR_hat, C, N, d, lambda_, angle))
                                                                                   
    plt.figure()
    
    # On peut tracer les courbes avec les labels et toutes les fioritures immaginables
    plt.plot(theta_array, SINR_mvdr, "-m", label="MVDR")
    plt.plot(theta_array, SINR_mpdr, "*c", label="MPDR")
    plt.plot(theta_array, SINR_mvdr_hat, "-b", label="MVDR_hat")
    plt.plot(theta_array, SINR_mpdr_hat, "-k", label="MPDR_hat")

    plt.title("Comparaison des SINR (avec et sans hat)")
    plt.xlabel("Angle (°)")
    plt.ylabel("SINR (dB)")
    plt.legend()  
    plt.grid(True)

    plt.show()


