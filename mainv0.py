import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(y):
    """
    Estime la matrice de covariance des signaux reçus.
    """
    K = y.shape[1]  # Nombre de snapshots
    R = (1 / K) * (y @ y.conj().T)
    return R

def compute_power_conventional_beamforming(R, theta, a_func):
    """
    Calcule la puissance de sortie pour le beamforming conventionnel.
    """
    P_cbf = []
    for angle in theta:
        a_theta = a_func(angle)
        w_cbf = a_theta / np.linalg.norm(a_theta)
        P_theta = np.abs(w_cbf.conj().T @ R @ w_cbf)
        P_cbf.append(P_theta)
    return np.array(P_cbf)

def compute_power_capon(R, theta, a_func):
    """
    Calcule la puissance de sortie pour le beamforming adaptatif (Capon).
    """
    P_capon = []
    R_inv = np.linalg.inv(R)
    for angle in theta:
        a_theta = a_func(angle)
        numerator = 1
        denominator = (a_theta.conj().T @ R_inv @ a_theta).real
        P_theta = numerator / denominator
        P_capon.append(P_theta)
    return np.array(P_capon)

def compute_power_music(R, theta, a_func, num_sources):
    """
    Calcule la puissance de sortie pour la méthode MUSIC.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    U_n = eigenvectors[:, :-num_sources]  # Sous-espace du bruit

    P_music = []
    for angle in theta:
        a_theta = a_func(angle)
        denominator = np.linalg.norm(U_n.conj().T @ a_theta)**2
        P_theta = 1 / denominator if denominator != 0 else 0
        P_music.append(P_theta)
    return np.array(P_music)

if __name__ == "__main__":
    # Paramètres
    N = 25  # Nombre d'antennes
    lambda_ = 200  # Longueur d'onde
    d = 50  # Distance entre antennes
    theta_range = np.linspace(-100, 100, 1000)  # Intervalle d'angles
    K = 500  # Nombre de snapshots

    # Fonction de vecteur directeur
    def steering_vector(theta):
        k = 2 * np.pi / lambda_
        n = np.arange(N)
        return np.exp(-1j * k * n * d * np.sin(np.radians(theta)))

    # Demande à l'utilisateur quelle question exécuter
    print("Choisissez une question à exécuter :")
    print("2. Comparer le beamforming conventionnel (CBF) et adaptatif (Capon) -25, 20, 25")
    print("3. Comparer le beamforming conventionnel (CBF) et adaptatif (Capon) -25, 20, 21")
    print("7. Ajouter la méthode MUSIC à la comparaison")
    choice = input("Entrez le numéro de votre choix : ")

    if choice == "2":
        source_angles = [-25, 20, 25]  # Angles des sources
        num_sources = len(source_angles)  # Nombre de sources


        # Simulation des signaux reçus
        np.random.seed(42)
        s = np.random.randn(3, K) + 1j * np.random.randn(3, K)
        A = np.array([steering_vector(angle) for angle in source_angles]).T
        noise = 0.1 * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
        y = A @ s + noise
        R = compute_covariance_matrix(y)  # Matrice de covariance
        # Calcul des puissances
        P_cbf = compute_power_conventional_beamforming(R, theta_range, steering_vector)
        P_capon = compute_power_capon(R, theta_range, steering_vector)

        # Tracé des résultats
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, 10 * np.log10(P_cbf), label="Conventional Beamforming (CBF)")
        plt.plot(theta_range, 10 * np.log10(P_capon), label="Capon's Method")
        for angle in source_angles:
            plt.axvline(x=angle, color='red', linestyle=':', linewidth=0.8)
        plt.xlabel("Angle (°)")
        plt.ylabel("Power (dB)")
        plt.title("Comparison of CBF and Capon's Method")
        plt.legend()
        plt.grid()
        plt.show()

    if choice == "3":
        source_angles = [-25, 20, 21]  # Angles des sources
        num_sources = len(source_angles)  # Nombre de sources


        # Simulation des signaux reçus
        np.random.seed(42)
        s = np.random.randn(3, K) + 1j * np.random.randn(3, K)
        A = np.array([steering_vector(angle) for angle in source_angles]).T
        noise = 0.1 * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
        y = A @ s + noise
        R = compute_covariance_matrix(y)  # Matrice de covariance
        # Calcul des puissances
        P_cbf = compute_power_conventional_beamforming(R, theta_range, steering_vector)
        P_capon = compute_power_capon(R, theta_range, steering_vector)
        # Calcul des puissances
        P_cbf = compute_power_conventional_beamforming(R, theta_range, steering_vector)
        P_capon = compute_power_capon(R, theta_range, steering_vector)

        # Tracé des résultats
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, 10 * np.log10(P_cbf), label="Conventional Beamforming (CBF)")
        plt.plot(theta_range, 10 * np.log10(P_capon), label="Capon's Method")
        for angle in source_angles:
            plt.axvline(x=angle, color='red', linestyle=':', linewidth=0.8)
        plt.xlabel("Angle (°)")
        plt.ylabel("Power (dB)")
        plt.title("Comparison of CBF and Capon's Method")
        plt.legend()
        plt.grid()
        plt.show()

    elif choice == "7":
        source_angles = [-25, 20, 21.5]  # Angles des sources
        num_sources = len(source_angles)  # Nombre de sources


        # Simulation des signaux reçus
        np.random.seed(42)
        s = np.random.randn(3, K) + 1j * np.random.randn(3, K)
        A = np.array([steering_vector(angle) for angle in source_angles]).T
        noise = 0.1 * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
        y = A @ s + noise
        R = compute_covariance_matrix(y)  # Matrice de covariance

        # Calcul des puissances
        #P_cbf = compute_power_conventional_beamforming(R, theta_range, steering_vector)
        P_capon = compute_power_capon(R, theta_range, steering_vector)
        P_music = compute_power_music(R, theta_range, steering_vector, num_sources)

        # Tracé des résultats
        plt.figure(figsize=(10, 6))
        #plt.plot(theta_range, 10 * np.log10(P_cbf), label="Conventional Beamforming (CBF)")
        plt.plot(theta_range, 10 * np.log10(P_capon), label="Capon's Method")
        plt.plot(theta_range, 10 * np.log10(P_music), label="MUSIC Method")
        for angle in source_angles:
            plt.axvline(x=angle, color='red', linestyle=':', linewidth=0.8)
        plt.xlabel("Angle (°)")
        plt.ylabel("Power (dB)")
        #plt.title("Comparison of CBF, Capon's Method, and MUSIC")
        plt.title("Comparison of Capon's method and MUSIC method")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print("Choix invalide. Veuillez exécuter à nouveau et entrer un choix valide")
