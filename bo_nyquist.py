import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import pandas as pd


# --- Supuestos iniciales ---
# theta contiene: [f1, tau1, c1, rho1, ..., fN, tauN, cN, rhoN, rho_0]
# n_inclusiones: número de inclusiones
# omega: array de frecuencias
# Salida: Re(sigma_e), Im(sigma_e) y gráfico Nyquist

# --- Cálculo de la conductividad ---
def compute_sigma_e(theta, omega):
    params_por_inclusion = 4
    # --- Descomposición del vector theta ---
    n_inclusiones = (len(theta) - 1) // params_por_inclusion
    inclusiones = np.reshape(theta[:-1], (n_inclusiones, params_por_inclusion))
    rho_0 = theta[-1]
    sigma_0 = 1 / rho_0
    jomega = 1j * omega
    sigma_e = np.full_like(omega, sigma_0, dtype=complex)
    for f_l, tau_l, c_l, rho_l in inclusiones:
        M_l = 3 * (rho_0 - rho_l) / (2 * rho_l + rho_0)
        denom = 1 + (jomega * tau_l) ** c_l
        sigma_e += sigma_0 * f_l * M_l * (1 - 1/denom)
    """
    plt.figure(figsize=(6, 6))
    plt.plot(np.real(sigma_e), -np.imag(sigma_e), label=f"{n_inclusiones} inclusión(es)")
    plt.xlabel("Re(σ)")
    plt.ylabel("Im(σ)")
    plt.title("Diagrama de Nyquist generado")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return sigma_e

# --- Definición de la función objetivo L(theta) ---
def L_theta(theta, omega, sigma_ref):
    # Modelado
    sigma_model = compute_sigma_e(theta, omega)
    # Distancia euclídea acumulada (suma de diferencias al cuadrado)
    return np.sum(np.abs(sigma_model - sigma_ref)**2)

# --- Preprocesamiento de theta ---
def preprocess_theta(theta):
    """
    Escala theta según:
    - indetheta mod4 in {0,2} (f_l, c_l): estandarización.
    - indetheta mod4 in {1,3} y última columna (tau_l, rho_l, rho_0): log10 + estandarización.
    Retorna theta_scaled, scaler_info para inversión.
    """
    theta = np.array(theta, dtype=float)
    n_features = theta.shape[1]
    scaler_info = []
    theta_scaled = np.zeros_like(theta)
    for j in range(n_features):
        if j % 4 in [0, 2]:  # f_l o c_l
            mu = theta[:, j].mean()
            sigma = theta[:, j].std(ddof=0)
            theta_scaled[:, j] = (theta[:, j] - mu) / sigma
            scaler_info.append(('standard', mu, sigma))
        else:
            logtheta = np.log10(theta[:, j])
            mu = logtheta.mean()
            sigma = logtheta.std(ddof=0)
            theta_scaled[:, j] = (logtheta - mu) / sigma
            scaler_info.append(('log_standard', mu, sigma))
    return theta_scaled, scaler_info


def inverse_preprocess_theta(theta_scaled, scaler_info):
    """
    Invierte la transformación de preprocess_theta.
    """
    theta = np.zeros_like(theta_scaled)
    for j, (scale_type, mu, sigma) in enumerate(scaler_info):
        if scale_type == 'standard':
            theta[:, j] = theta_scaled[:, j] * sigma + mu
        else:
            logtheta = theta_scaled[:, j] * sigma + mu
            theta[:, j] = 10 ** logtheta
    return theta


def preprocess_L(L, eps=1e-8):
    """
    Aplica log(L + eps) y estandarización.
    """
    L = np.array(L, dtype=float)
    logL = np.log(L + eps)
    mu = logL.mean()
    sigma = logL.std(ddof=0)
    L_scaled = (logL - mu) / sigma
    return L_scaled, (mu, sigma)


def inverse_preprocess_L(L_scaled, mu_sigma):
    """
    Invierte la transformación de preprocess_L.
    """
    mu, sigma = mu_sigma
    logL = L_scaled * sigma + mu
    return np.exp(logL)

# --- Generacion de 10 datos como Warmup para inicializar el GP ---
def generate_warmup(n_inclusiones, omega, sigma_ref, n_warmup=10, seed=42):

    # --- Rangos optimos para muestreo aleatorio ---
    ranges = {
        'f_l':    (0.01, 0.3),
        'tau_l':  (1e-6, 1e-2),
        'c_l':    (0.5, 1.0),
        'rho_l':(1.0, 100.0),
        'rho0': (0.1, 10.0)
    }
    # --- Generación de theta aleatorio dentro de los rangos---
    np.random.seed(seed)
    thetas_warmup = []
    L_values = []
    for _ in range(n_warmup):
        theta_rand = []
        for _ in range(n_inclusiones):
            theta_rand.extend([
                np.random.uniform(*ranges['f_l']),
                np.random.uniform(*ranges['tau_l']),
                np.random.uniform(*ranges['c_l']),
                np.random.uniform(*ranges['rho_l'])
            ])
        theta_rand.append(np.random.uniform(*ranges['rho0']))
        thetas_warmup.append(theta_rand)
        L_values.append(L_theta(theta_rand, omega, sigma_ref))
    return thetas_warmup, L_values



# Parámetros de entrada (ejemplo para testeo inicial)
n_inclusiones = 1
omega = np.logspace(-2, 6, 100) * 2 * np.pi  # frecuencias en rad/s

# Vector "ground truth" para generar referencia
theta_true = [
    0.1, 1e-4, 0.8, 10.0,  # f_1, tau_1, c_1, rho_1
    1.0                    # rho_0 (global)
]
sigma_ref = compute_sigma_e(theta_true, omega)

X_warm, Y_warm = generate_warmup(n_inclusiones, omega, sigma_ref)
# --- Escalonamiento e inversión ---

X_scaled, scaler_info = preprocess_theta(X_warm)
X_recovered = inverse_preprocess_theta(X_scaled, scaler_info)

print (X_warm[1])
print (X_scaled)
print (X_recovered)

