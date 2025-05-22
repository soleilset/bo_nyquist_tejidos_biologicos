import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# --- Supuestos iniciales ---
# theta contiene: [f1, tau1, c1, rho1, ..., fN, tauN, cN, rhoN, rho_0]
# n_inclusiones: número de inclusiones
# omega: array de frecuencias
# Salida: Re(sigma_e), Im(sigma_e) y gráfico Nyquist

# --- Cálculo de la conductividad ---
def compute_sigma_e(theta, omega, n_inclusiones):
    params_por_inclusion = 4
    # --- Descomposición del vector theta_true ---
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
def L_theta(theta, omega, sigma_ref, n_inclusiones):
    # Modelado
    sigma_model = compute_sigma_e(theta, omega, n_inclusiones)
    # Distancia euclídea acumulada (suma de diferencias al cuadrado)
    return np.sum(np.abs(sigma_model - sigma_ref)**2)

# --- Rangos para muestreo aleatorio (warmup) ---
ranges = {
    'f_l':    (0.01, 0.3),
    'tau_l':  (1e-6, 1e-2),
    'c_l':    (0.5, 1.0),
    'rho_l':(1.0, 100.0),
    'rho0': (0.1, 10.0)
}

# Generar 10 vectores theta aleatorios como warmup
np.random.seed(42)
thetas_warmup = []
L_values = []

for _ in range(10):
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
    L_values.append(L_theta(theta_rand, omega, sigma_ref, n_inclusiones))




    # Parámetros de entrada (ejemplo para testeo inicial)
n_inclusiones = 1
omega = np.logspace(-2, 6, 100) * 2 * np.pi  # frecuencias en rad/s

# Vector "ground truth" para generar referencia
theta_true = [
    0.1, 1e-4, 0.8, 10.0,  # f_1, tau_1, c_1, rho_1
    1.0                    # rho_0 (global)
]
