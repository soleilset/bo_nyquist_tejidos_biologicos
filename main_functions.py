import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import product


# --- Cálculo de la conductividad ---
def compute_sigma_e(theta, omega, n_inclusiones):
    theta = np.ravel(theta)
    # --- Descomposición del vector theta ---
    inclusiones = np.reshape(theta[:-1], (n_inclusiones, 4))
    rho_0 = theta[-1]
    sigma_0 = 1 / rho_0
    jomega = 1j * omega
    sigma_e = np.full_like(omega, sigma_0, dtype=complex)
    for f_l, tau_l, c_l, rho_l in inclusiones:
        M_l = 3 * (rho_0 - rho_l) / (2 * rho_l + rho_0)
        denom = 1 + (jomega * tau_l) ** c_l
        sigma_e += sigma_0 * f_l * M_l * (1 - 1/denom)
    return sigma_e

# --- Definición de la función objetivo L(theta) ---
def L_theta(theta, omega, sigma_ref):
    # Modelado
    theta = np.ravel(theta)  
    n_inclusiones = (len(theta) - 1) // 4  # Último parámetro es rho_0  
    sigma_model = compute_sigma_e(theta, omega, n_inclusiones)
    # Distancia euclídea acumulada (suma de diferencias al cuadrado)
    return np.sum(np.abs(sigma_model - sigma_ref)**2)

def acquisition_fun(thetas_scaled, L_best_scaled, gp, t, alpha, kappa, mode, xi=0.02):
    """
    Función de adquisición configurable.

    Parámetros:
    -----------
    thetas_scaled : array-like, shape (D,)
        Punto candidato en escala normalizada.
    L_best_scaled : float
        Mejor valor de la función objetivo (ya escalado).
    gp : GaussianProcessRegressor
        Modelo GP entrenado.
    t : int
        Iteración actual (para el decaimiento en híbrido).
    alpha : float
        Parámetro de decaimiento para la mezcla híbrida.
    kappa : float
        Parámetro de exploración para UCB/LCB.
    mode : str, opcional {'ei', 'lcb', 'ucb', 'ts', 'hybrid'}
        Modo de adquisición.

    Retorna:
    --------
    float
        Valor de la función de adquisición en thetas_scaled.
    """
    thetas_scaled = np.array(thetas_scaled, dtype=float)
    mu_a, sigma_a = gp.predict(thetas_scaled.reshape(1, -1), return_std=True)
    mu = mu_a.item()
    sigma = sigma_a.item()

    # Expected Improvement
    z = (L_best_scaled - mu) / (sigma + 1e-12)
    EI = (L_best_scaled - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Lower Confidence Bound
    LCB = mu - kappa * sigma

    # Upper Confidence Bound
    UCB = mu + kappa * sigma

    # Probability of Improvement
    PI = norm.cdf((L_best_scaled - mu - xi) / (sigma + 1e-12))

    # Thompson Sampling: muestra de la distribución posterior
    TS_sample = np.random.normal(loc=mu, scale=sigma)

    if mode == 'ei':
        return float(EI)
    elif mode == 'lcb':
        return float(LCB)
    elif mode == 'ucb':
        return float(UCB)
    elif mode == 'ts':
        return float(TS_sample)
    elif mode == 'pi':
        return float(PI)
    elif mode  in ['hybrid', 'hybrid_lin']:
        D = thetas_scaled.size
        if mode == 'hybrid':
            # peso exponencial decreciente en [0,1]
            w = np.exp(-t / (alpha * D))
        else:
            # peso lineal decreciente en [0,1]
            w = max(0.0, 1.0 - t / (alpha * D))
        return float(w * LCB + (1 - w) * EI)
    else:
        raise ValueError(f"Modo desconocido '{mode}'. Elija 'ei', 'lcb', 'ucb', 'ts' o 'hybrid'.")

def propose_next_theta(L_scaled, bounds, gp, t, alpha, kappa, n_restarts, mode):
    """
    Propone el siguiente theta en escala normalizada usando multistart L-BFGS-B.
    """
    bounds = bounds.T.tolist()  # [(lo,hi)...]
    L_best = np.min(np.array(L_scaled))
    best_theta, best_val = None, -np.inf
    for i in range(n_restarts):
        theta0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(lambda x: -acquisition_fun(x, L_best, gp, t, alpha, kappa, mode),
                       theta0, method='L-BFGS-B', bounds=bounds)
        if res.success:
            val = res.fun
            if val > best_val:
                best_val, best_theta = val, res.x
    return best_theta, best_val

# --- Propuesta de theta aleatorio ---
def propose_next_theta_random(L_scaled, bounds, gp, t, alpha, kappa, n_samples=500, mode='hybrid'):
    """
    Propone el siguiente θ_scaled muestreando aleatoriamente n_samples puntos
    en el dominio normalizado y eligiendo el que maximiza la adquisición.
    
    Parámetros:
    -----------
    L_scaled     : array-like, valores de L ya escalados
    bounds       : array shape (2, D) con límites normalizados
    gp           : GaussianProcessRegressor entrenado
    t            : int, iteración actual
    alpha, kappa : float, parámetros de acquisition_fun
    n_samples    : int, número de puntos aleatorios a muestrear
    mode         : 'ei', 'lcb' o 'hybrid'
    
    Retorna:
    --------
    best_x  : array (D,) con θ_scaled que maximiza la adquisición
    best_val: float, valor de adquisición en best_x
    """

    # convertir bounds a lista de tuplas (lo, hi)
    bounds_list = list(zip(bounds[0], bounds[1]))
    D = len(bounds_list)

    # generar muestras uniformes
    lows  = np.array([b[0] for b in bounds_list])
    highs = np.array([b[1] for b in bounds_list])
    X_rand = np.random.uniform(low=lows, high=highs, size=(n_samples, D))

    # evaluar adquisición
    L_best = np.min(L_scaled)
    acq_vals = np.array([
        acquisition_fun(x, L_best, gp, t, alpha, kappa, mode)
        for x in X_rand
    ])

    # escoger el mejor
    idx = np.argmax(acq_vals)
    best_x   = X_rand[idx]
    best_val = acq_vals[idx]
    return best_x, best_val


def propose_next_theta_meshgrid_full(L_scaled, bounds, gp, t, alpha, kappa, mode='hybrid', n_per_dim=5):
    """
    Búsqueda por malla completa sobre todas las dimensiones del espacio normalizado.

    Parámetros:
    -----------
    L_scaled   : array-like de los valores L ya escalados.
    bounds     : ndarray shape (2, D) con [min, max] normalizados para cada θ_j.
    gp         : GaussianProcessRegressor ya entrenado.
    t          : int, iteración actual.
    alpha      : float, parámetro de la adquisición.
    kappa      : float, parámetro de la adquisición.
    mode       : 'ei', 'lcb' o 'hybrid'.
    n_per_dim  : int, número de puntos en la malla por dimensión.

    Retorna:
    --------
    best_x     : ndarray (D,) con el θ_scaled que maximiza la adquisición.
    best_val   : float, valor máximo de adquisición obtenido.
    """
    # mejor L escalado
    L_best = np.min(L_scaled)

    lo, hi = bounds[0], bounds[1]
    D = bounds.shape[1]

    # crear listas de valores equiespaciados por dimensión
    grids = [np.linspace(lo[j], hi[j], n_per_dim) for j in range(D)]

    best_val = -np.inf
    best_x   = None

    # iterar sobre el producto cartesiano de todas las mallas
    for point in product(*grids):
        x = np.array(point, dtype=float)
        val = acquisition_fun(x, L_best, gp, t, alpha, kappa, mode)
        if val > best_val:
            best_val, best_x = val, x

    return best_x, best_val
