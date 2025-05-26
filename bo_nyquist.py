import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


# --- Supuestos iniciales ---
# theta contiene: [f1, tau1, c1, rho1, ..., fN, tauN, cN, rhoN, rho_0]
# n_inclusiones: número de inclusiones
# omega: array de frecuencias
# Salida: Re(sigma_e), Im(sigma_e) y gráfico Nyquist

# --- Cálculo de la conductividad ---
def compute_sigma_e(theta, omega):
    theta = np.ravel(theta)
    n_inclusiones = (len(theta) - 1) // 4
    params_por_inclusion = 4
    # --- Descomposición del vector theta ---
    inclusiones = np.reshape(theta[:-1], (n_inclusiones, params_por_inclusion))
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
    sigma_model = compute_sigma_e(theta, omega)
    # Distancia euclídea acumulada (suma de diferencias al cuadrado)
    return np.sum(np.abs(sigma_model - sigma_ref)**2)

# --- Generacion de 10 datos como Warmup para inicializar el GP ---
def generate_warmup(omega, sigma_ref, n_inclusiones,  n_warmup, seed=42):
    assert len(omega) == len(sigma_ref)
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
    L_warmup = []
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
        L_warmup.append(L_theta(theta_rand, omega, sigma_ref))
    return np.array(thetas_warmup), np.array(L_warmup)

# --- Preprocesamiento de theta ---
def preprocess_theta(theta):
    """
    Escala theta según:
    - indetheta mod4 in {0,2} (f_l, c_l): estandarización.
    - indetheta mod4 in {1,3} y última columna (tau_l, rho_l, rho_0): log10 + estandarización.
    Retorna theta_scaled, scaler_info para inversión.
    """
    theta = np.atleast_2d(np.array(theta, dtype=float))
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
    theta_scaled = np.atleast_2d(theta_scaled)
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
    L = np.atleast_1d(np.array(L, dtype=float))
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
    L_scaled = np.atleast_1d(L_scaled)
    logL = L_scaled * sigma + mu
    return np.exp(logL)

def get_normalized_bounds(scaler_info):
    """
    Devuelve un array (2, D) con los límites normalizados para cada parámetro
    usando exclusivamente la información de 'scaler_info' (mu y sigma) de los datos del warmup.
    Los rangos óptimos están definidos internamente.

    - scaler_info: lista de tuplas (scale_type, mu, sigma) de longitud D.
    - Retorna matriz shape (2, D): fila 0 = límites mínimos escalados, fila 1 = límites máximos escalados.
    """
    # Rangos óptimos para cada parámetro en escala real
    # Rangos físicos base por tipo de parámetro
    base_ranges = [
        (0.01, 0.3),    # f_l
        (1e-6, 1e-2),   # tau_l
        (0.5, 1.0),     # c_l
        (1.0, 100.0),   # rho_l
    ]
    # Rango óptimo para rho_0 que no depende de las inclusiones
    rho0_range = (0.1, 10.0)
    D = len(scaler_info)
    # Construir optimal_ranges según patrón [f, tau, c, rho] * inclusiones + rho_0
    optimal_ranges = []
    for j in range(D):
        if j == D - 1:
            optimal_ranges.append(rho0_range)
        else:
            optimal_ranges.append(base_ranges[j % 4])

    # Calcular límites normalizados analíticamente
    bounds = np.zeros((2, D), dtype=float)
    for j, ((scale_type, mu, sigma)) in enumerate(scaler_info):
        min_val, max_val = optimal_ranges[j]
        if scale_type == 'standard':
            lo = (min_val - mu) / sigma
            hi = (max_val - mu) / sigma
        else:  # log_standard
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            lo = (log_min - mu) / sigma
            hi = (log_max - mu) / sigma
        bounds[0, j] = lo
        bounds[1, j] = hi
    return bounds

# --- Adquisición híbrida UCB/EI ---
def hybrid_acquisition(thetas_scaled, L_best_scaled, gp, t, alpha, kappa):
    """
    Combina UCB y EI con peso exponencial w=exp(-t/(alpha*D)).
    """
    thetas_scaled = np.array(thetas_scaled, dtype=float)
    mu_a, sigma_a = gp.predict(thetas_scaled.reshape(1, -1), return_std=True)
    mu = mu_a.item()
    sigma = sigma_a.item()
    # EI
    z = (L_best_scaled - mu) / sigma
    EI = (L_best_scaled - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # UCB
    UCB = mu - kappa * sigma
    # peso de exploración
    D = thetas_scaled.size  # número de dimensiones
    w = np.exp(-t / (alpha * D))    
    return w * UCB + (1 - w) * EI

def propose_next_theta(L_scaled, bounds, gp, t, alpha, kappa, n_restarts):
    """
    Propone el siguiente theta en escala normalizada usando multistart L-BFGS-B.
    """
    bounds = bounds.T.tolist()  # [(lo,hi)...]
    L_best = np.min(np.array(L_scaled))
    best_theta, best_val = None, -np.inf
    for i in range(n_restarts):
        theta0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(lambda x: -hybrid_acquisition(x, L_best, gp, t, alpha, kappa),
                       theta0, method='L-BFGS-B', bounds=bounds)
        if res.success:
            val = res.fun
            if val > best_val:
                best_val, best_theta = val, res.x
    return best_theta, best_val

# --- Proceso iterativo de Optimización Bayesiana ---

def bayes_optimize(omega, sigma_ref, n_inclusiones,
                   warmup_n=10, max_iter=50, max_no_improve=10,
                   epsilon=1e-8, tol=1e-6,
                   alpha=5.0, kappa=2.0, n_restarts=10):
    # Warm-up y límites
    thetas, Ls = generate_warmup(omega, sigma_ref, n_inclusiones, warmup_n)
    theta_scaled, scaler_info = preprocess_theta(thetas)
    bounds = get_normalized_bounds(scaler_info)

    # Inicialización
    idx0 = np.argmin(Ls)
    theta_best, L_best = thetas[idx0], Ls[idx0]
    no_improve = 0
    history_L = [L_best]

    # Bucle principal
    for t in range(1, max_iter+1):
        # Ajuste GP
        Xs,Xs_scaler_info = preprocess_theta(thetas)
        Ys, _ = preprocess_L(Ls)
        D = Xs.shape[1]
        kernel = Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2,1e2), nu=2.5) + WhiteKernel(noise_level=1e-8, noise_level_bounds='fixed')
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False)
        gp.fit(Xs, Ys)

        print(f"[iter {t:02d}] Kernel: {gp.kernel_}")
        print(f"[iter {t:02d}] Length-scales: {gp.kernel_.k1.length_scale}")

        # Propuesta y diagnóstico
        theta_next_scaled, acq_val = propose_next_theta(Ys, bounds, gp, t, alpha, kappa, n_restarts)
        w = np.exp(-t/(alpha*D))
        mu_pred, sigma_pred = gp.predict(theta_next_scaled.reshape(1,-1), return_std=True)
        mu_pred, sigma_pred = mu_pred.item(), sigma_pred.item()

        print(f"[iter {t:02d}] w(t)={w:.3f}, acq={acq_val:.3e}, mu={mu_pred:.3e}, sigma={sigma_pred:.3e}")

        # Inversión y evaluación
        theta_next = inverse_preprocess_theta(theta_next_scaled, Xs_scaler_info)
        L_next = L_theta(theta_next, omega, sigma_ref)
        print(f"[iter {t:02d}] L(theta_next)={L_next:.3e}")

        # Actualizar
        thetas = np.vstack([thetas, theta_next])
        Ls = np.append(Ls, L_next)
        history_L.append(min(L_best, L_next))

        # Mejora
        if L_next < L_best - epsilon:
            theta_best, L_best = theta_next, L_next
            no_improve = 0
        else:
            no_improve += 1

        if L_best < tol:
            print("Convergencia alcanzada: L_best < tol")
            break
        if no_improve >= max_no_improve:
            print("Detenido por falta de mejora")
            break

    # Visualizaciones finales
    plt.figure(figsize=(6,4))
    plt.plot(history_L, '-o')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor L')
    plt.title('Evolución de la función objetivo')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sigma_best = compute_sigma_e(theta_best, omega)
    plt.figure(figsize=(6,6))
    plt.plot(np.real(sigma_ref), -np.imag(sigma_ref), 'o', label='Referencia')
    plt.plot(np.real(sigma_best), -np.imag(sigma_best), '-', label='Modelo óptimo')
    plt.xlabel('Re(σ)')
    plt.ylabel('-Im(σ)')
    plt.title('Diagrama de Nyquist final')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return theta_best, L_best, thetas, Ls

# Ejecución de ejemplo con 1 inclusión
if __name__ == "__main__":
    # Definimos frecuencias omega
    omega = np.logspace(1, 6, 100) * 2 * np.pi
    # Parámetros reales para generar datos sintéticos (1 inclusión):
    # f1, tau1, c1, rho1, rho0
    theta_true = [0.1, 1e-4, 0.8, 10.0, 1.0]
    sigma_ref = compute_sigma_e(theta_true, omega)

    # Ejecutamos la optimización Bayesiana
    theta_opt, L_opt, thetas_hist, Ls_hist = bayes_optimize(
        omega, sigma_ref,
        n_inclusiones=1,
        warmup_n=10,
        max_iter=30,
        max_no_improve=10,
        epsilon=1e-8,
        tol=1e-3,
        alpha=5.0,
        kappa=2.0,
        n_restarts=10
    )

    # Resultados
    print("Theta óptimo (1 inclusión):", theta_opt)
    print("Valor de L óptimo:", L_opt)
    # La función ya genera las gráficas de evolución y Nyquist
