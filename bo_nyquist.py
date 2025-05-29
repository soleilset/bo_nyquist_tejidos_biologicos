import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic
from scipy.stats.mstats import winsorize
from itertools import product
from scipy.stats import qmc


# --- Supuestos iniciales ---
# theta contiene: [f1, tau1, c1, rho1, ..., fN, tauN, cN, rhoN, rho_0]
# n_inclusiones: número de inclusiones
# omega: array de frecuencias
# Salida: Re(sigma_e), Im(sigma_e) y gráfico Nyquist

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

# --- Generacion de 10 datos como Warmup para inicializar el GP ---

def generate_warmup(bounds, warmup_n, omega, sigma_ref, scaler_info=None, seed=42):

    bounds = np.array(bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    D = lo.size

    # Muestreo LatinHypercube en [0,1]^D
    sampler = qmc.LatinHypercube(d=D, seed=seed)
    unit = sampler.random(warmup_n)

    # Escalar a [lo, hi]
    thetas = lo + unit * (hi - lo)

    # Si está normalizado, inversión a escala real antes de L_theta
    if scaler_info is not None:
        thetas_real = inverse_preprocess_theta(thetas, scaler_info)
    else:
        thetas_real = thetas

    Ls = np.array([L_theta(theta, omega, sigma_ref) for theta in thetas_real])

    return thetas, Ls


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

def preprocess_L(L, eps=1e-8, limits=(0.05, 0.05)):
    """
    Aplica log(L + eps) y estandarización.
    """
    L = np.atleast_1d(np.array(L, dtype=float))
    L_winsorized = winsorize(L, limits=limits)
    logL = np.log(L_winsorized + eps)
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


def get_real_bounds(D):
    """
    Devuelve:
      - real_ranges:   array shape (2, D) con los rangos de un tejido real.
    """
    # Rangos físicos base por tipo de parámetro
    base_ranges = [
        (0.01, 0.3),    # f_l
        (1e-6, 1e-2),   # tau_l
        (0.5, 1.0),     # c_l
        (1.0, 100.0),   # rho_l
    ]
    rho0_range = (0.1, 10.0)

    # Construir lista de rangos reales (min,max)
    real_list = []
    for j in range(D):
        if j == D - 1:
            real_list.append(rho0_range)
        else:
            real_list.append(base_ranges[j % 4])

    # Convertir a array (2, D)
    real_ranges = np.array(real_list).T  # shape (2,D)

    return real_ranges


def get_normalized_bounds(new_real_bound, scaler_info):
        # Calcular límites normalizados analíticamente
    D = len(scaler_info)    
    real_ranges = new_real_bound
    bounds_scaled = np.zeros((2, D), dtype=float)
    for j, ((scale_type, mu, sigma)) in enumerate(scaler_info):
        lo, hi = real_ranges[:, j]
        if scale_type == 'standard':
            bounds_scaled[0, j] = (lo - mu) / sigma
            bounds_scaled[1, j] = (hi - mu) / sigma
        else:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            bounds_scaled[0, j] = (log_lo - mu) / sigma
            bounds_scaled[1, j] = (log_hi - mu) / sigma
    return bounds_scaled

def extract_good_subset(thetas, Ls, n_inclusiones):
    """
    Selecciona automáticamente el subgrupo 'bueno' basado en el percentil = 10 * n_inclusiones.

    Parámetros:
    -----------
    thetas          : array-like, shape (N, D)
                      Conjunto de muestras θ en escala real.
    Ls              : array-like, shape (N,)
                      Valores de la función objetivo correspondientes.
    n_inclusiones   : int
                      Número de inclusiones (de 1 a 10).

    Retorna:
    --------
    thetas_good     : array, shape (M, D)
                      Subconjunto de θ con L <= percentil.
    Ls_good         : array, shape (M,)
                      Valores L correspondientes.
    threshold       : float
                      Valor de L en el percentil usado.
    """
    pct = min(100, 10 * n_inclusiones)
    threshold = np.percentile(Ls, pct)
    mask = Ls <= threshold
    return thetas[mask], Ls[mask], threshold


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
# escala un único theta (1D) usando scaler_info en nuestro espacio normalizado
def scale_single_theta(theta, scaler_info):
    """
    Escala un único theta (1D) usando scaler_info sin volver a estimar medias/desv.
    """
    arr = np.array(theta, dtype=float)
    D = len(arr)
    x_scaled = np.zeros(D, dtype=float)
    for j, (stype, mu, sigma) in enumerate(scaler_info):
        if sigma == 0:
            sigma = 1.0
        if stype == 'standard':
            x_scaled[j] = (arr[j] - mu) / sigma
        else:  # log_standard
            x_scaled[j] = (np.log10(arr[j]) - mu) / sigma
    return x_scaled


# --- Diagnóstico del GP ---
def diagnose_gp(omega, thetas, real_bounds, scaler_info, gp, mode, t):
    """
    Etapa 1: diagnóstico del GP
    Subtareas:
      1. Comparar μ(θ_i) vs. L(θ_i) en puntos nuevos
      2. Visualizar μ y σ sobre el plano (f1, τ1) en un único plot con dos subplots
      3. Imprimir length_scales por dimensión
    """
    # 1) μ vs L
    # ------------------------------------------------
    bounds_real = real_bounds.T.tolist()
    # muestreo de prueba
    thetas_test = np.array([
        [np.random.uniform(lo, hi) for lo, hi in bounds_real]
        for _ in range(100)
    ])
    Ls_test = np.array([L_theta(th, omega, sigma_ref) for th in thetas_test])
    Xs_test, _ = preprocess_theta(thetas_test)
    mu_pred, _ = gp.predict(Xs_test, return_std=True)
    L_true, _ = preprocess_L(Ls_test)

    plt.figure(figsize=(5,5))
    plt.scatter(L_true, mu_pred, alpha=0.6)
    mn, mx = L_true.min(), L_true.max()
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel('L_true')
    plt.ylabel('μ_GP')
    plt.title(f'Comparación de μ(θ_i) vs L(θ_i) para {mode} – iter {t}')
    plt.tight_layout()
    plt.show()

    # 2) Mapas de μ y σ juntos para (f1, τ1)
    theta_fix = thetas[-1].copy()
    i, j = 0, 1
    xi = np.linspace(*bounds_real[i], 40)
    xj = np.linspace(*bounds_real[j], 40)
    MU = np.zeros((40,40))
    SIG = np.zeros((40,40))
    for ii, vi in enumerate(xi):
        for jj, vj in enumerate(xj):
            θ = theta_fix.copy()
            θ[i], θ[j] = vi, vj
            Xs = scale_single_theta(θ, scaler_info).reshape(1, -1)
            mu2, sigma2 = gp.predict(Xs, return_std=True)
            MU[ii, jj] = mu2
            SIG[ii, jj] = sigma2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    c1 = ax1.contourf(xj, xi, MU, levels=20)
    ax1.set_title(f'GP μ (θ[{i}],θ[{j}])')
    ax1.set_xlabel(f'θ[{j}]'); ax1.set_ylabel(f'θ[{i}]')
    fig.colorbar(c1, ax=ax1)

    c2 = ax2.contourf(xj, xi, SIG, levels=20)
    ax2.set_title(f'GP σ (θ[{i}],θ[{j}])')
    ax2.set_xlabel(f'θ[{j}]'); ax2.set_ylabel(f'θ[{i}]')
    fig.colorbar(c2, ax=ax2)
    plt.suptitle(f'Diagnóstico del GP para la adquisicion {mode} en iteración {t}')
    plt.tight_layout()
    plt.show()

    # 3) Length-scales
    try:
        ls = gp.kernel_.k1.length_scale
    except:
        ls = getattr(gp.kernel_, 'length_scale', None)
    print("Length-scales del GP:", ls)


def diagnose_acquisition(theta_base, theta_next, bounds, scaler_info, gp, L_best_scaled, t, alpha, kappa, n_grid=40):
    """
    Etapa 2: diagnóstico de la función de adquisición en 6 modos,
    extendiendo la malla para incluir theta_next si queda fuera.
    """
    # límites normalizados de cada dimensión [(lo, hi), ...]
    real_bounds = bounds.T.tolist()

    # fijar índices de f1 y tau1
    n, m = 0, 1

    # asegurar vectores 1D
    theta_base = np.ravel(theta_base)
    theta_next = np.ravel(theta_next)

    # extraer límites originales para dims n,m
    lo_n, hi_n = real_bounds[n]
    lo_m, hi_m = real_bounds[m]

    # extender el dominio para incluir theta_next si hace falta
    x_min = min(lo_n, theta_next[n])
    x_max = max(hi_n, theta_next[n])
    y_min = min(lo_m, theta_next[m])
    y_max = max(hi_m, theta_next[m])

    # malla adaptada
    xvals = np.linspace(x_min, x_max, n_grid)
    yvals = np.linspace(y_min, y_max, n_grid)

    # preparar figura 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    modes = ['ei', 'lcb', 'hybrid', 'hybrid_lin', 'pi', 'ts']
    axes_flat = axes.flatten()

    for ax, mode in zip(axes_flat, modes):
        Z = np.zeros((n_grid, n_grid))
        for i, xi in enumerate(xvals):
            for j, yj in enumerate(yvals):
                θ = theta_base.copy()
                θ[n], θ[m] = xi, yj
                x_scaled = scale_single_theta(θ, scaler_info)
                Z[i, j] = acquisition_fun(x_scaled, L_best_scaled, gp, t, alpha, kappa, mode)
        c = ax.contourf(yvals, xvals, Z, levels=20)
        ax.scatter(
            theta_next[m], theta_next[n],
            c='red', s=50, marker='X', label='next_eval'
        )
        ax.set_title(f'{mode.upper()}  (θ[{n}],θ[{m}])')
        ax.set_xlabel(f'θ[{m}]')
        ax.set_ylabel(f'θ[{n}]')
        ax.legend()
        fig.colorbar(c, ax=ax)

    plt.suptitle(f'Diagnóstico de adquisición {mode} en iteración {t}')
    plt.tight_layout()
    plt.show()


# --- Proceso iterativo de Optimización Bayesiana ---
def bayes_optimize(omega, sigma_ref, n_inclusiones, mode,
                   warmup_n=10, max_iter=50, max_no_improve=10,
                   epsilon=1e-8, tol=1e-6,
                   alpha=5.0, kappa=2.0, n_restarts=10):
    # Warm-up y límitesget_real_bounds((4*n_inclusiones)+1), warmup_n
        # 1) Warm-up inicial en todo el dominio real
    thetas, Ls = generate_warmup(get_real_bounds(4*n_inclusiones+1), warmup_n, omega, sigma_ref)

    # 2) Extraer subconjunto “bueno” con percentil = 10·n_inclusiones
    thetas_good, Ls_good, threshold = extract_good_subset(thetas, Ls, n_inclusiones)

    # 3) Calcular μ y σ de ese subgrupo
    mu    = thetas_good.mean(axis=0)
    sigma = thetas_good.std(axis=0)

    # 4) Definir nuevo dominio real alrededor de μ±σ (k=1)
    k = 1.0
    real_lo = mu - k*sigma
    real_hi = mu + k*sigma
    real_dom = get_real_bounds((4*n_inclusiones)+1)
    real_lo = np.maximum(real_lo, real_dom[0])
    real_hi = np.minimum(real_hi, real_dom[1])
    new_real_bounds = np.vstack([real_lo, real_hi])

    # 5) Reinicializar warm-ups en el subdominio concentrado
    thetas, Ls = generate_warmup(new_real_bounds, warmup_n, omega, sigma_ref)

    # 6) Normalizar y obtener bounds para BO
    theta_scaled, scaler_info = preprocess_theta(thetas)
    bounds = get_normalized_bounds(new_real_bounds, scaler_info)

    # --- Ahora continúa con la inicialización y el bucle GP/adquisición ---
    idx0 = np.argmin(Ls)
    theta_best, L_best = thetas[idx0], Ls[idx0]
    no_improve = 0
    history_L = [L_best]
    adqisition_vals = []
    for t in range(1, max_iter+1):
        # Ajuste GP
        Xs,Xs_scaler_info = preprocess_theta(thetas)
        Ys, _ = preprocess_L(Ls)
        D = Xs.shape[1]
        
        # Filtrar NaNs antes de entrenar
        if np.isnan(Xs).any() or np.isnan(Ys).any():
            mask = (~np.isnan(Ys)) & (~np.isnan(Xs).any(axis=1))
            Xs, Ys = Xs[mask], Ys[mask]
            thetas = thetas[mask]
            Ls = Ls[mask]

        """kernel = RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e5)) \
               + Matern(length_scale=np.ones(D), nu=0.5, length_scale_bounds=(1e-2, 1e5))"""

        """kernel = DotProduct() * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e5)) \
            + Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e5), nu=1.5)""" 
        best_score = np.inf
        best_gp = None
        best_kernel = None
        for i in range(5):
            kernel = (ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.random.uniform(1e-1, 1e3, size=D), length_scale_bounds=(1e-1, 1e3))
            * RationalQuadratic(length_scale=np.random.uniform(1e-1, 1e3), alpha=1.0)
            * Matern(length_scale=np.random.uniform(1e-1, 1e3, size=D), length_scale_bounds=(1e-1, 1e3), nu=1.5) )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-5,
                normalize_y=False,
                n_restarts_optimizer=5)
            gp.fit(Xs, Ys)
            score = -gp.log_marginal_likelihood()
            if score < best_score:
                best_score = score
                best_gp = gp
                best_kernel = gp.kernel_
        print(best_kernel)
        gp = best_gp

        """testing GP"""
        if t == 1 or t % 5 == 0:
            diagnose_gp(omega, thetas, new_real_bounds, Xs_scaler_info, gp, mode=mode, t=t)

        # Propuesta y diagnóstico
        theta_next_scaled, _ = propose_next_theta_random(Ys, bounds, gp, t, alpha, kappa, n_samples=500, mode=mode)
        #propose_next_theta(Ys, bounds, gp, t, alpha, kappa, n_restarts, mode=mode)
        #propose_next_theta_meshgrid_full(Ys, bounds, gp, t, alpha, kappa, mode, n_per_dim=7)
        mu_pred, sigma_pred = gp.predict(theta_next_scaled.reshape(1,-1), return_std=True)
        mu_pred, sigma_pred = mu_pred.item(), sigma_pred.item()

        """ Visualizacion de la funcien adquisición """
        adqisition_vals.append(acquisition_fun(theta_next_scaled, Ys.min(), gp, t, alpha, kappa, mode))

        # Inversión y evaluación
        theta_next = inverse_preprocess_theta(theta_next_scaled, Xs_scaler_info)
        L_next = L_theta(theta_next, omega, sigma_ref)
        print(f"[iter {t:02d}] L(theta_next)={L_next:.3e}")

        """testing acquisition"""
        if t == 1 or t % 5 == 0:
            diagnose_acquisition(theta_best, theta_next, new_real_bounds, Xs_scaler_info, gp, Ys.min(), t, alpha, kappa)

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
            print("La mafucking bestia, se logro maldita sea")
            break
        if no_improve >= max_no_improve:
            print("Detenido por falta de mejora")
            break


    # Evluacion de pesos y adquicision
    t_vals = np.arange(1, len(adqisition_vals) + 1) 
    plt.plot(t_vals, adqisition_vals, 'o-', label='Acquisition')
    plt.ylabel('Acquisition')
    plt.xlabel('Iteración')
    plt.title('Diagnóstico de adquisición vs iteracion')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualizaciones finales
    plt.figure(figsize=(6,4))
    plt.plot(history_L, '-o')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor L')
    plt.title('Evolución de la función objetivo')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sigma_best = compute_sigma_e(theta_best, omega, n_inclusiones)
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
    theta_true = [0.1, 1e-4, 0.8, 10, 1.0]
    sigma_ref = compute_sigma_e(theta_true, omega, 1)

    # Ejecutamos la optimización Bayesiana
    theta_bo, L_bo, thetas_hist, Ls_hist = bayes_optimize(
        omega, sigma_ref,
        n_inclusiones=1, 
        mode = 'ei',
        warmup_n=25,
        max_iter=20,
        max_no_improve=20,
        epsilon=1e-8,
        tol=1e-3,
        alpha=1/4,
        kappa=0.2,
        n_restarts=20
    )

    # Resultados
    print("Theta óptimo (1 inclusión):", theta_bo)
    print("Valor de L óptimo:", L_bo)
