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

def generate_warmup(omega, sigma_ref, n_inclusiones, n_warmup, seed=42):
    """
    Genera n_warmup puntos de θ usando muestreo Latin Hypercube para
    una mejor cobertura del espacio, combinando escalas lineal y log.
    """
    assert len(omega) == len(sigma_ref)

    # 1) Definimos los rangos “reales” ordenados tal como van en θ
    base_ranges = [
        (0.01, 0.3),    # f_l  (lineal)
        (1e-6, 1e-2),   # tau_l (log)
        (0.5, 1.0),     # c_l  (lineal)
        (1.0, 100.0),   # rho_l (log)
    ]
    rho0_range = (0.1, 10.0)  # rho0 (log)
    D = n_inclusiones * 4 + 1
    ranges_list = []
    for j in range(D):
        if j == D-1:
            ranges_list.append(rho0_range)
        else:
            ranges_list.append(base_ranges[j % 4])

    # 2) Montamos un muestreador LatinHypercube en [0,1]^D
    sampler = qmc.LatinHypercube(d=D, seed=seed)
    unit_samples = sampler.random(n_warmup)  # shape (n_warmup, D)

    # 3) Escalamos a cada rango, usando log-scale donde toca
    lo = np.array([rng[0] for rng in ranges_list])
    hi = np.array([rng[1] for rng in ranges_list])
    # Para los índices log (j%4 in {1,3} ó último), transformamos en log10
    is_log = np.array([(j % 4 in {1,3}) or j == D-1 for j in range(D)])
    # convertimos uniform [0,1] → real:
    thetas = []
    for u in unit_samples:
        theta = np.empty(D)
        # lineal:
        theta[~is_log] = lo[~is_log] + u[~is_log] * (hi[~is_log] - lo[~is_log])
        # log:
        log_lo, log_hi = np.log10(lo[is_log]), np.log10(hi[is_log])
        theta[is_log] = 10 ** (log_lo + u[is_log] * (log_hi - log_lo))
        thetas.append(theta)

    # 4) Calculamos L para cada θ
    thetas = np.array(thetas)
    Ls = np.array([L_theta(theta, omega, sigma_ref) for theta in thetas])

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
    mode : str, opcional {'ei', 'lcb', 'ucb', 'hybrid'}
        Modo de adquisición:
          - 'ei'     : Expected Improvement        
          - 'lcb'    : Lower Confidence Bound (minimizar)
          - 'hybrid' : Mezcla w·LCB + (1–w)·EI, con w=exp(–t/(α·D))

    xi : float
    Parámetro de exploración para PI.
    
    Retorna:
    --------
    float
        Valor de la función de adquisición en thetas_scaled.
    """

    thetas_scaled = np.array(thetas_scaled, dtype=float)
    D = len(thetas_scaled)
    mu_a, sigma_a = gp.predict(thetas_scaled.reshape(1, -1), return_std=True)
    mu = mu_a.item()
    sigma = sigma_a.item()
    
    # EI
    z = (L_best_scaled - mu) / (sigma + 1e-12)  # Evitar división por cero
    EI = (L_best_scaled - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # LCB
    LCB = mu - kappa * sigma  
    # Probability of Improvement (PI)
    PI = norm.cdf((L_best_scaled - mu - xi) / (sigma + 1e-12))
    if mode == 'ei':
        return float(EI)
    elif mode == 'lcb':
        return float(LCB)
    elif mode == 'pi':
        return float(PI)
    elif mode in ('hybrid', 'hybrid_lin'):
        if mode == 'hybrid':
            # peso exponencial
            w = np.exp(-t / (alpha * D))
        else:
            # peso lineal decreciente en [0,1]
            w = max(0.0, 1.0 - t / (alpha * D))

        # mezclamos LCB (exploración) con EI (explotación)
        return float(w * LCB + (1 - w) * EI)
    else:
        raise ValueError(f"Modo desconocido '{mode}'. Elige'ei','pi','lcb','ucb','hybrid' o 'hybrid_lin'.")

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

# --- Propuesta de theta en malla ---
def propose_next_theta_grid(L_scaled, bounds, gp, t, alpha, kappa, mode='hybrid', n_grid=40):
    """
    Igual que `propose_next_theta` pero en vez de L-BFGS-B usa un barrido en malla
    (solo sobre f₁ y τ₁, i.e. θ[0] y θ[1]), manteniendo los otros θ fijos en 0 (el centro
    de la escala normalizada).

    Parámetros:
    -----------
    L_scaled:        array-like de los valores L ya escalados.
    bounds:          array.shape == (2, D) con los límites normalizados para cada θ_j.
    gp:              GaussianProcessRegressor entrenado.
    t:               iteración actual (entero).
    alpha, kappa:    floats para la adquisición.
    mode:            'ei', 'lcb' o 'hybrid'.
    n_grid:          número de puntos por eje en la malla.

    Retorna:
    --------
    best_x:          array (D,) con el θ_scaled que maximiza la adquisición en la malla.
    best_val:        valor máximo de adquisición obtenido.
    """

    # mejor L escalado
    L_best = np.min(L_scaled)

    # dimensiones y límites
    D = bounds.shape[1]
    lo, hi = bounds[0], bounds[1]

    # vamos a barrer solo las dos primeras dimensiones
    grid0 = np.linspace(lo[0], hi[0], n_grid)
    grid1 = np.linspace(lo[1], hi[1], n_grid)

    best_val = -np.inf
    best_x   = None

    # punto base: resto de dims = 0 (centro de la escala)
    base = np.zeros(D, dtype=float)

    for v0 in grid0:
        for v1 in grid1:
            x = base.copy()
            x[0], x[1] = v0, v1
            val = acquisition_fun(x, L_best, gp, t, alpha, kappa, mode)
            if val > best_val:
                best_val = val
                best_x   = x

    return best_x, best_val

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
def diagnose_gp(omega, thetas, Ls, scaler_info, gp):
    """
    Etapa 1: diagnóstico del GP
    Subtareas:
      1. Comparar μ(θ_i) vs. L(θ_i) en puntos nuevos
      2. Visualizar μ y σ sobre el plano (f1, τ1)
      3. Imprimir length_scales por dimensión
    """
    # 1) Comparación μ vs L en puntos nuevos
    # ------------------------------------------------
    # Genero 100 muestras aleatorias en rango físico
    D = scaler_info.__len__()  # dimensiones totales
    # reconstruyo rangos reales a partir de scaler_info
    bounds_real = []
    base = [(0.01,0.3),(1e-6,1e-2),(0.5,1.0),(1.0,100.0)]
    rho0_range = (0.1,10.0)
    for j, (_, mu, sigma) in enumerate(scaler_info):
        if j == D-1: bounds_real.append(rho0_range)
        else:        bounds_real.append(base[j%4])
    # muestreo
    thetas_test = np.array([[np.random.uniform(lo,hi) for lo,hi in bounds_real] for _ in range(100)])
    Ls_test = np.array([L_theta(theta, omega, sigma_ref) for theta in thetas_test])
    # predicción GP
    Xs_test, _ = preprocess_theta(thetas_test)
    mu_pred, _ = gp.predict(Xs_test, return_std=True)
    # L real
    L_true, _ = preprocess_L(Ls_test)
    # gráfico μ vs L
    plt.figure()
    plt.scatter(L_true, mu_pred, alpha=0.6)
    plt.plot([L_true.min(), L_true.max()],[L_true.min(), L_true.max()], 'k--')
    plt.xlabel('L_true')
    plt.ylabel('μ_GP')
    plt.title('Diagnóstico GP: μ vs L')
    plt.show()

    # 2) Mapas de μ y σ para cada par (i,j)
    theta_fix = thetas[-1].copy()
    grid_n = 40
    for i in range(1):
        for j in range(i+1, 2):
            # generar malla
            xi = np.linspace(*bounds_real[i], grid_n)
            xj = np.linspace(*bounds_real[j], grid_n)
            MU = np.zeros((grid_n, grid_n))
            SIG = np.zeros((grid_n, grid_n))
            for ii, vi in enumerate(xi):
                for jj, vj in enumerate(xj):
                    θ = theta_fix.copy()
                    θ[i], θ[j] = vi, vj
                    Xs = scale_single_theta(θ, scaler_info)
                    mu2, sigma2 = gp.predict(Xs.reshape(1, -1) , return_std=True)
                    MU[ii,jj] = mu2
                    SIG[ii,jj] = sigma2

            # plot μ
            plt.figure(figsize=(4,3))
            plt.contourf(xj, xi, MU, levels=20)
            plt.xlabel(f'θ[{j}]'); plt.ylabel(f'θ[{i}]')
            plt.title(f'μ_GP en (θ[{i}],θ[{j}])')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

            # plot σ
            plt.figure(figsize=(4,3))
            plt.contourf(xj, xi, SIG, levels=20)
            plt.xlabel(f'θ[{j}]'); plt.ylabel(f'θ[{i}]')
            plt.title(f'σ_GP en (θ[{i}],θ[{j}])')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    # 3) Imprimir length_scales
    #print("Length-scales del GP:", gp.kernel_.k1.length_scale)

def diagnose_acquisition(theta_base, theta_next, scaler_info, gp, L_best_scaled, t, alpha, kappa, n_grid=40):
    """
    Etapa 2: diagnóstico de la función de adquisición en el plano (f₁, τ₁).
    
    Parámetros:
    -----------     
    theta_base       : array-like, valor de θ en el que fijar los demás parámetros.
    scaler_info      : lista de tuplas (stype, mu, sigma) de preprocess_theta.
    gp               : GaussianProcessRegressor ya entrenado.
    L_best_scaled    : float, mejor L escalado.
    t                : int, iteración actual.
    alpha, kappa     : float, parámetros de acquisition_fun.
    mode             : str, uno de 'ei', 'lcb' o 'hybrid'.
    n_grid           : int, número de puntos por eje en la malla.
    """

    # Reconstruir rangos reales para cada parámetro
    base = [(0.01, 0.3), (1e-6, 1e-2), (0.5, 1.0), (1.0, 100.0)]
    rho0_range = (0.1, 10.0)
    D = len(scaler_info)
    real_bounds = [rho0_range if j == D-1 else base[j % 4] for j in range(D)]

    # Malla sobre f1 y tau1 (índices 0 y 1)

    n ,m = 0, 1
    theta_base = np.atleast_2d(theta_base)[0]
    theta_next = np.atleast_2d(theta_next)[0]
    xvals = np.linspace(*real_bounds[n], n_grid)
    yvals = np.linspace(*real_bounds[m], n_grid)
    # Evaluar acquisition_fun en cada punto de la malla
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    modes = ['ei', 'lcb', 'hybrid']
    for ax, mode in zip(axes, modes):
        Z = np.zeros((n_grid, n_grid))
        for i, xi in enumerate(xvals):
            for j, yj in enumerate(yvals):
                θ = theta_base.copy()
                θ[n], θ[m] = xi, yj
                x_scaled = scale_single_theta(θ, scaler_info)
                x_scaled = np.atleast_2d(x_scaled)[0]
                Z[i, j] = acquisition_fun(x_scaled, L_best_scaled, gp, t, alpha, kappa, mode)
        c = ax.contourf(yvals, xvals, Z, levels=20)
        ax.scatter( theta_next[m], theta_next[n], c='red', s=50, marker='X', label= 'next_evaluation')
        ax.set_title(f'Acquisition ({mode}) en plano (θ[{n}], θ[{m}])')
        ax.set_xlabel(f'θ[{m}]')
        ax.set_ylabel(f'θ[{n}]')
        ax.legend()
        fig.colorbar(c, ax=ax)
    plt.suptitle(f'Diagnóstico de adquisición en iteración {t}')
    plt.tight_layout()
    plt.show()

# --- Proceso iterativo de Optimización Bayesiana ---
def bayes_optimize(omega, sigma_ref, n_inclusiones, mode,
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
    adqisition_vals = []
    weights = []
    # Bucle principal
    for t in range(1, max_iter+1):
        # Ajuste GP
        Xs,Xs_scaler_info = preprocess_theta(thetas)
        Ys, _ = preprocess_L(Ls)
        D = Xs.shape[1]
        """
        # Filtrar NaNs antes de entrenar
        if np.isnan(Xs).any() or np.isnan(Ys).any():
            mask = (~np.isnan(Ys)) & (~np.isnan(Xs).any(axis=1))
            Xs, Ys = Xs[mask], Ys[mask]
            thetas = thetas[mask]
            Ls = Ls[mask]
            Xs, Xs_scaler_info = preprocess_theta(thetas)
            Ys, _ = preprocess_L(Ls)"""


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
            diagnose_gp(omega, thetas, Ls, scaler_info, gp)

        # Propuesta y diagnóstico
        theta_next_scaled, _ = propose_next_theta_random(Ys, bounds, gp, t, alpha, kappa, n_samples=500, mode=mode)
        #propose_next_theta_meshgrid_full(Ys, bounds, gp, t, alpha, kappa, mode, n_per_dim=7)
        #propose_next_theta(Ys, bounds, gp, t, alpha, kappa, n_restarts, mode=mode)
        mu_pred, sigma_pred = gp.predict(theta_next_scaled.reshape(1,-1), return_std=True)
        mu_pred, sigma_pred = mu_pred.item(), sigma_pred.item()

        """ Visualizacion de los pesos y la funcien adquisición """
        weight  = np.exp(-t / (alpha * D))
        adqisition_vals.append(acquisition_fun(theta_next_scaled, Ys.min(), gp, t, alpha, kappa, mode))
        weights.append(weight)

        # Inversión y evaluación
        theta_next = inverse_preprocess_theta(theta_next_scaled, Xs_scaler_info)
        L_next = L_theta(theta_next, omega, sigma_ref)
        print(f"[iter {t:02d}] L(theta_next)={L_next:.3e}")

        """testing acquisition"""
        if t !=0 and t == 1 or t % 5 == 0:
            diagnose_acquisition(theta_best, theta_next, scaler_info, gp, Ys.min(), t, alpha, kappa)

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
    t_vals = np.arange(1, len(weights) + 1) 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    # Subplot 1: evolución de la adquisición
    ax1.plot(t_vals, adqisition_vals, 'o-', label='Acquisition')
    ax1.set_ylabel('Acquisition')
    ax1.set_title('Diagnóstico de adquisición vs peso')
    ax1.grid(True)
    ax1.legend()
    # Subplot 2: evolución del peso w(t)
    ax2.plot(t_vals, weights, 'o-', color='orange', label='Weight $w(t)$')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Weight')
    ax2.grid(True)
    ax2.legend()
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
        mode = 'hybrid_lin',
        warmup_n=20,
        max_iter=30,
        max_no_improve=10,
        epsilon=1e-8,
        tol=1e-3,
        alpha=1.5,
        kappa=0.1,
        n_restarts=30
    )

    # Resultados
    print("Theta óptimo (1 inclusión):", theta_opt)
    print("Valor de L óptimo:", L_opt)
    # La función ya genera las gráficas de evolución y Nyquist
