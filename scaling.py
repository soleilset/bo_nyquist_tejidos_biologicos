import numpy as np
from scipy.stats.mstats import winsorize

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
