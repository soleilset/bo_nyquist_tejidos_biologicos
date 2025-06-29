import numpy as np
from scipy.stats import qmc
import scaling as sc
from main_functions import L_theta

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
        thetas_real = sc.inverse_preprocess_theta(thetas, scaler_info)
    else:
        thetas_real = thetas

    Ls = np.array([L_theta(theta, omega, sigma_ref) for theta in thetas_real])

    return thetas, Ls


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