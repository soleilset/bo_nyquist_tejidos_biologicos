import numpy as np
import matplotlib.pyplot as plt
import scaling as sc
from main_functions import L_theta, acquisition_fun

def diagnose_gp(omega, thetas, sigma_ref, real_bounds, scaler_info, gp, mode, t):
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
    Xs_test, _ = sc.preprocess_theta(thetas_test)
    mu_pred, _ = gp.predict(Xs_test, return_std=True)
    L_true, _ = sc.preprocess_L(Ls_test)

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
            Xs = sc.scale_single_theta(θ, scaler_info).reshape(1, -1)
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


def diagnose_acquisition(theta_base, theta_next, bounds, scaler_info, gp, L_best_scaled, t, alpha, kappa, model, n_grid=40):
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
                x_scaled = sc.scale_single_theta(θ, scaler_info)
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

    plt.suptitle(f'Diagnóstico de adquisición {model} en iteración {t}')
    plt.tight_layout()
    plt.show()
