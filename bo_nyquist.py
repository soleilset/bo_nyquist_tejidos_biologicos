import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic
import scaling as sc
import search_zone as sz
import diagnose_functions as df
import main_functions as mf


# --- Supuestos iniciales ---
# theta contiene: [f1, tau1, c1, rho1, ..., fN, tauN, cN, rhoN, rho_0]
# n_inclusiones: número de inclusiones
# omega: array de frecuencias
# Salida: Re(sigma_e), Im(sigma_e) y gráfico Nyquist

# --- Proceso iterativo de Optimización Bayesiana ---
def bayes_optimize(omega, sigma_ref, n_inclusiones, mode,
                   warmup_n=10, max_iter=50, max_no_improve=10,
                   epsilon=1e-8, tol=1e-6,
                   alpha=5.0, kappa=2.0, n_restarts=10):
    # Warm-up y límitesget_real_bounds((4*n_inclusiones)+1), warmup_n
        # 1) Warm-up inicial en todo el dominio real
    thetas, Ls = sz.generate_warmup(sz.get_real_bounds(4*n_inclusiones+1), warmup_n, omega, sigma_ref)

    # 2) Extraer subconjunto “bueno” con percentil = 10·n_inclusiones
    thetas_good, Ls_good, threshold = sz.extract_good_subset(thetas, Ls, n_inclusiones)

    # 3) Calcular μ y σ de ese subgrupo
    mu    = thetas_good.mean(axis=0)
    sigma = thetas_good.std(axis=0)

    # 4) Definir nuevo dominio real alrededor de μ±σ (k=1)
    k = 1.0
    real_lo = mu - k*sigma
    real_hi = mu + k*sigma
    real_dom = sz.get_real_bounds((4*n_inclusiones)+1)
    real_lo = np.maximum(real_lo, real_dom[0])
    real_hi = np.minimum(real_hi, real_dom[1])
    new_real_bounds = np.vstack([real_lo, real_hi])

    # 5) Reinicializar warm-ups en el subdominio concentrado
    thetas, Ls = sz.generate_warmup(new_real_bounds, warmup_n, omega, sigma_ref)

    # 6) Normalizar y obtener bounds para BO
    theta_scaled, scaler_info = sc.preprocess_theta(thetas)
    bounds = sc.get_normalized_bounds(new_real_bounds, scaler_info)

    # --- Ahora continúa con la inicialización y el bucle GP/adquisición ---
    idx0 = np.argmin(Ls)
    theta_best, L_best = thetas[idx0], Ls[idx0]
    no_improve = 0
    history_L = [L_best]
    adqisition_vals = []
    for t in range(1, max_iter+1):
        # Ajuste GP
        Xs,Xs_scaler_info = sc.preprocess_theta(thetas)
        Ys, _ = sc.preprocess_L(Ls)
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
            df.diagnose_gp(omega, thetas, sigma_ref, new_real_bounds, Xs_scaler_info, gp, mode=mode, t=t)

        # Propuesta y diagnóstico
        theta_next_scaled, _ = mf.propose_next_theta_random(Ys, bounds, gp, t, alpha, kappa, n_samples=500, mode=mode)
        #propose_next_theta(Ys, bounds, gp, t, alpha, kappa, n_restarts, mode=mode)
        #propose_next_theta_meshgrid_full(Ys, bounds, gp, t, alpha, kappa, mode, n_per_dim=7)
        mu_pred, sigma_pred = gp.predict(theta_next_scaled.reshape(1,-1), return_std=True)
        mu_pred, sigma_pred = mu_pred.item(), sigma_pred.item()

        """ Visualizacion de la funcien adquisición """
        adqisition_vals.append(mf.acquisition_fun(theta_next_scaled, Ys.min(), gp, t, alpha, kappa, mode))

        # Inversión y evaluación
        theta_next = sc.inverse_preprocess_theta(theta_next_scaled, Xs_scaler_info)
        L_next = mf.L_theta(theta_next, omega, sigma_ref)
        print(f"[iter {t:02d}] L(theta_next)={L_next:.3e}")

        """testing acquisition"""
        if t == 1 or t % 5 == 0:
            df.diagnose_acquisition(theta_best, theta_next, new_real_bounds, Xs_scaler_info, gp, Ys.min(), t, alpha, kappa, model=mode)

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
            print("La motherfucking bestia, se logro maldita sea")
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

    sigma_best = mf.compute_sigma_e(theta_best, omega, n_inclusiones)
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
    sigma_ref = mf.compute_sigma_e(theta_true, omega, 1)

    # Ejecutamos la optimización Bayesiana
    theta_bo, L_bo, thetas_hist, Ls_hist = bayes_optimize(
        omega, sigma_ref,
        n_inclusiones=1, 
        mode = 'ei',
        warmup_n=35,
        max_iter=10,
        max_no_improve=10,
        epsilon=1e-8,
        tol=1e-3,
        alpha=1/4,
        kappa=0.2,
        n_restarts=20
    )

    # Resultados
    print("Theta óptimo (1 inclusión):", theta_bo)
    print("Valor de L óptimo:", L_bo)
