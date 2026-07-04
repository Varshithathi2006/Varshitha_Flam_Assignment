import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
try:
    import emcee
except ImportError:
    print("install emcee : pip install emcee")
    raise
try:
    import corner
except ImportError:
    corner = None
    print("corner not found and skipping corner")
class BCF:    
    def __init__(self, data_path="xy_data.csv", noise_sigma=0.5):
        self.noise_sigma = noise_sigma
        # Load the observed dataset
        df = pd.read_csv(data_path)
        self.points = df[["x", "y"]].to_numpy()
        print(f"Initialized BayesianCurveFitter(BCF) with {len(self.points)} observed points from {data_path}.")
        # Define priors (Bounds)
        # Uniform priors strictly limit the parameter space to physically meaningful bounds.
        self.theta_bounds = (0.0, 50.0)
        self.m_bounds = (-0.05, 0.05)
        self.x_bounds = (0.0, 100.0)
        self.bounds = [self.theta_bounds, self.m_bounds, self.x_bounds]
        # uniformly sample t using 3000 points to get a smooth approx of parametric curve
        self.t_grid = np.linspace(6.0, 60.0, 3000) 
    def build_parametric_curve(self, theta, m, x_shift):
        theta = np.radians(theta)
        envelope = np.exp(m * np.abs(self.t_grid)) * np.sin(0.3 * self.t_grid) # curve that is tangent to each member of the family
        x_pred = self.t_grid * np.cos(theta) - envelope * np.sin(theta) + x_shift
        y_pred = 42 + self.t_grid * np.sin(theta) + envelope * np.cos(theta)
    
        return np.column_stack([x_pred, y_pred])
    def check_prior_support(self, theta, m, x_shift):
        return (self.theta_bounds[0] < theta < self.theta_bounds[1] and
                self.m_bounds[0] < m < self.m_bounds[1] and
                self.x_bounds[0] < x_shift < self.x_bounds[1])
    def residuals(self, theta, m, x_shift):
        generated_curve = self.build_parametric_curve(theta, m, x_shift)
        tree = cKDTree(generated_curve)
        errors, _ = tree.query(self.points)
        return errors
    def posterior_log(self, params):
        theta, m, x_shift = params
        # 1. Log-Prior
        if not self.check_prior_support(theta, m, x_shift):
            return -np.inf
        # 2. Log-Likelihood
        errors = self.residuals(theta, m, x_shift)
        log_likelihood = -0.5 * np.sum((errors / self.noise_sigma) ** 2)
        return log_likelihood
    def parameter_initialization(self):
        def objective(p):
            lp = self.posterior_log(p)
            return -lp if np.isfinite(lp) else 1e10
        print("\n[Step 1] Running Differential Evolution for initial estimate...")
        result = differential_evolution(
            objective, self.bounds,
            seed=42, # optimization efficiency is better at 42
            maxiter=200, # 200 iterations provide sufficient convergence for this,three-parameter optimization problem
            tol=1e-8, # Convergence tolerance.
            polish=True, # final solution to local optimizer
            workers=1  #no.of CPU
        )
        print(f"DE Optimization complete. Initial guess: Theta={result.x[0]:.4f}, M={result.x[1]:.5f}, X={result.x[2]:.4f}")
        return result.x

    def run_mcmc(self, initial_guess, n_walkers=32, n_steps=3000):
        n_dim = len(initial_guess)
        # Initialize walkers in a tiny cloud around the optimized start point
        init_spread = np.array([0.5, 0.001, 0.5]) 
        ensemble_initial_state = initial_guess + init_spread * np.random.randn(n_walkers, n_dim)
        # Clip strictly to prior bounds to avoid -inf likelihood at step 0
        for i, (lo, hi) in enumerate(self.bounds):
            ensemble_initial_state[:, i] = np.clip(ensemble_initial_state[:, i], lo + 1e-6, hi - 1e-6)
        print(f"\n[Step 2] Running Ensemble MCMC Sampler ({n_walkers} walkers, {n_steps} steps)...")
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.posterior_log)
        sampler.run_mcmc(ensemble_initial_state, n_steps, progress=True)
        return sampler
    def figure_plot(self, sampler, burn_in=800):
        print(f"\n[Step 3] Analyzing Posterior Distributions (discarding first {burn_in} steps as burn-in)...")
        try:
            tau = sampler.get_autocorr_time(tol=0)
            thin = max(1, int(np.min(tau) / 2))
        except Exception:
            thin = 10           
        flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        print(f"Retained {len(flat_samples)} posterior samples after burn-in/thinning.")

        theta_samples, m_samples, x_samples = flat_samples.T
        
        theta_mean, theta_std = theta_samples.mean(), theta_samples.std()
        m_mean, m_std = m_samples.mean(), m_samples.std()
        x_mean, x_std = x_samples.mean(), x_samples.std()

        print(" FINAL RECOVERED PARAMETERS")
        print(f" Theta     = {theta_mean:.4f} deg +/- {theta_std:.4f} deg")
        print(f" M         = {m_mean:.5f}      +/- {m_std:.5f}")
        print(f" X         = {x_mean:.4f}      +/- {x_std:.4f}")
        print("plot generation")
        
        # 1. Overlay Plot
        fitted_curve = self.build_parametric_curve(theta_mean, m_mean, x_mean)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.points[:, 0], self.points[:, 1], s=4, alpha=0.5, label="Observed Data", color="#1f77b4")
        plt.plot(fitted_curve[:, 0], fitted_curve[:, 1], color="#d62728", lw=2, label="Fitted Curve (Posterior Mean)")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.title("Observed Data vs. MCMC Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig("fit_overlay.png", dpi=150)
        plt.close()

        # 2. Corner Plot
        labels = [r"$\theta$ (deg)","M Parameter","X Offset"]

        if corner is not None:
            fig = corner.corner(
                flat_samples, 
                
                labels=labels,
                truths=[theta_mean, m_mean, x_mean],
                truth_color="#d62728",
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True
            )
            fig.savefig("posterior_corner.png", dpi=150)
            plt.close()

        print("Saved plots: 'fit_overlay.png', 'posterior_corner.png'")


def main():
    fitter = BCF(data_path="xy_data.csv")
    initial_guess = fitter.parameter_initialization()
    sampler = fitter.run_mcmc(initial_guess=initial_guess, n_walkers=32, n_steps=3000)
    fitter.figure_plot(sampler, burn_in=800) #discard the initial 800 samples and all the markov chains to reach the stationary PD.


if __name__ == "__main__":
    main()