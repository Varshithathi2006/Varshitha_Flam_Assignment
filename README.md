# Bayesian Parameter Estimation of a Parametric Curve Using MCMC

**Name:** Varshitha Thilak Kumar (CB.SC.U4AIE23258)

## Overview
This work uses Markov Chain Monte Carlo (MCMC) and Bayesian inference to estimate the unknown parameters of a nonlinear parametric curve. MCMC allows for both parameter estimation and uncertainty analysis by estimating the probability distribution of each parameter rather than just identifying a single best-fit solution.

After obtaining a good initial estimate through Differential Evolution, the implementation uses the **emcee** affine-invariant ensemble sampler to refine the solution.

---

## Recovered Parameters
| Parameter | Output | Uncertainty (±1σ) | Bounds |
| **θ** | **30.0003°** | ±0.0311° | 0° < θ < 50° |
| **M** | **0.03000** | ±0.00013 | −0.05 < M < 0.05 |
| **X** | **55.0005** | ±0.0278 | 0 < X < 100 |

---

## Estimated Parametric Curve

**Domain:** `6 < t < 60`

\[
\left(
t\cos(0.5236)-e^{0.03|t|}\sin(0.3t)\sin(0.5236)+55,\;
42+t\sin(0.5236)+e^{0.03|t|}\sin(0.3t)\cos(0.5236)
\right)
\]

### Desmos-Compatible Equation

```latex
\left(t*\cos\left(0.5236\right)-e^{0.03\left|t\right|}*\sin\left(0.3t\right)*\sin\left(0.5236\right)+55,\ 42+t*\sin\left(0.5236\right)+e^{0.03\left|t\right|}*\sin\left(0.3t\right)*\cos\left(0.5236\right)\right)
```

### Explicit Equations

```text
x(t) = t cos(0.5236) − e^(0.03|t|) sin(0.3t) sin(0.5236) + 55

y(t) = 42 + t sin(0.5236) + e^(0.03|t|) sin(0.3t) cos(0.5236)
```

where

- θ = 30.0003° (≈ 0.5236 radians)
- M = 0.03000
- X = 55.0005

---

## Methodology

The Markov Chain Monte Carlo (MCMC) method is used to estimate the unknown parameters (θ, M, and X) using Bayesian inference.

Initially, a solid first estimate of the parameters is obtained via Differential Evolution. The MCMC walkers are then initialised near the optimal solution using these parameters.

The **emcee** affine-invariant ensemble sampler is used to carry out the MCMC simulation with:

Thirty-two walkers
3000 steps for sampling
800 steps for burn-in

A parametric curve is created by sampling 3000 points throughout the interval **6 ≤ t ≤ 60** for every candidate parameter set.

The nearest-point Euclidean distance between each observed point and the sampled curve is computed using a **cKDTree** from `scipy.spatial` to assess how closely the created curve matches the observed data. During Bayesian inference, the likelihood is calculated using this distance.

The final parameter values and their uncertainties are estimated using the posterior samples following convergence.
---

## Project Files

| File | Description |
| `mcmc.py` | Python implementation of Bayesian parameter estimation | 
| `xy_data.csv` | 1500-point observed dataset |
| `report.tex` | LaTeX source of the report |
| `report.pdf` | Final project report | 
| `references.bib` | Bibliography file | 
| `fit_overlay.png` | Comparison between fitted curve and observed data |
| `posterior_corner.png` | Posterior distributions of estimated parameters | 
| `trace_plots.png` | MCMC trace plots for convergence analysis |

---

## Python Libraries Used

- NumPy
- SciPy
- Pandas
- Matplotlib
- emcee
- corner

Install the required packages using:

```bash
pip install numpy scipy pandas matplotlib emcee corner
```


## How to Run

1. Open the project folder and add `xy_data.csv`.
2. Set up the necessary Python libraries.
3. Run
```bash python mcmc.py```
It creates the fitted curve, estimates the unknown parameters, and creates the convergence and posterior graphs.
---

## Results

The retrieved parameters closely resemble the initial values that were used to create the dataset. The trace graphs reveal that the MCMC sampler has high convergence, and the posterior distributions display little uncertainty.
---

## References

1. Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013). *emcee: The MCMC Hammer*. PASP, 125(925), 306.
2. Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications*. Biometrika, 57(1), 97–109.
3. Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*. Journal of Chemical Physics, 21(6), 1087–1092.
4. Goodman, J., & Weare, J. (2010). *Ensemble Samplers with Affine Invariance*. CAMCOS, 5(1), 65–80.
5. Storn, R., & Price, K. (1997). *Differential Evolution: A Simple and Efficient Heuristic for Global Optimization*. Journal of Global Optimization, 11(4), 341–359.
6. Gelman, A., & Rubin, D. B. (1992). *Inference from Iterative Simulation Using Multiple Sequences*. Statistical Science, 7(4), 457–472.
7. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
8. Bentley, J. L. (1975). *Multidimensional Binary Search Trees Used for Associative Searching*. Communications of the ACM, 18(9), 509–517.