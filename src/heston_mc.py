import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.special import ndtr


@dataclass
class HestonParams:
    S0: float = 100.0 # initial stock price
    V0: float = 0.04
    r: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    T: float = 1.0

    @property
    def feller_condition(self) -> float:
        """Returns 2*kappa*theta / xi^2. Should be > 1 for strong Feller"""
        return 2 * self.kappa * self.theta / self.xi**2
    
    def __post_init__(self):
        if self.feller_condition < 1:
            print(f"Feller condition not satisfied: "
                  f"2κθ/ξ² = {self.feller_condition:.3f} < 1"
                  f"Variance process may hit zero")


SchemeType = Literal["euler", "milstein", "qe"]


def simulate_heston(
    params: HestonParams,
    n_paths: int = 10_000,
    n_steps: int = 252, # trading days per year
    scheme: SchemeType = "euler",
    return_paths: bool = False,
    antithetic: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Simulate Heston paths using the chosen discretization scheme

    antithetic: use antithetic variates for variance reduction

    Returns
    dict with keys:
        S_T        : terminal spot prices, shape (n_paths_total,)
        V_T        : terminal variances
        S_paths    : full paths if return_paths=True, else None
        V_paths    : full paths if return_paths=True, else None
        n_paths    : effective number of paths used
        dt         : time step size
        scheme     : scheme name
    """
    rng = np.random.default_rng(seed) # random number generator
    p = params
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)

    # only need to explicitly calculate half of the paths, the other half are calculated mirrored.
    if antithetic:
        n_sim = n_paths // 2
    else:
        n_sim = n_paths

    S = np.full(n_sim, p.S0) # shape (n_sim,)
    V = np.full(n_sim, p.V0) # shape (n_sim,)

    if return_paths:
        S_paths = np.zeros((n_sim, n_steps + 1))
        V_paths = np.zeros((n_sim, n_steps + 1))
        S_paths[:, 0] = S # shape (n_sim, n_steps+1)
        V_paths[:, 0] = V # shape (n_sim, n_steps+1)
        if antithetic:
            S_anti = np.full(n_sim, p.S0)
            V_anti = np.full(n_sim, p.V0)
            S_paths_anti = np.zeros((n_sim, n_steps + 1))
            V_paths_anti = np.zeros((n_sim, n_steps + 1))
            S_paths_anti[:, 0] = S_anti
            V_paths_anti[:, 0] = V_anti
    else:
        if antithetic:
            S_anti = np.full(n_sim, p.S0)
            V_anti = np.full(n_sim, p.V0)

    # Cholesky decomposition for correlated Brownians
    # W^S = Z1, W^V = rho*Z1 + sqrt(1-rho^2)*Z2
    sqrt_one_minus_rho2 = np.sqrt(1 - p.rho**2)

    dispatch = {
        "euler": _step_euler,
        "milstein": _step_milstein,
        "qe": _step_qe,
    }
    step_fn = dispatch[scheme]

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_sim) # shape (n_sim,)
        Z2 = rng.standard_normal(n_sim) # shape (n_sim,)
        ZS = Z1
        ZV = p.rho * Z1 + sqrt_one_minus_rho2 * Z2

        S, V = step_fn(S, V, ZS, ZV, dt, sqrt_dt, p)

        if antithetic:
            S_anti, V_anti = step_fn(S_anti, V_anti, -ZS, -ZV, dt, sqrt_dt, p)

        if return_paths:
            S_paths[:, t + 1] = S
            V_paths[:, t + 1] = V
            if antithetic:
                S_paths_anti[:, t + 1] = S_anti
                V_paths_anti[:, t + 1] = V_anti

    if antithetic:
        S_T = np.concatenate([S, S_anti]) # shape (n_sim*2,), 
        V_T = np.concatenate([V, V_anti]) # shape (n_sim*2,)
        n_total = 2 * n_sim
    else:
        S_T = S
        V_T = V
        n_total = n_sim

    result = {
        "S_T": S_T, # only return terminal values by default to save memory
        "V_T": V_T,
        "S_paths": None,
        "V_paths": None,
        "n_paths": n_total,
        "dt": dt,
        "scheme": scheme,
        "T": p.T
    }

    if return_paths: # for asian options
        if antithetic:
            result["S_paths"] = np.vstack([S_paths, S_paths_anti])
            result["V_paths"] = np.vstack([V_paths, V_paths_anti])
        else:
            result["S_paths"] = S_paths
            result["V_paths"] = V_paths

    return result



def _step_euler(S, V, ZS, ZV, dt, sqrt_dt, p):
    """
    Euler-Maruyama with full truncation
    """
    # avoid standard Euler who may accidentally step into negative variance territory
    V_plus = np.maximum(V, 0.0)
    sqrt_V = np.sqrt(V_plus)

    # ln(S): eliminatenegative stock prices
    log_S = np.log(S) + (p.r - 0.5 * V_plus) * dt + sqrt_V * sqrt_dt * ZS
    S_new = np.exp(log_S)

    # Variance
    V_new = V + p.kappa * (p.theta - V_plus) * dt + p.xi * sqrt_V * sqrt_dt * ZV

    return S_new, V_new


def _step_milstein(S, V, ZS, ZV, dt, sqrt_dt, p):
    """
    Milstein scheme with full truncation.
    """
    V_plus = np.maximum(V, 0.0)
    sqrt_V = np.sqrt(V_plus)

    # Same as Euler
    log_S = np.log(S) + (p.r - 0.5 * V_plus) * dt + sqrt_V * sqrt_dt * ZS
    S_new = np.exp(log_S)

    # Adds a correction term of order dt to the Euler for V in order to correct the curvature of the diffusion coefficient
    milstein_correction = 0.25 * p.xi**2 * dt * (ZV**2 - 1)
    V_new = V + p.kappa * (p.theta - V_plus) * dt + p.xi * sqrt_V * sqrt_dt * ZV + milstein_correction

    return S_new, V_new


def _step_qe(S, V, ZS, ZV, dt, sqrt_dt, p):
    """
    Quadratic-Exponential (QE), matches the first two conditional moments of the CIR process
    (the variance process in Heston) exactly, without requiring V >= 0
    It switches between two representations based on the ratio psi = var/mean^2:
      - psi <= psi_c : quadratic approximation (Gaussian-like regime)
      - psi >  psi_c : exponential approximation (chi-squared-like regime)
    """
    kappa_bar = p.kappa
    theta_bar = p.theta
    xi = p.xi

    exp_kdt = np.exp(-kappa_bar * dt)
    m = theta_bar + (V - theta_bar) * exp_kdt

    # Conditional variance of V_{t+dt} | V_t 
    s2 = (V * xi**2 * exp_kdt / kappa_bar * (1 - exp_kdt) + theta_bar * xi**2 / (2 * kappa_bar) * (1 - exp_kdt)**2)
    s2 = np.maximum(s2, 0.0)

    psi = s2 / (m**2 + 1e-12)
    psi_c = 1.5 # original paper

    V_new = np.empty_like(V)

    # Quadratic regime (psi <= psi_c)
    mask_q = psi <= psi_c
    if mask_q.any():
        psi_q = psi[mask_q]
        m_q = m[mask_q]
        b2 = 2 / psi_q - 1 + np.sqrt(2 / psi_q * (2 / psi_q - 1))
        a = m_q / (1 + b2)
        b = np.sqrt(b2)
        Zv_q = ZV[mask_q]
        V_new[mask_q] = a * (b + Zv_q)**2

    # Exponential regime (psi > psi_c)
    # If the variance drifts close to zero, its probability profile flattens into an exponential distribution with a discrete probability mass concentrated exactly at zero.
    mask_e = ~mask_q
    if mask_e.any():
        psi_e = psi[mask_e]
        m_e = m[mask_e]
        p_exp = (psi_e - 1) / (psi_e + 1)
        beta = 2 / (m_e * (psi_e + 1))
        # transform random normal variables into a uniform distribution
        U = 0.5 * (1 + ZV[mask_e])
        U = ndtr(ZV[mask_e])
        V_new_e = np.where(U > p_exp, np.log((1 - p_exp) / (1 - U + 1e-12)) / (beta + 1e-12), 0.0)
        V_new[mask_e] = V_new_e

    # Update asset price using the matching conditional variance profile
    # QE only modifies the variance update, the asset price update is the same as Euler with full truncation
    V_plus = np.maximum(V, 0.0)
    sqrt_V = np.sqrt(V_plus)
    log_S = np.log(S) + (p.r - 0.5 * V_plus) * dt + sqrt_V * sqrt_dt * ZS
    S_new = np.exp(log_S)

    return S_new, V_new



