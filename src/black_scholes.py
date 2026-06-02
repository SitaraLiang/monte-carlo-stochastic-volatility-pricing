import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_d1_d2(S, K, r, sigma, T):
    """
    Compute d1 and d2 for Black-Scholes
    sigma: annualized volatility
    T: time to maturity (in years)
    """
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    log_SK = np.log(S / K)
    sigma_sqrtT = sigma * np.sqrt(T)
    d1 = (log_SK + (r + 0.5 * sigma**2) * T) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT
    return d1, d2


def bs_call(S, K, r, sigma, T):
    """Black-Scholes call price"""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    """
    Black-Scholes put price
    Put = Call - S + K * e^{-rT}
    """
    call = bs_call(S, K, r, sigma, T)
    return call - S + K * np.exp(-r * T)


def bs_greeks(S, K, r, sigma, T, option_type="call"):
    """
    Risk Management: Greeks measure how sensitive the option's price is to changes in market parameters (stock price, time, or volatility)
    """
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    exp_rT = np.exp(-r * T)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d2 = norm.cdf(-d2)

    # Gamma (sensitivity of Delta to asset price changes)
    gamma = n_d1 / (S * sigma * sqrt_T)
    # Vega (sensitivity to volatility changes)
    vega = S * n_d1 * sqrt_T  # per unit of sigma (not per 1%)

    if option_type == "call":
        delta = N_d1 
        theta = (- S * n_d1 * sigma / (2 * sqrt_T)
                 - r * K * exp_rT * N_d2)
        rho_val = K * T * exp_rT * N_d2
    else:
        delta = N_d1 - 1
        theta = (- S * n_d1 * sigma / (2 * sqrt_T)
                 + r * K * exp_rT * N_neg_d2)
        rho_val = -K * T * exp_rT * N_neg_d2

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,   # per 1% move in vol (conventional)
        "theta": theta / 365, # per calendar day
        "rho": rho_val / 100, # per 1bp move in r
    }


def bs_implied_vol(price, S, K, r, T, option_type="call", tol=1e-8, max_iter=200):
    """
    Newton-Raphson implied volatility solver.
    Returns:
    sigma_iv : implied volatility (annualized)
            - "What volatility would make Black-Scholes produce a price of exactly $5?"
    """
    if T <= 0 or price <= 0:
        return np.nan

    df = np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(S - K * df, 0.0)
        upper = S
    else:
        intrinsic = max(K * df - S, 0.0)
        upper = K * df

    # Check if the market price breaks these bounds
    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return np.nan
    if abs(price - intrinsic) < tol:
        return np.nan

    # Initial guess via Brenner-Subrahmanyam approximation (starting point)
    F = S * np.exp(r * T) # forward price: the expected future level of the asset under risk-free growth
    sigma = np.clip(np.sqrt(2 * np.pi / T) * price / F, 0.05, 1.5)

    pricer = bs_call if option_type == "call" else bs_put

    for _ in range(max_iter):
        px = pricer(S, K, r, sigma, T) # option price of model
        d1, _ = bs_d1_d2(S, K, r, sigma, T)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if abs(vega) < 1e-12:
            return np.nan
        sigma -= (px - price) / vega
        if abs(px - price) < tol:
            return sigma
        if sigma <= 0: # Newton-Raphson diverged
            break

    def objective(vol):
        return pricer(S, K, r, vol, T) - price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return np.nan


def vol_surface_bs(S, strikes, maturities, r, sigma):
    """
    Compute BS call prices over a grid of strikes and maturities.
    Returns array of shape (len(maturities), len(strikes)).
    """
    surface = np.zeros((len(maturities), len(strikes)))
    # Loops through every combination of maturity and strike price
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface[i, j] = bs_call(S, K, r, sigma, T)
    return surface


def heston_char_fn(u, t, S0, V0, r, kappa, theta, xi, rho):
    """
    Heston characteristic function phi(u) = E[exp(i*u*ln(S_T/S_0))]

    u: The transform variable (frequency space)
    """
    i = 1j # imaginary i
    d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
    exp_dt = np.exp(-d * t)

    C = (r * i * u * t + kappa * theta / xi**2 * ( (kappa - rho * xi * i * u - d) * t - 2 * np.log((1 - g * exp_dt) / (1 - g))))
    D = ((kappa - rho * xi * i * u - d) / xi**2 * (1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(C + D * V0 + i * u * np.log(S0))


def heston_call_price(S0, K, r, T, V0, kappa, theta, xi, rho,
                      integration_limit=200, n_points=1000):
    """
    Formula:
        C = S0 * P1 - K * exp(-r*T) * P2
    where P1, P2 are probabilities computed via characteristic function inversion
    """

    # Integrate across frequency space to find P1 and P2
    def integrand_P1(u):
        phi = heston_char_fn(u - 1j, T, S0, V0, r, kappa, theta, xi, rho)
        phi0 = heston_char_fn(-1j, T, S0, V0, r, kappa, theta, xi, rho)
        val = (np.exp(-1j * u * np.log(K)) * phi / (1j * u * phi0)).real
        return val

    def integrand_P2(u):
        phi = heston_char_fn(u, T, S0, V0, r, kappa, theta, xi, rho)
        val = (np.exp(-1j * u * np.log(K)) * phi / (1j * u)).real
        return val

    us = np.linspace(1e-6, integration_limit, n_points)

    P1 = 0.5 + (1 / np.pi) * np.trapezoid([integrand_P1(u) for u in us], us)
    P2 = 0.5 + (1 / np.pi) * np.trapezoid([integrand_P2(u) for u in us], us)

    call = S0 * P1 - K * np.exp(-r * T) * P2
    return max(call, 0.0)


def heston_put_price(S0, K, r, T, V0, kappa, theta, xi, rho, **kwargs):
    """Heston put price via put-call parity."""
    call = heston_call_price(S0, K, r, T, V0, kappa, theta, xi, rho, **kwargs)
    return call - S0 + K * np.exp(-r * T)

