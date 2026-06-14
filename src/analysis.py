import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm as spnorm, kurtosis, skew
from scipy.stats import lognorm
from heston_mc import HestonParams, simulate_heston
from black_scholes import (bs_call, bs_put, bs_implied_vol,
                           heston_call_price, heston_put_price)
from options import (european_call, european_put, asian_call, asian_put,
                     up_and_out_call, down_and_out_put, variance_swap,
                     mc_implied_vol_smile)

warnings.filterwarnings("ignore")

PARAMS = HestonParams(
    S0=100.0,
    V0=0.04,
    r=0.05,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    T=1.0,
)
K = 100.0 # strike price for European/Asian/Barrier options
SEED = 42
SCHEMES = ["euler", "milstein", "qe"]
SCHEME_COLORS = {"euler": "#E63946", "milstein": "#F4A261", "qe": "#2A9D8F"}
SCHEME_LABELS = {"euler": "Euler-Maruyama", "milstein": "Milstein", "qe": "QE (Andersen)"}

# helper
def _add_sim_meta(result, params, S0=None):
    result["T"] = params.T
    result["S0"] = S0 or params.S0
    return result


def convergence_study(params=PARAMS, K=K, n_steps=252, seed=SEED,
                      path_counts=None):
    """
    Verifies how fast the Monte Carlo pricing model narrows down on the right answer as the simulation paths grow
    use QE (most accurate)
    """
    if path_counts is None:
        path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    # calculate reference price using fourier inversion of the Heston function
    # baseline we want to converge to
    ref_price = heston_call_price(
        params.S0, K, params.r, params.T,
        params.V0, params.kappa, params.theta, params.xi, params.rho
    )

    results = []
    for n in path_counts:
        sim = simulate_heston(params, n_paths=n, n_steps=n_steps,
                              scheme="qe", antithetic=True, seed=seed)
        _add_sim_meta(sim, params)
        res = european_call(sim, K, params.r)
        res["n_paths"] = n
        res["error"] = abs(res["price"] - ref_price)
        results.append(res)

    return results, ref_price


def discretization_bias_study(params=PARAMS, K=K, n_paths=50000, seed=SEED,
                               step_counts=None):
    """
    European call price vs number of time steps for each scheme (euler, milstein, QE)
    Measures convergence to the true Heston price as dt -> 0 (nb_steps -> infinity)
    """
    if step_counts is None:
        step_counts = [10, 20, 50, 100, 252, 500, 1000]

    # baseline
    ref_price = heston_call_price(
        params.S0, K, params.r, params.T,
        params.V0, params.kappa, params.theta, params.xi, params.rho
    )

    all_results = {scheme: [] for scheme in SCHEMES}

    for scheme in SCHEMES:
        for n_steps in step_counts:
            sim = simulate_heston(params, n_paths=n_paths, n_steps=n_steps,
                                  scheme=scheme, antithetic=True, seed=seed)
            _add_sim_meta(sim, params)
            res = european_call(sim, K, params.r)
            res["n_steps"] = n_steps
            res["dt"] = params.T / n_steps
            res["bias"] = res["price"] - ref_price
            res["abs_bias"] = abs(res["bias"])
            all_results[scheme].append(res)

    return all_results, ref_price


def volatility_smile(params=PARAMS, n_paths=50000, n_steps=252, seed=SEED,
                     moneyness=None):
    """Implied volatility smile under Heston vs flat Black-Scholes volatility"""

    if moneyness is None:
        # 0.7 to 1.3 by convention
        moneyness = np.linspace(0.7, 1.3, 25)

    # moneyess = strike / S0, strike = moneyness * S0
    strikes = moneyness * params.S0
    flat_vol = np.sqrt(params.theta) # long-run volatility under Heston (where volatility never changes)

    # obtain simulated option prices across every strike
    sim = simulate_heston(params, n_paths=n_paths, n_steps=n_steps,
                          scheme="qe", antithetic=True, seed=seed)
    _add_sim_meta(sim, params)

    # Black-Scholes volatility (running bs in inverse direction)
    ivols_mc, _ = mc_implied_vol_smile(sim, strikes, params.r, "call")

    # Heston semi-analytical implied volatilities
    ivols_heston = []
    for K_i in strikes:
        px = heston_call_price(
            params.S0, K_i, params.r, params.T,
            params.V0, params.kappa, params.theta, params.xi, params.rho
        )
        iv = bs_implied_vol(px, params.S0, K_i, params.r, params.T, "call")
        ivols_heston.append(iv)
    ivols_heston = np.array(ivols_heston)

    # BS flat volatilities (constant across strikes)
    ivols_bs = np.full_like(strikes, flat_vol)

    return {
        "moneyness": moneyness,
        "strikes": strikes,
        "ivols_mc": ivols_mc,
        "ivols_heston": ivols_heston,
        "ivols_bs": ivols_bs,
        "flat_vol": flat_vol,
    }


def price_all_options(params=PARAMS, K=K, n_paths=50000, n_steps=252,
                      seed=SEED, scheme="qe"):
    """
    Price European, Asian, Barrier, and Variance Swap under Heston.
    Returns a summary dict.
    """
    # here we suppose S0=100 
    barrier_up = 130.0 
    barrier_down = 80.0

    sim = simulate_heston(params, n_paths=n_paths, n_steps=n_steps,
                          scheme=scheme, antithetic=True,
                          return_paths=True, seed=seed)
    _add_sim_meta(sim, params)

    results = {}

    # European
    results["european_call"] = european_call(sim, K, params.r)
    results["european_put"] = european_put(sim, K, params.r)

    # Analytical benchmarks
    results["european_call"]["heston_analytical"] = heston_call_price(
        params.S0, K, params.r, params.T,
        params.V0, params.kappa, params.theta, params.xi, params.rho
    )

    results["european_put"]["heston_analytical"] = heston_put_price(
        params.S0, K, params.r, params.T,
        params.V0, params.kappa, params.theta, params.xi, params.rho
    )
    
    results["european_call"]["bs_analytical"] = bs_call(
        params.S0, K, params.r, np.sqrt(params.theta), params.T
    )
    results["european_put"]["bs_analytical"] = bs_put(
        params.S0, K, params.r, np.sqrt(params.theta), params.T
    )

    # Asian
    results["asian_call"] = asian_call(sim, K, params.r)
    results["asian_put"] = asian_put(sim, K, params.r)

    # Barrier
    results["up_out_call"] = up_and_out_call(sim, K, params.r, barrier_up)
    results["down_out_put"] = down_and_out_put(sim, K, params.r, barrier_down)

    # Variance swap
    results["variance_swap"] = variance_swap(sim, r=params.r)

    return results, sim


STYLE = {
    "figure.facecolor": "#FFFFFF",       
    "axes.facecolor": "#FFFFFF",     
    "axes.edgecolor": "#111111",      
    "axes.labelcolor": "#111111",    
    "xtick.color": "#333333",   
    "ytick.color": "#333333",
    "text.color": "#111111",    
    "grid.color": "#E5E5E5",  
    "grid.linewidth": 0.8, 
    "font.family": "monospace", 
    "legend.facecolor": "#FFFFFF", 
    "legend.edgecolor": "#CCCCCC", 
}

def plot_convergence(results, ref_price, save_path=None):
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Monte Carlo Convergence — European Call (Heston, QE scheme)",
                     fontsize=13, y=1.02)

        n_paths = [r["n_paths"] for r in results]
        prices = [r["price"] for r in results]
        ci_low = [r["ci_low"] for r in results]
        ci_high = [r["ci_high"] for r in results]
        errors = [r["error"] for r in results]

        # Panel 1: Price convergence with confidence intervals
        ax1.axhline(ref_price, color="#FFD166", lw=1.5, ls="--",
                    label=f"Heston analytical: {ref_price:.4f}")
        ax1.fill_between(n_paths, ci_low, ci_high, alpha=0.2, color="#2A9D8F",
                         label="95% CI")
        ax1.plot(n_paths, prices, "o-", color="#2A9D8F", lw=2, ms=5,
                 label="MC estimate")
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of paths")
        ax1.set_ylabel("Option price")
        ax1.set_title("Price convergence")
        ax1.legend(fontsize=9)
        ax1.grid(True)

        # Panel 2: Absolute error and 1/sqrt(N) reference
        N = np.array(n_paths, dtype=float)
        ref_rate = errors[0] * np.sqrt(n_paths[0]) / np.sqrt(N)

        ax2.loglog(n_paths, errors, "o-", color="#E63946", lw=2, ms=5,
                   label="|MC price − analytical|")
        ax2.loglog(n_paths, ref_rate, "--", color="#888888", lw=1,
                   label=r"$\propto 1/\sqrt{N}$ (MC rate)")
        ax2.set_xlabel("Number of paths")
        ax2.set_ylabel("Absolute error")
        ax2.set_title("Error convergence")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_discretization_bias(all_results, save_path=None):
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Discretization Bias: European Call vs. Time Steps per Scheme",
                     fontsize=13, y=1.02)

        for scheme in SCHEMES:
            res_list = all_results[scheme]
            n_steps = [r["n_steps"] for r in res_list]
            biases = [r["bias"] for r in res_list]
            abs_biases = [r["abs_bias"] for r in res_list]
            dts = [r["dt"] for r in res_list]
            color = SCHEME_COLORS[scheme]
            label = SCHEME_LABELS[scheme]

            ax1.plot(n_steps, biases, "o-", color=color, lw=2, ms=5, label=label)
            ax2.loglog(dts, abs_biases, "o-", color=color, lw=2, ms=5, label=label)

        ax1.axhline(0, color="#FFD166", lw=1, ls="--", label="Zero bias (ref)")
        ax1.set_xlabel("Number of time steps")
        ax1.set_ylabel("Bias = MC price − Heston analytical")
        ax1.set_title("Signed bias")
        ax1.legend(fontsize=9)
        ax1.grid(True)

        # Reference lines for O(dt) and O(dt^0.5) convergence rates
        dts_ref = np.array(sorted([r["dt"] for r in all_results["euler"]], reverse=True))
        b0 = abs(all_results["euler"][0]["bias"])
        if b0 > 0:
            ax2.loglog(dts_ref, b0 * dts_ref / dts_ref[0], "--",
                       color="#888888", lw=1, alpha=0.7, label=r"$O(\Delta t)$")
        ax2.set_xlabel(r"$\Delta t = T / n\_steps$")
        ax2.set_ylabel("|Bias|")
        ax2.set_title("Absolute bias (log-log)")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_volatility_smile(smile_data, save_path=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        m = smile_data["moneyness"]
        ax.plot(m, smile_data["ivols_heston"] * 100, "-",
                color="#2A9D8F", lw=2.5, label="Heston (semi-analytical)")
        ax.plot(m, smile_data["ivols_mc"] * 100, "o",
                color="#FFD166", ms=4, alpha=0.8, label="Heston (MC, QE)")
        ax.plot(m, smile_data["ivols_bs"] * 100, "--",
                color="#E63946", lw=2, label=f"Black-Scholes (flat vol = {smile_data['flat_vol']*100:.1f}%)")

        ax.set_xlabel("Moneyness K/S₀")
        ax.set_ylabel("Implied volatility (%)")
        ax.set_title("Implied Volatility Smile: Heston vs. Black-Scholes")
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.axvline(1.0, color="#555555", lw=1, ls=":")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_terminal_distribution(sim_result, params=PARAMS, K=K, save_path=None):
    """Compare terminal distribution of S_T: Heston MC vs. BS lognormal."""
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Terminal Distribution of S_T: Heston vs. Black-Scholes",
                     fontsize=13, y=1.02)

        S_T = sim_result["S_T"]
        sigma_bs = np.sqrt(params.theta)

        # Panel 1: Histogram
        ax1.hist(S_T, bins=100, density=True, color="#2A9D8F", alpha=0.7,
                 label="Heston MC")

        # BS lognormal reference
        mu_ln = (np.log(params.S0) + (params.r - 0.5 * sigma_bs**2) * params.T)
        sig_ln = sigma_bs * np.sqrt(params.T)
        x = np.linspace(S_T.min(), S_T.max(), 300)
        pdf_bs = lognorm.pdf(x, s=sig_ln, scale=np.exp(mu_ln))
        ax1.plot(x, pdf_bs, "--", color="#E63946", lw=2, label="BS lognormal")
        ax1.axvline(K, color="#FFD166", lw=1.5, ls=":", label=f"K={K}")
        ax1.set_xlabel("S_T")
        ax1.set_ylabel("Density")
        ax1.set_title("Terminal price distribution")
        ax1.legend(fontsize=9)
        ax1.grid(True)

        # Panel 2: Log-return distribution (check skewness / excess kurtosis)
        log_ST = np.log(S_T / params.S0)
        ax2.hist(log_ST, bins=100, density=True, color="#F4A261", alpha=0.7,
                 label="Heston MC log-returns")
        x_ln = np.linspace(log_ST.min(), log_ST.max(), 300)
        pdf_norm = spnorm.pdf(x_ln, loc=log_ST.mean(), scale=log_ST.std())
        ax2.plot(x_ln, pdf_norm, "--", color="#E63946", lw=2, label="Normal (same mean/std)")
        sk = skew(log_ST)
        kurt = kurtosis(log_ST)
        ax2.set_xlabel("log(S_T / S₀)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Log-return distribution\n(skew={sk:.3f}, excess kurt={kurt:.3f})")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_sample_paths(sim_result, n_display=20, save_path=None):
    """Plot a sample of simulated S and V paths."""
    S_paths = sim_result.get("S_paths")
    V_paths = sim_result.get("V_paths")
    if S_paths is None:
        print("No paths stored. Run with return_paths=True.")
        return

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Sample Heston Paths: Spot and Variance", fontsize=13)

        T = sim_result.get("T", 1.0)
        n_steps = S_paths.shape[1] - 1
        t_grid = np.linspace(0, T, n_steps + 1)

        cmap = plt.cm.plasma
        idx = np.random.choice(len(S_paths), n_display, replace=False)

        for i, path_idx in enumerate(idx):
            color = cmap(i / n_display)
            ax1.plot(t_grid, S_paths[path_idx], lw=0.8, alpha=0.7, color=color)
            ax2.plot(t_grid, np.sqrt(np.maximum(V_paths[path_idx], 0)) * 100,
                     lw=0.8, alpha=0.7, color=color)

        ax1.set_ylabel("Spot price S_t")
        ax1.set_title(f"Spot price ({n_display} paths)")
        ax1.grid(True)

        ax2.axhline(np.sqrt(PARAMS.theta) * 100, color="#FFD166", lw=1.5,
                    ls="--", label=f"Long-run vol = {np.sqrt(PARAMS.theta)*100:.1f}%")
        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("Instantaneous vol (%)")
        ax2.set_title("Instantaneous volatility √V_t")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig

