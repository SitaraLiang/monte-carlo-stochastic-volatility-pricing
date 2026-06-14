import numpy as np
from black_scholes import bs_implied_vol


def _mc_price(payoffs, r, T):
    """ 
    Calculate the final price using payoffs

    payoffs: shape (n_paths,) 
    r: risk free rate
    T: time to maturity (in years)
    """
    df = np.exp(-r * T) # discount factor
    discounted = df * payoffs

    # monte carlo price
    price = discounted.mean() # fair market price, average of all the simulated present-day payout
    std_err = discounted.std() / np.sqrt(len(discounted)) # fluctuation 
    return {
        "price": price,
        "std_error": std_err,
        "ci_low": price - 1.96 * std_err, # 95% confidence interval
        "ci_high": price + 1.96 * std_err,
        "n_paths": len(payoffs),
    }


def european_call(sim_result, K, r):
    """
    European options only care about the asset price on the very last day (expiration)

    sim_result: dict from heston_mc.simulate_heston()
    K: strike price
    r: risk free rate (same as used in simulation)
    """
    S_T = sim_result["S_T"] # final prices
    T = sim_result["dt"] * (sim_result["S_T"].shape[0] > 0)  # placeholder
    # Recover T from params if stored, otherwise from dt * n_steps
    # We store it in the result dict when calling from analysis.py
    T = sim_result.get("T", 1.0)
    payoffs = np.maximum(S_T - K, 0.0) # profit if the stock price above the strike price
    return _mc_price(payoffs, r, T)


def european_put(sim_result, K, r):
    S_T = sim_result["S_T"]
    T = sim_result.get("T", 1.0)
    payoffs = np.maximum(K - S_T, 0.0) # profit if the stock price falls below the strike price
    return _mc_price(payoffs, r, T)


def asian_call(sim_result, K, r):
    S_paths = sim_result.get("S_paths")
    if S_paths is None:
        raise ValueError("Asian options require full paths"
                        "Set return_paths=True in simulate_heston()")
    T = sim_result.get("T", 1.0)
    A_T = S_paths[:, 1:].mean(axis=1)  # exclude t=0 (starting price), calculates the average price for each simulated path
    payoffs = np.maximum(A_T - K, 0.0)
    return _mc_price(payoffs, r, T)


def asian_put(sim_result, K, r):
    S_paths = sim_result.get("S_paths")
    if S_paths is None:
        raise ValueError("Asian options require full paths"
                        "Set return_paths=True in simulate_heston()")
    T = sim_result.get("T", 1.0)
    A_T = S_paths[:, 1:].mean(axis=1)
    payoffs = np.maximum(K - A_T, 0.0)
    return _mc_price(payoffs, r, T)


def barrier_option(sim_result, K, r, barrier, barrier_type="up-and-out",
                   option_type="call", rebate=0.0):
    """
    Barrier options active or deactivate based on whether the stock price ever touches a specific "trigger" line 
    (the barrier) during its lifetime

    barrier: barrier level H
    barrier_type: "up-and-out", "up-and-in", "down-and-out", "down-and-in"
    rebate: amount paid if barrier is hit (for knock-out options)
    """
    S_paths = sim_result.get("S_paths")
    if S_paths is None:
        raise ValueError("Barrier options require full paths"
                        "Set return_paths=True in simulate_heston()")

    T = sim_result.get("T", 1.0)
    S_T = S_paths[:, -1]

    if "up" in barrier_type:
        crossed = np.any(S_paths >= barrier, axis=1)
    else:  # down
        crossed = np.any(S_paths <= barrier, axis=1)

    if option_type == "call":
        terminal_payoff = np.maximum(S_T - K, 0.0)
    else:
        terminal_payoff = np.maximum(K - S_T, 0.0)

    # Apply barrier condition
    if "out" in barrier_type:
        # Knock-out: option lives if barrier NOT crossed
        payoffs = np.where(crossed, rebate, terminal_payoff)
    else:
        # Knock-in: option lives if barrier WAS crossed
        payoffs = np.where(crossed, terminal_payoff, rebate)

    result = _mc_price(payoffs, r, T)
    result["barrier"] = barrier
    result["barrier_type"] = barrier_type
    result["fraction_crossed"] = crossed.mean() # percentage of simulations that hit the barrier
    return result


def up_and_out_call(sim_result, K, r, barrier, rebate=0.0):
    # Knock out call: option is deactivated if the stock price ever goes above the barrier
    return barrier_option(sim_result, K, r, barrier, "up-and-out", "call", rebate)

def down_and_out_put(sim_result, K, r, barrier, rebate=0.0):
    # Knock out put: option is deactivated if the stock price ever goes below the barrier
    return barrier_option(sim_result, K, r, barrier, "down-and-out", "put", rebate)

def up_and_in_call(sim_result, K, r, barrier, rebate=0.0):
    # Knock in call: option is activated only if the stock price ever goes above the barrier
    return barrier_option(sim_result, K, r, barrier, "up-and-in", "call", rebate)

def down_and_in_put(sim_result, K, r, barrier, rebate=0.0):
    # Knock in put: option is activated only if the stock price ever goes below the barrier
    return barrier_option(sim_result, K, r, barrier, "down-and-in", "put", rebate)



def variance_swap(sim_result, K_var=None, notional=1.0):
    """
    Forward contracts on future realized variance of an asset's returns

    K_var: variance strike (annualized). If None, returns the fair strike
    notional: 1 (per unit of variance)

    Returns
    dict with keys:
        fair_variance_strike: K_var at which swap has zero value
        price: MC price of the swap for given K_var
        realized_var_mean: mean realized variance across paths
        realized_var_std: std of realized variance across paths
    """
    S_paths = sim_result.get("S_paths")
    if S_paths is None:
        raise ValueError("Variance swaps require full paths"
                        "Set return_paths=True in simulate_heston()")

    T = sim_result.get("T", 1.0)

    # the percentage change of the stock at every step
    log_returns = np.diff(np.log(S_paths), axis=1)  # shape (n_paths, n_steps)
    # realized variance per path (annualized sum of squared log returns)
    realized_var = (1.0 / T) * np.sum(log_returns**2, axis=1) # shape (n_paths,)

    # average variance across all paths
    # entry price where neither the buyer nor seller has an immediate advantage
    fair_strike = realized_var.mean()

    result = {
        "fair_variance_strike": fair_strike,
        "fair_vol_strike": np.sqrt(fair_strike), # annualized volatility
        "realized_var_mean": fair_strike,
        "realized_var_std": realized_var.std(),
        "realized_var_paths": realized_var,
    }

    if K_var is not None:
        payoffs = notional * (realized_var - K_var)
        pricing = _mc_price(payoffs, r=0.0, T=T)  # var swaps discounted at 0
        result.update(pricing)
        result["K_var"] = K_var

    return result


def mc_implied_vol_smile(sim_result, strikes, r, option_type="call"):
    """
    Compute implied volatility smile from MC-priced options

    strikes: array of strike prices, shape (n_strikes,)
    """
    T = sim_result.get("T", 1.0)
    ivols = [] # implied volatilities
    prices = [] # option prices

    for K in strikes:
        if option_type == "call":
            res = european_call(sim_result, K, r)
        else:
            res = european_put(sim_result, K, r)

        price = res["price"]
        prices.append(price)

        S0 = sim_result.get("S0", 100.0)
        iv = bs_implied_vol(price, S0, K, r, T, option_type)
        ivols.append(iv)

    return np.array(ivols), np.array(prices)
