import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product


def mad_optimization(scenarios_matrix: pd.DataFrame, desired_return: float) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    w = cp.Variable(n_assets)
    
    y = scenarios_matrix.values.T @ w  # shape = (n_scenarios,)
    
    mu = cp.sum(y) / n_scenarios
    
    d = cp.Variable(n_scenarios, nonneg=True)
    
    # vincoli per MAD
    constraints = [
        d >= y - mu,    # d_t >= y_t - mu
        d >= -(y - mu), # d_t >= -(y_t - mu)
        cp.sum(y)/n_scenarios == desired_return, # vincolo sul rendimento medio
        cp.sum(w) == 1, # portafoglio completamente investito
        w >= 0          
    ]
    
    # problema: minimizza MAD
    mad = cp.sum(d)/n_scenarios
    problem = cp.Problem(cp.Minimize(mad), constraints)
    
    problem.solve()

    if problem.status == "optimal":
        optimal_w = w.value
        portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
        expected_return = portfolio_returns_opt.mean()
        mad_value = np.mean(np.abs(portfolio_returns_opt - expected_return))
        
        print("Expected Return (mean):", expected_return)
        print("MAD:", mad_value)
        return pd.Series(w.value, index=scenarios_matrix.index)

    return "unfeasible"

def semi_mad_optimization(scenarios_matrix: pd.DataFrame, desired_return: float) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    w = cp.Variable(n_assets)
    
    y = scenarios_matrix.values.T @ w  # shape = (n_scenarios,)
    
    mu = cp.sum(y) / n_scenarios
    
    d = cp.Variable(n_scenarios, nonneg=True)
    
    # vincoli per MAD
    constraints = [
        d >= y - mu,    # d_t >= y_t - mu
        cp.sum(y)/n_scenarios == desired_return, # vincolo sul rendimento medio
        cp.sum(w) == 1, # portafoglio completamente investito
        w >= 0          
    ]
    
    # problema: minimizza MAD
    mad = cp.sum(d)/n_scenarios
    problem = cp.Problem(cp.Minimize(mad), constraints)
    
    problem.solve()

    if problem.status == "optimal":
        optimal_w = w.value
        portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
        expected_return = portfolio_returns_opt.mean()
        mad_value = np.mean(np.abs(portfolio_returns_opt - expected_return))
        
        print("Expected Return (mean):", expected_return)
        print("MAD:", mad_value)
        return pd.Series(w.value, index=scenarios_matrix.index)

    return "unfeasible"

def gini_optimization(scenarios_matrix: pd.DataFrame, desired_return: float) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    # variabili: pesi del portafoglio
    w = cp.Variable(n_assets)
    
    # portafoglio simulato in tutti gli scenari
    y = scenarios_matrix.values.T @ w  # shape = (n_scenarios,)
    
    # probabilità uniforme degli scenari
    p = 1 / n_scenarios
    
    # variabili d_{t',t''} >= 0 per ogni coppia t' != t''
    d = {}
    for t1, t2 in product(range(n_scenarios), repeat=2):
        if t1 != t2:
            d[(t1,t2)] = cp.Variable(nonneg=True)
    
    # vincoli sulle differenze
    constraints = []
    for (t1,t2), d_var in d.items():
        constraints.append(d_var >= y[t1] - y[t2])
    
    # vincolo rendimento medio desiderato
    mean_return = cp.sum(y) / n_scenarios
    constraints.append(mean_return == desired_return)
    
    # portafoglio completamente investito e long-only
    constraints.append(cp.sum(w) == 1)
    constraints.append(w >= 0)
    
    # obiettivo: somma pesata su tutte le coppie
    objective = cp.sum([p*p*d_var for d_var in d.values()])
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    if problem.status == "optimal":
        optimal_w = w.value
        portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
        expected_return = portfolio_returns_opt.mean()
        mad_value = np.mean(np.abs(portfolio_returns_opt - expected_return))
        
        print("Expected Return (mean):", expected_return)
        print("MAD:", mad_value)
        return pd.Series(w.value, index=scenarios_matrix.index)

    return "unfeasible"

def minmax_optimization(scenarios_matrix: pd.DataFrame, desired_return: float) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    # Variabili di decisione
    w = cp.Variable(n_assets) # Vettore x nell'immagine
    y_min = cp.Variable()      # Variabile scalare y da massimizzare
    
    # Calcolo dei rendimenti per ogni scenario (r_jt * x_j)
    # portfolio_returns ha forma (n_scenarios,)
    portfolio_returns = scenarios_matrix.values.T @ w 
    
    # Rendimento atteso (mu)
    mu = cp.sum(portfolio_returns) / n_scenarios
    
    # Vincoli basati sull'immagine:
    constraints = [
        # 1. Somma(r_jt * x_j) >= y per ogni t=1..T
        portfolio_returns >= y_min,
        
        # 2. mu >= mu_0 (rendimento desiderato)
        mu >= desired_return,
        
        # 3. x in Q (vincoli standard di portafoglio: somma 1 e no-short)
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Problema: Massimizzare il rendimento minimo (y)
    problem = cp.Problem(cp.Maximize(y_min), constraints)
    
    problem.solve()

    if problem.status in ["optimal", "feasible"]:
        optimal_w = w.value
        portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
        expected_return = portfolio_returns_opt.mean()
        min_return = np.min(portfolio_returns_opt)
        
        print("Expected Return (mean):", expected_return)
        print("Worst-case Return (y):", min_return)
        
        return pd.Series(optimal_w, index=scenarios_matrix.index)

    return "unfeasible"

def var_optimization(scenarios_matrix: pd.DataFrame, desired_return: float, beta: float=0.05) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    # Variabili di decisione
    w = cp.Variable(n_assets)       # Vettore x (pesi portafoglio)
    y = cp.Variable()                # Il valore del VaR da massimizzare
    z = cp.Variable(n_scenarios, boolean=True) # Variabile binaria (0 o 1)
    
    # Parametro M (un numero "abbastanza grande")
    # Deve essere maggiore della differenza tra il rendimento massimo e minimo possibile
    M = 100
    
    # Probabilità degli scenari (assumiamo equiprobabili se non specificato)
    p = 1.0 / n_scenarios
    
    # Rendimenti per scenario
    portfolio_returns = scenarios_matrix.values.T @ w
    
    # Rendimento medio
    mu = cp.sum(portfolio_returns) / n_scenarios

    # Vincoli dell'immagine
    constraints = [
        # 1. Rendimento scenario t >= y - M * z_t
        # Se z_t = 0, il rendimento deve essere >= y. 
        # Se z_t = 1, il vincolo è reso inattivo dal valore grande M.
        portfolio_returns >= y - M * z,
        
        # 2. Somma delle probabilità degli scenari scartati <= beta
        # Qui beta rappresenta il livello di confidenza (es. 0.05)
        cp.sum(p * z) <= beta,
        
        # 3. Rendimento atteso >= mu_0
        mu == desired_return,
        
        # 4. x in Q (budget e no-short)
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Problema: Massimizzare il livello di rendimento y (VaR)
    # Nota: massimizzare y equivale a minimizzare il VaR (inteso come perdita)
    problem = cp.Problem(cp.Maximize(y), constraints)
    
    # Risoluzione (richiede un solver MILP come GLPK_MI, CBC o GUROBI)
    problem.solve()

    if problem.status in ["optimal", "feasible"]:
        optimal_w = w.value
        print(f"Optimal VaR at {beta*100}% level: {y.value:.4f}")
        print(f"Expected Return: {mu.value:.4f}")
        print(f"Scenarios ignored: {int(np.sum(z.value))}")
        
        return pd.Series(optimal_w, index=scenarios_matrix.index)

    return "unfeasible"

def var_rev_optimization(scenarios_matrix: pd.DataFrame, min_var_return: float, beta: float=0.05) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    # Variabili di decisione
    w = cp.Variable(n_assets)       
    z = cp.Variable(n_scenarios, boolean=True) # Variabile binaria per ignorare gli scenari
    
    # Parametro M (Big M)
    # Deve essere abbastanza grande da "disattivare" il vincolo per gli scenari z_t = 1
    M = 100 
    
    # Probabilità degli scenari (equiprobabili)
    p = 1.0 / n_scenarios
    
    # Rendimenti per scenario: y_t = sum(r_jt * x_j)
    portfolio_returns = scenarios_matrix.values.T @ w
    
    # Funzione Obiettivo: Massimizzare il rendimento medio (mu)
    mu = cp.sum(portfolio_returns) / n_scenarios
    objective = cp.Maximize(mu)

    # Vincoli basati sulla logica dell'immagine
    constraints = [
        # 1. Il rendimento deve essere >= min_var_return, eccetto per gli scenari "scartati" (z_t=1)
        portfolio_returns >= min_var_return - M * z,
        
        # 2. La somma delle probabilità degli scenari scartati non deve superare beta
        cp.sum(p * z) <= beta,
        
        # 3. Vincoli di portafoglio (x in Q)
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Risoluzione MILP
    problem = cp.Problem(objective, constraints)
    problem.solve() # Nota: potrebbe essere necessario specificare solver=cp.GLPK_MI

    if problem.status in ["optimal", "feasible"]:
        optimal_w = w.value
        actual_mu = mu.value
        ignored = int(np.sum(z.value))
        
        print(f"--- Ottimizzazione Completata ---")
        print(f"Rendimento medio massimizzato: {actual_mu:.4f}")
        print(f"Vincolo VaR (min return al {100*(1-beta)}%): {min_var_return:.4f}")
        print(f"Scenari ignorati (sotto la soglia): {ignored} su {n_scenarios}")
        
        return pd.Series(optimal_w, index=scenarios_matrix.index)

    return "unfeasible"

def cvar_optimization(scenarios_matrix: pd.DataFrame, desired_return: float, beta: float) -> pd.Series:
    n_assets, n_scenarios = scenarios_matrix.shape
    
    # Variabili di decisione
    w = cp.Variable(n_assets)        # Vettore x (pesi portafoglio)
    eta = cp.Variable()              # Variabile ausiliaria per il calcolo del VaR
    d_minus = cp.Variable(n_scenarios) # Variabile d_t^- (scostamenti negativi)
    
    # Probabilità degli scenari (assumiamo equiprobabili 1/T)
    p = 1.0 / n_scenarios
    
    # Calcolo dei rendimenti per scenario (y_t = sum(r_jt * x_j))
    portfolio_returns = scenarios_matrix.values.T @ w
    
    # Rendimento medio (mu)
    mu = cp.sum(portfolio_returns) / n_scenarios

    # Vincoli basati sulla formulazione CVaR dell'immagine
    constraints = [
        # d_t^- >= eta - y_t
        d_minus >= eta - portfolio_returns,
        
        # d_t^- >= 0
        d_minus >= 0,
        
        # Rendimento atteso >= mu_0
        mu >= desired_return,
        
        # x in Q (budget e no-short)
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Funzione Obiettivo: max (eta - 1/beta * sum(p_t * d_t^-))
    # Nota: Massimizzare questo termine equivale a minimizzare il rischio di coda
    cvar_objective = eta - (1.0 / beta) * cp.sum(p * d_minus)
    
    problem = cp.Problem(cp.Maximize(cvar_objective), constraints)
    
    # CVaR è un problema convesso, quindi non serve un solver MILP (basta ECOS o SCS)
    problem.solve()

    if problem.status in ["optimal", "feasible"]:
        optimal_w = w.value
        # Calcolo dei valori per log/stampa
        portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
        expected_return = portfolio_returns_opt.mean()
        
        print(f"Optimal Expected CVaR (at {beta*100}%): {problem.value:.4f}")
        print(f"Portfolio Expected Return: {expected_return:.4f}")
        print(f"Optimal Eta (VaR proxy): {eta.value:.4f}")
        
        return pd.Series(optimal_w, index=scenarios_matrix.index)

    return "unfeasible"

def approx_markowitz_optimization(
    scenarios_matrix: pd.DataFrame,
    covariance_matrix: np.array,
    desired_return: float,
    n_intervals: int = 3
) -> pd.Series:

    # ------------------------------------------------------------------ #
    # FIX 1 — LDLᵀ corretta: estrae L triangolare inferiore e d diagonale
    #          L'originale non estraeva L ma sovrascriveva C in modo errato,
    #          perdendo la struttura necessaria per il cambio di variabile y=Lᵀw
    # ------------------------------------------------------------------ #
    def ldlt_decomposition(C_input):
        C = np.array(C_input, dtype=float)
        n = C.shape[0]
        L = np.eye(n)
        d = np.zeros(n)

        for p in range(n):
            if abs(C[p, p]) < 1e-14:
                raise ValueError(
                    f"Pivot quasi-nullo in posizione {p}: "
                    "la matrice potrebbe non essere definita positiva."
                )
            d[p] = C[p, p]
            for i in range(p + 1, n):
                L[i, p] = C[i, p] / d[p]
            for i in range(p + 1, n):
                for j in range(p + 1, n):
                    C[i, j] -= L[i, p] * d[p] * L[j, p]

        return L, d

    L, d_vector = ldlt_decomposition(covariance_matrix)
    n = len(d_vector)

    # ------------------------------------------------------------------ #
    # FIX 2 — bound superiori di yᵢ = (Lᵀw)ᵢ basati sui dati
    #          L'originale usava costante arbitraria 10 per tutti gli asset
    # ------------------------------------------------------------------ #
    sigma_diag = np.diag(np.array(covariance_matrix, dtype=float))
    max_y = []
    for i in range(n):
        # Lᵀ[i,:] = L[:,i] — bound sul simplesso = norma-1 della riga
        row_lt = L[:, i]
        bound = max(np.sum(np.abs(row_lt)), 5.0 * np.sqrt(sigma_diag[i]))
        max_y.append(bound)

    # ------------------------------------------------------------------ #
    # FIX 3 — pendenze della spezzata corrette
    #          La secante di y² su [t_{j-1}, t_j] ha pendenza t_{j-1}+t_j,
    #          NON una media pesata come nell'originale.
    #          Le pendenze sono crescenti (convessità) → LP corretto.
    # ------------------------------------------------------------------ #
    all_slopes = []
    all_deltas = []
    for i in range(n):
        delta_i = max_y[i] / n_intervals
        slopes_i = np.zeros(n_intervals)
        for j in range(1, n_intervals + 1):
            t_prev = delta_i * (j - 1)
            t_curr = delta_i * j
            slopes_i[j - 1] = t_prev + t_curr   # = (t_curr²-t_prev²)/delta_i
        all_slopes.append(slopes_i)
        all_deltas.append(delta_i)

    # ------------------------------------------------------------------ #
    # FIX 4 + FIX 5 — problema LP con variabili ausiliarie per la spezzata
    #
    #  L'originale calcolava un unico scalare a_i e lo proiettava su w,
    #  perdendo completamente la struttura per segmento.
    #  La versione corretta introduce:
    #    y  = Lᵀ w                 (cambio di variabile LDLᵀ)
    #    s[i,j] ∈ [0, delta_i]     (quota del j-mo segmento di yᵢ)
    #    yᵢ = Σⱼ s[i,j]            (ricostruzione di yᵢ)
    #  Obiettivo: Σᵢ dᵢ · slopeᵢ · s[i,:]  ≈  wᵀΣw  (lineare!)
    # ------------------------------------------------------------------ #
    def linear_coefficient_optimization(
        scenarios_matrix: pd.DataFrame,
        desired_return: float,
    ) -> pd.Series:

        n_assets = scenarios_matrix.shape[0]
        n_scenarios = scenarios_matrix.shape[1]

        # Variabili di decisione
        w = cp.Variable(n_assets, nonneg=True)
        y = cp.Variable(n)                               # y = Lᵀ w
        s = cp.Variable((n, n_intervals), nonneg=True)   # segmenti spezzata

        # Rendimento atteso
        portfolio_returns = scenarios_matrix.values.T @ w   # (n_scenarios,)
        mu = cp.sum(portfolio_returns) / n_scenarios

        # Vincoli
        constraints = [
            mu >=        desired_return,
            cp.sum(w) == 1,
            w >= 0,
            y == L.T @ w,                         # cambio variabile LDLᵀ
        ]
        for i in range(n):
            constraints += [
                cp.sum(s[i, :]) == y[i],           # yᵢ = Σⱼ sᵢⱼ
                s[i, :] <= all_deltas[i],           # sᵢⱼ ∈ [0, Δᵢ]
            ]

        # Funzione obiettivo lineare ≈ varianza
        risk_approx = sum(
            d_vector[i] * (all_slopes[i] @ s[i, :])
            for i in range(n)
        )

        problem = cp.Problem(cp.Minimize(risk_approx), constraints)
        problem.solve(cp.GUROBI)

        # FIX 6 — "feasible" non è uno status valido in cvxpy
        if problem.status in ["optimal", "optimal_inaccurate"]:
            optimal_w = w.value
            portfolio_returns_opt = scenarios_matrix.values.T @ optimal_w
            expected_return = portfolio_returns_opt.mean()

            final_risk_value = float(risk_approx.value)
            true_variance = float(optimal_w.T @ covariance_matrix @ optimal_w)

            print("Expected Return (mean):", expected_return)
            print("Minimized Risk Value (Linear approx):", final_risk_value)
            print("True Variance:", true_variance)
            print(f"Errore relativo: {abs(true_variance - final_risk_value) / true_variance:.2%}")

            return pd.Series(optimal_w, index=scenarios_matrix.index)

        return "unfeasible"

    return linear_coefficient_optimization(scenarios_matrix, desired_return)

def mean_variance_optimization(expected_returns: pd.Series, cov_matrix: pd.DataFrame, desired_return: float) -> pd.Series:
    n_assets = len(expected_returns)
    mu = expected_returns.values
    Sigma = cov_matrix.values
    
    # Variabile di decisione (vettore pesi w)
    w = cp.Variable(n_assets)
    
    # Funzione Obiettivo: Minimizzare la varianza del portafoglio (1/2 * w' * Sigma * w)
    # cp.quad_form è la funzione specifica per le forme quadratiche in CVXPY
    variance_op = 0.5 * cp.quad_form(w, Sigma)
    objective = cp.Minimize(variance_op)
    
    # Vincoli basati sul modello classico (image_7b84a2.png)
    constraints = [
        w @ mu == desired_return,  # Vincolo sul rendimento atteso: w'mu = E(R)
        cp.sum(w) == 1,           # Vincolo di budget: w'1 = 1
        w >= 0                    # Vincolo di no-short selling (standard Q)
    ]
    
    # Risoluzione del problema quadratico
    problem = cp.Problem(objective, constraints)
    problem.solve(cp.GUROBI)

    if problem.status in ["optimal", "feasible"]:
        optimal_w = w.value
        
        # Calcolo dei parametri di output
        # La varianza è w' * Sigma * w
        portfolio_variance = optimal_w.T @ Sigma @ optimal_w
        # Il rendimento è w' * mu
        portfolio_return = optimal_w.T @ mu
        
        print(f"--- Markowitz Numerical Optimization ---")
        print(f"Expected Return (target): {portfolio_return:.4f}")
        print(f"Portfolio Variance: {portfolio_variance:.6f}")
        print(f"Portfolio Volatility (Std Dev): {np.sqrt(portfolio_variance):.4f}")
        
        return pd.Series(optimal_w, index=expected_returns.index)

    return "unfeasible"


if __name__ == "__main__":
    print(cp.installed_solvers())