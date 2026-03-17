import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def analyze_returns(df: pd.DataFrame, confidence_level: float = 0.95, annualized: bool=True, rolling: int=252) -> pd.DataFrame:
    """
    Calcola statistiche descrittive per ogni colonna (ETF) del dataframe.
    
    Parametri:
        df : pd.DataFrame
            DataFrame con rendimenti logaritmici (righe = giorni, colonne = ETF)
        confidence_level : float
            Livello di confidenza per VaR e CVaR (default 0.95)
        annualized : bool
            Se True, restituisce statistiche annualizzate
            
    Ritorna:
        pd.DataFrame con statistiche per ETF:
        - mean
        - std
        - skewness
        - kurtosis
        - VaR
        - CVaR
    """

    assert not annualized or rolling > 0

    stats = {}
    alpha = 1 - confidence_level
    
    for col in df.columns:
        if annualized:
            rolling_series = df[col].dropna().rolling(rolling).sum().dropna()

        series = df[col].dropna()
        
        mu = series.mean()
        sigma = series.std()
        
        skewness_val = skew(rolling_series if annualized else series)
        kurtosis_val = kurtosis(rolling_series if annualized else series)  # Fisher=True per sottrarre 3
        
        var_val = np.quantile(rolling_series if annualized else series, alpha)
        
        cvar_series = rolling_series if annualized else series
        cvar_val = cvar_series[cvar_series <= var_val].mean()
        
        stats[col] = {
            "mean": mu * rolling if annualized else mu,
            "std": sigma * np.sqrt(rolling) if annualized else sigma,
            "skewness": skewness_val,
            "kurtosis": kurtosis_val,
            f"VaR_{int(confidence_level*100)}": var_val,
            f"CVaR_{int(confidence_level*100)}": cvar_val
        }
    
    stats_df = pd.DataFrame(stats).T  # Trasponi per avere ETF come righe
    return stats_df

def covariance_matrix(df: pd.DataFrame, annualized: bool=True) -> pd.DataFrame:
    """
    Calcola la matrice di covarianza dei rendimenti logaritmici.
    
    Parametri:
        df : pd.DataFrame
            DataFrame con rendimenti logaritmici (righe = giorni, colonne = ETF)
        annualized : bool
            Se True, restituisce la matrice di covarianza annualizzata
            
    Ritorna:
        pd.DataFrame matrice di covarianza
    """
    return df.cov() * 252 if annualized else df.cov()

