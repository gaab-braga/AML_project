"""
Information Value (IV) Calculator Module

This module provides functions for calculating Information Value (IV) and Weight of Evidence (WOE)
for feature selection and risk modeling in fraud detection and credit scoring applications.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


def calculate_iv(df: pd.DataFrame,
                 target_col: str,
                 bins: int = 10,
                 max_iv: float = 10.0,
                 min_samples: int = 10,
                 max_unique_values: int = 1000) -> pd.DataFrame:
    """
    Calculate Information Value (IV) based on Weight of Evidence (WOE) with protections against overfitting.

    Information Value measures the predictive power of a feature for binary classification tasks.
    Higher IV values indicate stronger predictive relationships with the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and target variable
    target_col : str
        Name of the target column (binary: 0/1)
    bins : int, default=10
        Number of bins for discretizing continuous numeric variables
    max_iv : float, default=10.0
        Maximum IV value to prevent overfitting (capped at this value)
    min_samples : int, default=10
        Minimum samples per category to avoid overfitting
    max_unique_values : int, default=1000
        Maximum unique values allowed for a variable (filters out high-cardinality features)

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['variable', 'IV', 'unique_values', 'is_suspect'] sorted by IV descending.
        IV interpretation:
        - < 0.02: Not predictive
        - 0.02-0.1: Weak
        - 0.1-0.3: Medium
        - 0.3-0.5: Strong
        - > 0.5: Very strong

    Examples
    --------
    >>> import pandas as pd
    >>> from src.features.iv_calculator import calculate_iv
    >>>
    >>> # Sample data
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5],
    ...     'feature2': ['A', 'B', 'A', 'B', 'A'],
    ...     'target': [0, 1, 0, 1, 0]
    ... })
    >>> iv_results = calculate_iv(df, 'target')
    >>> print(iv_results.head())
    """
    iv_list = []
    skipped_reasons = {'high_cardinality': [], 'constant': [], 'errors': []}

    # Total events (1) and non-events (0)
    total_events = (df[target_col] == 1).sum()
    total_non_events = (df[target_col] == 0).sum()

    # Check if there are enough cases for reliable IV calculation
    if total_events < 5 or total_non_events < 5:
        print(f"Warning: Insufficient cases for reliable IV calculation "
              f"(events={total_events}, non-events={total_non_events})")

    for col in df.columns:
        if col == target_col:
            continue

        try:
            x = df[col].copy()
            unique_values = x.nunique()

            # Skip variables with too many unique values (potential leakage/IDs)
            if unique_values > max_unique_values:
                skipped_reasons['high_cardinality'].append(col)
                continue

            # Skip constant variables
            if unique_values <= 1:
                skipped_reasons['constant'].append(col)
                continue

            # Create bins if continuous numeric
            if pd.api.types.is_numeric_dtype(x) and unique_values > bins:
                try:
                    x_binned = pd.qcut(x, bins, duplicates='drop')
                except ValueError:
                    # Fallback to equal-width bins if qcut fails
                    x_binned = pd.cut(x, bins, duplicates='drop')
            else:
                # For categorical variables or variables with few unique values
                # Keep NaN as separate category
                x_binned = x.astype(str)
                x_binned = x_binned.replace('nan', '**MISSING**')

            # Contingency table
            df_cut = pd.crosstab(x_binned, df[target_col])

            # Skip variable if missing columns (0 or 1)
            if 0 not in df_cut.columns or 1 not in df_cut.columns:
                continue

            # Filter categories with few samples
            df_cut = df_cut[(df_cut[0] + df_cut[1]) >= min_samples]

            if len(df_cut) == 0:
                continue

            iv = 0.0
            # Calculate WOE and accumulate IV
            for idx, row in df_cut.iterrows():
                count_non, count_ev = row.get(0, 0), row.get(1, 0)

                # Laplace smoothing to avoid zero/infinite values
                ev = count_ev + 0.5
                ne = count_non + 0.5
                pct_ev = ev / (total_events + 1)
                pct_ne = ne / (total_non_events + 1)

                # Calculate WOE with protection against extreme values
                if pct_ev > 0 and pct_ne > 0:
                    woe = np.log(pct_ev / pct_ne)
                    # Limit WOE to avoid extreme values
                    woe = np.clip(woe, -5, 5)
                    iv += (pct_ev - pct_ne) * woe

            # Limit maximum IV to prevent overfitting
            iv_capped = min(iv, max_iv)

            # Flag suspicious variables (potential leakage)
            is_suspect = (iv_capped == max_iv) or (unique_values > max_unique_values * 0.5)

            iv_list.append({
                'variable': col,
                'IV': iv_capped,
                'unique_values': unique_values,
                'is_suspect': is_suspect
            })

        except Exception as e:
            skipped_reasons['errors'].append(col)
            continue

    # Print a clean summary of skipped columns
    for reason, cols in skipped_reasons.items():
        if cols:
            reason_text = reason.replace('_', ' ')
            # Truncate list for cleaner output if it's too long
            cols_to_show = cols[:3]
            ellipsis = "..." if len(cols) > 3 else ""
            print(f"Info: Skipped {len(cols)} {reason_text} columns (e.g., {cols_to_show}){ellipsis}")

    return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)


def interpret_iv(iv_value: float) -> str:
    """
    Interpret Information Value based on standard thresholds.

    Parameters
    ----------
    iv_value : float
        Information Value to interpret

    Returns
    -------
    str
        Interpretation category
    """
    if iv_value < 0.02:
        return "Not predictive"
    elif iv_value < 0.1:
        return "Weak"
    elif iv_value < 0.3:
        return "Medium"
    elif iv_value < 0.5:
        return "Strong"
    else:
        return "Very strong"


def calculate_woe(df: pd.DataFrame,
                  target_col: str,
                  feature_col: str,
                  bins: int = 10) -> pd.DataFrame:
    """
    Calculate Weight of Evidence (WOE) for a specific feature.

    WOE measures the strength of the relationship between a feature and the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature and target
    target_col : str
        Name of the target column
    feature_col : str
        Name of the feature column
    bins : int, default=10
        Number of bins for continuous variables

    Returns
    -------
    pandas.DataFrame
        DataFrame with WOE values for each category/bin
    """
    x = df[feature_col].copy()

    # Create bins if continuous
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x_binned = pd.qcut(x, bins, duplicates='drop')
    else:
        x_binned = x.astype(str).replace('nan', '**NA**')

    # Contingency table
    df_cut = pd.crosstab(x_binned, df[target_col])

    # Calculate WOE
    total_events = (df[target_col] == 1).sum()
    total_non_events = (df[target_col] == 0).sum()

    woe_dict = {}
    for idx, row in df_cut.iterrows():
        count_non, count_ev = row.get(0, 0), row.get(1, 0)

        # Laplace smoothing
        ev = count_ev + 0.5
        ne = count_non + 0.5
        pct_ev = ev / (total_events + 1)
        pct_ne = ne / (total_non_events + 1)

        if pct_ev > 0 and pct_ne > 0:
            woe = np.log(pct_ev / pct_ne)
            woe = np.clip(woe, -5, 5)
        else:
            woe = 0

        woe_dict[idx] = woe

    return pd.DataFrame.from_dict(woe_dict, orient='index', columns=['WOE'])


def get_predictive_features(iv_results: pd.DataFrame,
                           min_iv: float = 0.02,
                           exclude_suspect: bool = True) -> pd.DataFrame:
    """
    Filter features based on minimum Information Value threshold.

    Parameters
    ----------
    iv_results : pandas.DataFrame
        Results from calculate_iv function
    min_iv : float, default=0.02
        Minimum IV threshold for predictive features
    exclude_suspect : bool, default=True
        Whether to exclude variables flagged as suspicious

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with predictive features only
    """
    filtered = iv_results[iv_results['IV'] >= min_iv].copy()
    if exclude_suspect and 'is_suspect' in filtered.columns:
        filtered = filtered[~filtered['is_suspect']]
    return filtered


def check_monotonicity(df: pd.DataFrame,
                      target_col: str,
                      feature_col: str,
                      bins: int = 10) -> bool:
    """
    Check if Weight of Evidence (WOE) is monotonic for a numeric feature.

    Monotonic WOE is desirable for credit scoring models.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature and target
    target_col : str
        Name of the target column
    feature_col : str
        Name of the numeric feature column
    bins : int, default=10
        Number of bins for discretization

    Returns
    -------
    bool
        True if WOE is monotonic, False otherwise
    """
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        return False

    try:
        woe_df = calculate_woe(df, target_col, feature_col, bins)
        woe_values = woe_df['WOE'].values

        # Check if monotonic increasing or decreasing
        return (np.all(np.diff(woe_values) >= 0) or  # increasing
                np.all(np.diff(woe_values) <= 0))    # decreasing
    except:
        return False
    
# Função auxiliar para calcular IV por bin de uma variável numérica
def calculate_iv_by_bins(df, target_col, feature_col, n_bins=10):
    """
    Calcula IV para cada bin de uma variável numérica
    """
    x = df[feature_col].copy()
    y = df[target_col]

    # Criar bins usando quantis
    try:
        bins = pd.qcut(x, n_bins, duplicates='drop', retbins=True)[1]
        x_binned = pd.cut(x, bins=bins, include_lowest=True)
    except:
        # Fallback para bins de largura igual
        x_binned = pd.cut(x, bins=n_bins, include_lowest=True)

    # Calcular estatísticas por bin
    bin_stats = []
    total_events = (y == 1).sum()
    total_non_events = (y == 0).sum()

    for bin_name, group in df.groupby(x_binned):
        if pd.isna(bin_name):
            continue

        bin_events = (group[target_col] == 1).sum()
        bin_non_events = (group[target_col] == 0).sum()
        bin_total = len(group)

        if bin_total < 5:  # Pular bins com poucos casos
            continue

        # Calcular WOE e IV para este bin
        pct_events = (bin_events + 0.5) / (total_events + 1)
        pct_non_events = (bin_non_events + 0.5) / (total_non_events + 1)

        if pct_events > 0 and pct_non_events > 0:
            woe = np.log(pct_events / pct_non_events)
            woe = np.clip(woe, -5, 5)
            iv_bin = (pct_events - pct_non_events) * woe
        else:
            woe = 0
            iv_bin = 0

        bin_stats.append({
            'bin': bin_name,
            'count': bin_total,
            'fraud_rate': bin_events / bin_total if bin_total > 0 else 0,
            'woe': woe,
            'iv': iv_bin
        })

    return pd.DataFrame(bin_stats)