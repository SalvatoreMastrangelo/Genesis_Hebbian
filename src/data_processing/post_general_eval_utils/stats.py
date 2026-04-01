from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, pearsonr
except Exception:
    mannwhitneyu = None
    pearsonr = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


def format_p_value(p_value: float | None) -> str:
    if p_value is None:
        return 'n/a'
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return 'n/a'
    if not np.isfinite(p):
        return 'n/a'
    if p < 1e-4:
        return f'{p:.1e}'
    return f'{p:.4f}'


def benjamini_hochberg(p_values) -> np.ndarray:
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return out
    finite_vals = arr[finite_mask]
    if multipletests is not None:
        _, q_vals, _, _ = multipletests(finite_vals, method='fdr_bh')
        out[finite_mask] = q_vals
        return out
    order = np.argsort(finite_vals)
    ranked = finite_vals[order]
    n = len(ranked)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * n / rank
        prev = min(prev, candidate)
        q[idx] = prev
    reordered = np.empty(n, dtype=float)
    reordered[order] = np.clip(q, 0.0, 1.0)
    out[finite_mask] = reordered
    return out


def pearson_corr_stats(x_values, y_values) -> dict[str, float | int]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    result = {'r': np.nan, 'p_value': np.nan, 'n': n}
    if n < 2:
        return result
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return result
    if pearsonr is None:
        result['r'] = float(np.corrcoef(x, y)[0, 1])
        return result
    r_value, p_value = pearsonr(x, y)
    result['r'] = float(r_value)
    result['p_value'] = float(p_value)
    return result


def mann_whitney_stats(x_values, y_values) -> dict[str, float | int]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    result = {
        'u_stat': np.nan,
        'p_value': np.nan,
        'n_x': int(len(x)),
        'n_y': int(len(y)),
        'median_delta': np.nan,
    }
    if len(x) == 0 or len(y) == 0:
        return result
    result['median_delta'] = float(np.median(x) - np.median(y))
    if mannwhitneyu is None:
        return result
    try:
        u_stat, p_value = mannwhitneyu(x, y, alternative='two-sided', method='auto')
    except TypeError:
        u_stat, p_value = mannwhitneyu(x, y, alternative='two-sided')
    result['u_stat'] = float(u_stat)
    result['p_value'] = float(p_value)
    return result


def significance_marker(*, q_value: float | None = None, p_value: float | None = None) -> str:
    q = float(q_value) if q_value is not None and np.isfinite(q_value) else np.nan
    p = float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan
    if np.isfinite(q):
        if q < 0.001:
            return '***'
        if q < 0.01:
            return '**'
        if q < 0.05:
            return '*'
        return ''
    if np.isfinite(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
    return ''


def save_stats_table(save_path, df: pd.DataFrame, suffix: str = '_stats.csv') -> Path | None:
    if save_path is None:
        return None
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path = target.with_name(f'{target.stem}{suffix}')
    df.to_csv(csv_path, index=False)
    return csv_path


def matrix_stats_frame(row_labels, col_labels, corr, p_values, q_values, counts, *, row_name='row', col_name='metric') -> pd.DataFrame:
    records = []
    for ri, row_label in enumerate(row_labels):
        for ci, col_label in enumerate(col_labels):
            records.append(
                {
                    row_name: row_label,
                    col_name: col_label,
                    'pearson_r': corr[ri, ci],
                    'p_value': p_values[ri, ci],
                    'q_value_fdr_bh': q_values[ri, ci],
                    'n': counts[ri, ci],
                }
            )
    return pd.DataFrame.from_records(records)
