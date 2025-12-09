"""
Collection of auxiliary functions to create pandas tables.

Ideal for sampling a finite amount of discrete points in a multi-dimensional parameter space.


note: Prefer sklearn.model_selection.ParameterGrid for similar functions if possible.
"""

from typing import Any, Iterable
import numpy as np
import pandas as pd


def _cartesian_product(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Compute the Cartesian product of two DataFrames."""
    x = x.copy()
    y = y.copy()
    x["key"] = 1
    y["key"] = 1
    return x.merge(y, on="key").drop("key", axis=1)


def append_scanned_dimension(
    table: pd.DataFrame,
    name: str,
    values: Iterable[Any],
) -> pd.DataFrame:
    """Cartesian product of the existing table with a new dimension."""
    return _cartesian_product(table, pd.DataFrame({name: list(values)}))


def make_grid_scan_table(params: Iterable[dict[str, Iterable[Any]]]) -> pd.DataFrame:
    """Make a DataFrame representing the cartesian product of the given parameter values."""
    grid = pd.DataFrame({"---ignore---": [1]})  # dummy unity
    for param in params:
        for name, values in param.items():
            grid = append_scanned_dimension(grid, name, values)
    grid.drop("---ignore---", axis=1, inplace=True)  # cleanup
    return grid


def make_uniformly_randomly_sampled_table(
    params: dict[str, tuple[float, float]], number_of_samples: int
) -> pd.DataFrame:
    """
    Make a DataFrame with uniformly randomly sampled parameters of given bounds and count.

    Args:
        params (dict): Dictionary of parameter names and their (min, max) bounds.
        number_of_samples (int): Amount of samples to generate.
    """
    data = {}
    for name, (p_min, p_max) in params.items():
        values = np.random.uniform(p_min, p_max, number_of_samples)
        data[name] = values
    return pd.DataFrame(data)


def make_gaussian_randomly_sampled_table(
    params: dict[str, tuple[float, float]], number_of_samples: int
) -> pd.DataFrame:
    """
    Make a DataFrame with Gaussian randomly sampled parameters of given bounds and count.

    Args:
        params (dict): Dictionary of parameter names and their (mu, sigma) values.
        number_of_samples (int): Amount of samples to generate.
    """
    data = {}
    for name, (mu, sigma) in params.items():
        values = np.random.normal(mu, sigma, number_of_samples)
        data[name] = values
    return pd.DataFrame(data)


def iter_rows(df: pd.DataFrame) -> Iterable[dict[str, Any]]:
    """Iterate over rows of a DataFrame as dictionaries."""
    for row in df.itertuples(index=False):
        yield row._asdict()


def unique_values_per_column(df: pd.DataFrame) -> dict[str, set[Any]]:
    """Get unique values per column in a DataFrame."""
    return {col: set(df[col].unique()) for col in df.columns}
