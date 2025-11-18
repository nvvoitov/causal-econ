"""
Weight optimization utilities for causal inference methods.

This module provides a unified optimization framework that eliminates duplicate
optimization code across Synthetic Control, SDiD, and other methods.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import linalg
import warnings


class WeightOptimizer:
    """
    Unified weight optimization for causal inference methods.

    This class eliminates ~150 lines of duplicate optimization code by providing
    a single, flexible interface for weight optimization across all methods.

    The class supports:
    - Unit weight optimization (Synthetic Control, SDiD)
    - Time weight optimization (SDiD)
    - Custom objective functions
    - Flexible constraints (sum-to-one, non-negativity, custom)
    - Multiple optimization methods

    Example:
    --------
    >>> optimizer = WeightOptimizer()
    >>> # Optimize unit weights for Synthetic Control
    >>> result = optimizer.optimize_unit_weights(
    ...     X_target=treated_data,
    ...     X_donors=donor_data,
    ...     method='rmse'
    ... )
    >>> weights = result['weights']
    """

    def __init__(self,
                 method: str = 'SLSQP',
                 max_iter: int = 1000,
                 tolerance: float = 1e-8,
                 verbose: bool = False):
        """
        Initialize the weight optimizer.

        Parameters:
        -----------
        method : str, optional
            Optimization method for scipy.optimize.minimize (default: 'SLSQP')
        max_iter : int, optional
            Maximum number of iterations (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-8)
        verbose : bool, optional
            Print optimization progress (default: False)
        """
        self.method = method
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose

    def optimize_unit_weights(self,
                             X_target: np.ndarray,
                             X_donors: np.ndarray,
                             method: str = 'rmse',
                             sum_to_one: bool = True,
                             non_negative: bool = True,
                             regularization: float = 0.0,
                             intercept: bool = False,
                             custom_objective: Optional[Callable] = None) -> Dict:
        """
        Optimize unit weights (used in SC and SDiD).

        This method minimizes the difference between target and weighted donors,
        commonly used for creating synthetic controls.

        Parameters:
        -----------
        X_target : np.ndarray
            Target values to match (shape: T x 1 or just T)
        X_donors : np.ndarray
            Donor values (shape: T x N_donors)
        method : str, optional
            Optimization criterion: 'rmse', 'mse', 'mae' (default: 'rmse')
        sum_to_one : bool, optional
            Add constraint that weights sum to 1 (default: True)
        non_negative : bool, optional
            Add constraint that weights >= 0 (default: True)
        regularization : float, optional
            L2 regularization penalty (default: 0.0)
        intercept : bool, optional
            Include intercept term (first parameter) (default: False)
        custom_objective : callable, optional
            Custom objective function with signature func(params) -> float

        Returns:
        --------
        dict : Optimization results containing:
            - 'weights': Optimized weight values (dict or array)
            - 'intercept': Intercept value if intercept=True
            - 'objective_value': Final objective function value
            - 'success': Whether optimization succeeded
            - 'message': Optimization status message
            - 'n_iterations': Number of iterations performed

        Example:
        --------
        >>> X_target = np.array([1, 2, 3, 4, 5])
        >>> X_donors = np.array([[1, 0], [2, 1], [3, 1], [4, 2], [5, 2]])
        >>> result = optimizer.optimize_unit_weights(X_target, X_donors)
        >>> print(result['weights'])
        """
        # Ensure arrays are 2D
        X_target = self._ensure_2d(X_target)
        X_donors = self._ensure_2d(X_donors)

        # Validate shapes
        if X_target.shape[0] != X_donors.shape[0]:
            raise ValueError(
                f"Shape mismatch: X_target has {X_target.shape[0]} rows, "
                f"X_donors has {X_donors.shape[0]} rows"
            )

        n_donors = X_donors.shape[1]
        n_params = n_donors + (1 if intercept else 0)

        # Create objective function
        if custom_objective:
            objective_func = custom_objective
        else:
            objective_func = self._create_unit_weight_objective(
                X_target, X_donors, method, regularization, intercept
            )

        # Setup constraints
        constraints = []
        if sum_to_one:
            if intercept:
                # Sum of weights (excluding intercept) equals 1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda params: np.sum(params[1:]) - 1
                })
            else:
                # Sum of all weights equals 1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda params: np.sum(params) - 1
                })

        # Setup bounds
        if intercept:
            # Intercept unbounded, weights potentially non-negative
            if non_negative:
                bounds = [(None, None)] + [(0, None)] * n_donors
            else:
                bounds = [(None, None)] * n_params
        else:
            # All weights potentially non-negative
            if non_negative:
                bounds = [(0, None)] * n_donors
            else:
                bounds = [(None, None)] * n_donors

        # Initial guess
        if intercept:
            initial_params = np.zeros(n_params)
            if sum_to_one:
                initial_params[1:] = 1.0 / n_donors
        else:
            if sum_to_one:
                initial_params = np.ones(n_donors) / n_donors
            else:
                initial_params = np.zeros(n_donors)

        # Optimize
        opt_result = minimize(
            objective_func,
            initial_params,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tolerance,
                'disp': self.verbose
            }
        )

        # Format results
        if intercept:
            return {
                'intercept': opt_result.x[0],
                'weights': opt_result.x[1:],
                'objective_value': opt_result.fun,
                'success': opt_result.success,
                'message': opt_result.message,
                'n_iterations': opt_result.nit if hasattr(opt_result, 'nit') else None
            }
        else:
            return {
                'weights': opt_result.x,
                'objective_value': opt_result.fun,
                'success': opt_result.success,
                'message': opt_result.message,
                'n_iterations': opt_result.nit if hasattr(opt_result, 'nit') else None
            }

    def optimize_time_weights(self,
                             Y_pre: np.ndarray,
                             Y_post: np.ndarray,
                             regularization: float = 0.0,
                             intercept: bool = False) -> Dict:
        """
        Optimize time weights for pre-treatment periods (used in SDiD).

        Finds weights for pre-treatment periods that best predict post-treatment
        outcomes for control units.

        Parameters:
        -----------
        Y_pre : np.ndarray
            Pre-treatment outcomes (shape: N_control x T_pre)
        Y_post : np.ndarray
            Post-treatment outcomes (shape: N_control,) - average across post periods
        regularization : float, optional
            L2 regularization penalty (default: 0.0)
        intercept : bool, optional
            Include intercept term (default: False)

        Returns:
        --------
        dict : Optimization results with 'intercept' and 'weights'

        Example:
        --------
        >>> # Y_pre: outcomes for 10 control units across 5 pre-treatment periods
        >>> Y_pre = np.random.randn(10, 5)
        >>> # Y_post: average post-treatment outcome for each control unit
        >>> Y_post = np.random.randn(10)
        >>> result = optimizer.optimize_time_weights(Y_pre, Y_post)
        """
        # Ensure correct shapes
        if Y_pre.ndim == 1:
            Y_pre = Y_pre.reshape(1, -1)
        if Y_post.ndim > 1:
            Y_post = Y_post.flatten()

        n_units, n_periods = Y_pre.shape

        if len(Y_post) != n_units:
            raise ValueError(
                f"Shape mismatch: Y_pre has {n_units} units, "
                f"Y_post has {len(Y_post)} units"
            )

        # Create objective function
        def objective(params):
            if intercept:
                lambda_0 = params[0]
                lambda_weights = params[1:]
            else:
                lambda_0 = 0
                lambda_weights = params

            # Weighted pre-period values
            weighted_pre = Y_pre @ lambda_weights

            # Difference from post-period mean
            diff = lambda_0 + weighted_pre - Y_post

            # Sum of squared differences + regularization
            sse = np.sum(diff**2)

            if regularization > 0:
                reg_penalty = regularization * np.sum(lambda_weights**2)
                return sse + reg_penalty
            else:
                return sse

        # Setup
        n_params = n_periods + (1 if intercept else 0)

        constraints = [{
            'type': 'eq',
            'fun': lambda params: np.sum(params[1:] if intercept else params) - 1
        }]

        if intercept:
            bounds = [(None, None)] + [(0, None)] * n_periods
            initial_params = np.zeros(n_params)
            initial_params[1:] = 1.0 / n_periods
        else:
            bounds = [(0, None)] * n_periods
            initial_params = np.ones(n_periods) / n_periods

        # Optimize
        opt_result = minimize(
            objective,
            initial_params,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tolerance,
                'disp': self.verbose
            }
        )

        # Format results
        if intercept:
            return {
                'intercept': opt_result.x[0],
                'weights': opt_result.x[1:],
                'objective_value': opt_result.fun,
                'success': opt_result.success,
                'message': opt_result.message
            }
        else:
            return {
                'weights': opt_result.x,
                'objective_value': opt_result.fun,
                'success': opt_result.success,
                'message': opt_result.message
            }

    def optimize_sc_weights(self,
                           treated_country: str,
                           donor_countries: List[str],
                           panel: pd.DataFrame,
                           pre_treatment_years: List[int]) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Optimize Synthetic Control weights (backward compatible interface).

        This method provides the same interface as the original get_sc_weights
        function for easy drop-in replacement.

        Parameters:
        -----------
        treated_country : str
            Name of treated country
        donor_countries : list
            List of donor country names
        panel : pd.DataFrame
            Panel data with countries as columns, years as index
        pre_treatment_years : list
            List of pre-treatment years

        Returns:
        --------
        tuple : (weights_dict, pre_rmse)
            - weights_dict: Dictionary mapping donor names to weights
            - pre_rmse: Pre-treatment RMSE (objective value)

        Example:
        --------
        >>> weights, rmse = optimizer.optimize_sc_weights(
        ...     'USA', ['Canada', 'UK'], panel, [2000, 2001]
        ... )
        """
        # Filter to available years and donors
        available_years = [y for y in pre_treatment_years if y in panel.index]
        available_donors = [d for d in donor_countries if d in panel.columns]

        if not available_years or not available_donors:
            return None, None

        # Extract data
        X_treated = panel.loc[available_years, treated_country].values
        X_donors = panel.loc[available_years, available_donors].values

        # Handle missing values
        mask = ~np.isnan(X_treated)
        if not np.any(mask):
            return None, None

        X_treated = X_treated[mask]
        X_donors = X_donors[mask]

        # Remove donors with missing values
        valid_donor_indices = [i for i, col in enumerate(X_donors.T)
                              if not np.any(np.isnan(col))]
        if not valid_donor_indices:
            return None, None

        X_donors = X_donors[:, valid_donor_indices]
        valid_donor_names = [available_donors[i] for i in valid_donor_indices]

        # Optimize
        result = self.optimize_unit_weights(
            X_target=X_treated,
            X_donors=X_donors,
            method='rmse',
            sum_to_one=True,
            non_negative=True
        )

        if not result['success']:
            warnings.warn(f"Optimization did not converge: {result['message']}")

        # Create weights dictionary
        weights_dict = dict(zip(valid_donor_names, result['weights']))
        # Fill in zero weights for donors that were excluded
        all_weights = {d: weights_dict.get(d, 0.0) for d in donor_countries}

        return all_weights, result['objective_value']

    def _create_unit_weight_objective(self,
                                     X_target: np.ndarray,
                                     X_donors: np.ndarray,
                                     method: str,
                                     regularization: float,
                                     intercept: bool) -> Callable:
        """
        Create objective function for unit weight optimization.

        Parameters:
        -----------
        X_target : np.ndarray
            Target values
        X_donors : np.ndarray
            Donor values
        method : str
            Optimization criterion ('rmse', 'mse', 'mae')
        regularization : float
            L2 regularization penalty
        intercept : bool
            Whether to include intercept

        Returns:
        --------
        callable : Objective function
        """
        def objective(params):
            if intercept:
                intercept_val = params[0]
                weights = params[1:]
                predicted = X_donors @ weights + intercept_val
            else:
                weights = params
                predicted = X_donors @ weights

            residuals = X_target.flatten() - predicted.flatten()

            # Calculate loss based on method
            if method == 'rmse':
                loss = np.sqrt(np.mean(residuals**2))
            elif method == 'mse':
                loss = np.mean(residuals**2)
            elif method == 'mae':
                loss = np.mean(np.abs(residuals))
            else:
                raise ValueError(f"Unknown method: {method}")

            # Add regularization
            if regularization > 0:
                reg_penalty = regularization * np.sum(weights**2)
                loss += reg_penalty

            return loss

        return objective

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D (convert 1D to column vector)."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    @staticmethod
    def create_weights_dict(weight_values: np.ndarray,
                           names: List[str],
                           threshold: float = 0.0) -> Dict[str, float]:
        """
        Convert weight array to dictionary with optional filtering.

        Parameters:
        -----------
        weight_values : np.ndarray
            Weight values
        names : list
            Names corresponding to weights
        threshold : float, optional
            Only include weights above this threshold (default: 0.0)

        Returns:
        --------
        dict : Dictionary mapping names to weights

        Example:
        --------
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> names = ['A', 'B', 'C']
        >>> WeightOptimizer.create_weights_dict(weights, names, threshold=0.25)
        {'A': 0.5, 'B': 0.3}
        """
        weights_dict = dict(zip(names, weight_values))

        if threshold > 0:
            weights_dict = {k: v for k, v in weights_dict.items() if v > threshold}

        return weights_dict
