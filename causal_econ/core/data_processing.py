"""
Data processing classes for panel data handling.

This module provides classes for loading, preprocessing, and transforming
panel data for causal analysis.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import warnings


class PanelDataLoader:
    """
    Load and validate panel data for causal analysis.

    Example:
    --------
    >>> loader = PanelDataLoader()
    >>> df = loader.load_from_csv('data.csv')
    >>> loader.validate_structure(df)
    """

    def __init__(self,
                 required_columns: Optional[List[str]] = None,
                 country_col: str = 'country',
                 year_col: str = 'year',
                 outcome_col: str = 'gdp_growth_annual_share'):
        """
        Initialize data loader.

        Parameters:
        -----------
        required_columns : list, optional
            List of required column names
        country_col : str, optional
            Name of country column (default: 'country')
        year_col : str, optional
            Name of year column (default: 'year')
        outcome_col : str, optional
            Name of outcome column (default: 'gdp_growth_annual_share')
        """
        self.required_columns = required_columns or [country_col, year_col, outcome_col]
        self.country_col = country_col
        self.year_col = year_col
        self.outcome_col = outcome_col

    def load_from_csv(self,
                     file_path: Union[str, Path],
                     **kwargs) -> pd.DataFrame:
        """
        Load panel data from CSV file.

        Parameters:
        -----------
        file_path : str or Path
            Path to CSV file
        **kwargs
            Additional arguments for pd.read_csv()

        Returns:
        --------
        pd.DataFrame : Loaded data

        Example:
        --------
        >>> df = loader.load_from_csv('data.csv', encoding='utf-8')
        """
        df = pd.read_csv(file_path, **kwargs)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df

    def validate_structure(self,
                          df: pd.DataFrame,
                          raise_error: bool = True) -> bool:
        """
        Validate panel data structure.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to validate
        raise_error : bool, optional
            Raise error if validation fails (default: True)

        Returns:
        --------
        bool : True if valid

        Raises:
        -------
        ValueError : If validation fails and raise_error=True

        Example:
        --------
        >>> is_valid = loader.validate_structure(df)
        """
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            if raise_error:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False

        # Check for duplicates
        if self.country_col in df.columns and self.year_col in df.columns:
            duplicates = df.duplicated(subset=[self.country_col, self.year_col])
            if duplicates.any():
                n_dups = duplicates.sum()
                msg = f"Found {n_dups} duplicate country-year combinations"
                if raise_error:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return False

        # Check data types
        if self.year_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.year_col]):
                msg = f"Year column '{self.year_col}' must be numeric"
                if raise_error:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return False

        print("✓ Data structure validation passed")
        return True


class PanelDataPreprocessor:
    """
    Preprocess panel data for causal analysis.

    Example:
    --------
    >>> preprocessor = PanelDataPreprocessor()
    >>> data_dict = preprocessor.process(df, treated_countries, treatment_years)
    """

    def __init__(self,
                 country_col: str = 'country',
                 year_col: str = 'year',
                 outcome_col: str = 'gdp_growth_annual_share'):
        """
        Initialize preprocessor.

        Parameters:
        -----------
        country_col : str
            Country column name
        year_col : str
            Year column name
        outcome_col : str
            Outcome column name
        """
        self.country_col = country_col
        self.year_col = year_col
        self.outcome_col = outcome_col

    def process(self,
                df: pd.DataFrame,
                treated_countries: List[str],
                treatment_years: Dict[str, int],
                fill_method: str = 'ffill') -> Dict:
        """
        Process panel data into analysis-ready format.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw panel data
        treated_countries : list
            List of treated countries
        treatment_years : dict
            Treatment years by country
        fill_method : str, optional
            Method for filling missing values: 'ffill', 'bfill', 'drop'
            (default: 'ffill')

        Returns:
        --------
        dict : Processed data dictionary containing:
            - 'df': Original dataframe
            - 'panel': Pivoted panel (years x countries)
            - 'treated_countries': List of treated countries
            - 'treatment_years': Treatment year mapping

        Example:
        --------
        >>> data_dict = preprocessor.process(df, ['USA', 'China'],
        ...                                  {'USA': 2002, 'China': 2005})
        """
        print("Processing panel data...")

        # Sort data
        df = df.sort_values([self.country_col, self.year_col]).copy()

        # Handle missing values
        df = self._handle_missing_values(df, fill_method)

        # Create pivoted panel
        panel = df.pivot_table(
            index=self.year_col,
            columns=self.country_col,
            values=self.outcome_col,
            aggfunc='mean'  # Handle duplicates
        )

        # Validate treated countries
        valid_treated = [c for c in treated_countries if c in panel.columns]
        if len(valid_treated) < len(treated_countries):
            missing = set(treated_countries) - set(valid_treated)
            warnings.warn(f"Treated countries not in panel: {missing}")

        # Create data dictionary
        data_dict = {
            'df': df,
            'panel': panel,
            'treated_countries': valid_treated,
            'treatment_years': treatment_years
        }

        print(f"✓ Processed panel: {panel.shape[0]} years x {panel.shape[1]} countries")
        print(f"✓ Treated countries: {len(valid_treated)}")

        return data_dict

    def _handle_missing_values(self,
                               df: pd.DataFrame,
                               method: str) -> pd.DataFrame:
        """Handle missing values in panel data."""
        if method == 'ffill':
            df = df.groupby(self.country_col).fillna(method='ffill')
        elif method == 'bfill':
            df = df.groupby(self.country_col).fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna(subset=[self.outcome_col])
        else:
            warnings.warn(f"Unknown fill method '{method}', using ffill")
            df = df.groupby(self.country_col).fillna(method='ffill')

        return df


class TreatmentIdentifier:
    """
    Identify treatment timing and create treatment indicators.

    Example:
    --------
    >>> identifier = TreatmentIdentifier()
    >>> treated, treatment_years = identifier.identify_from_indicator(
    ...     df, treatment_col='has_platform'
    ... )
    """

    def __init__(self,
                 country_col: str = 'country',
                 year_col: str = 'year'):
        """
        Initialize treatment identifier.

        Parameters:
        -----------
        country_col : str
            Country column name
        year_col : str
            Year column name
        """
        self.country_col = country_col
        self.year_col = year_col

    def identify_from_indicator(self,
                                df: pd.DataFrame,
                                treatment_col: str = 'has_platform') -> Tuple[List[str], Dict[str, int]]:
        """
        Identify treated countries and treatment years from indicator column.

        Parameters:
        -----------
        df : pd.DataFrame
            Panel data
        treatment_col : str, optional
            Treatment indicator column (default: 'has_platform')

        Returns:
        --------
        tuple : (treated_countries, treatment_years)

        Example:
        --------
        >>> treated, years = identifier.identify_from_indicator(df)
        >>> print(treated)
        ['USA', 'China']
        >>> print(years)
        {'USA': 2002, 'China': 2005}
        """
        if treatment_col not in df.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found")

        # Find first year of treatment for each country
        treated_df = df[df[treatment_col] == 1]

        if treated_df.empty:
            warnings.warn("No treated units found")
            return [], {}

        treatment_timing = treated_df.groupby(self.country_col)[self.year_col].min()

        treated_countries = treatment_timing.index.tolist()
        treatment_years = treatment_timing.to_dict()

        print(f"Identified {len(treated_countries)} treated countries")

        return treated_countries, treatment_years

    def create_indicators(self,
                         df: pd.DataFrame,
                         treatment_years: Dict[str, int]) -> pd.DataFrame:
        """
        Create treatment and post-treatment indicators.

        Parameters:
        -----------
        df : pd.DataFrame
            Panel data
        treatment_years : dict
            Treatment years by country

        Returns:
        --------
        pd.DataFrame : Data with added indicators

        Example:
        --------
        >>> df = identifier.create_indicators(df, treatment_years)
        >>> print(df[['country', 'year', 'treated', 'post_treatment']].head())
        """
        df = df.copy()

        df['treated'] = df[self.country_col].isin(treatment_years.keys())

        df['post_treatment'] = df.apply(
            lambda row: (
                row[self.year_col] >= treatment_years[row[self.country_col]]
                if row[self.country_col] in treatment_years
                else False
            ),
            axis=1
        )

        return df


class FeatureExtractor:
    """
    Extract and prepare features for analysis.

    Example:
    --------
    >>> extractor = FeatureExtractor()
    >>> features = extractor.get_feature_columns(df)
    >>> scaled_features = extractor.scale_features(df, features)
    """

    def __init__(self,
                 exclude_cols: Optional[List[str]] = None):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        exclude_cols : list, optional
            Columns to exclude from features
        """
        self.exclude_cols = exclude_cols or [
            'country', 'year', 'has_platform',
            'gdp_growth_annual_share'
        ]

    def get_feature_columns(self,
                           df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (exclude ID and outcome).

        Parameters:
        -----------
        df : pd.DataFrame
            Data

        Returns:
        --------
        list : Feature column names

        Example:
        --------
        >>> features = extractor.get_feature_columns(df)
        >>> print(features)
        ['gdp_per_capita', 'population', 'trade_openness']
        """
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        return feature_cols

    def aggregate_pretreatment(self,
                              df: pd.DataFrame,
                              country_col: str,
                              year_col: str,
                              treatment_years: Dict[str, int],
                              agg_func: str = 'mean') -> pd.DataFrame:
        """
        Aggregate pre-treatment features by country.

        Parameters:
        -----------
        df : pd.DataFrame
            Panel data
        country_col : str
            Country column
        year_col : str
            Year column
        treatment_years : dict
            Treatment years
        agg_func : str, optional
            Aggregation function (default: 'mean')

        Returns:
        --------
        pd.DataFrame : Aggregated pre-treatment features

        Example:
        --------
        >>> pre_features = extractor.aggregate_pretreatment(
        ...     df, 'country', 'year', treatment_years
        ... )
        """
        feature_cols = self.get_feature_columns(df)

        pre_data = []

        for country in df[country_col].unique():
            country_data = df[df[country_col] == country]

            if country in treatment_years:
                treatment_year = treatment_years[country]
                pre_data_country = country_data[country_data[year_col] < treatment_year]
            else:
                # Use all data for control countries
                pre_data_country = country_data

            if len(pre_data_country) > 0:
                if agg_func == 'mean':
                    agg_values = pre_data_country[feature_cols].mean()
                elif agg_func == 'median':
                    agg_values = pre_data_country[feature_cols].median()
                else:
                    agg_values = pre_data_country[feature_cols].agg(agg_func)

                agg_values[country_col] = country
                pre_data.append(agg_values)

        pre_df = pd.DataFrame(pre_data)
        pre_df = pre_df.set_index(country_col)

        return pre_df


class DataDictBuilder:
    """
    Build standardized data dictionary for analysis.

    Example:
    --------
    >>> builder = DataDictBuilder()
    >>> data_dict = builder.from_csv(
    ...     'data.csv',
    ...     treated_countries=['USA', 'China'],
    ...     treatment_years={'USA': 2002, 'China': 2005}
    ... )
    """

    def __init__(self):
        """Initialize data dict builder."""
        self.loader = PanelDataLoader()
        self.preprocessor = PanelDataPreprocessor()
        self.identifier = TreatmentIdentifier()

    def from_csv(self,
                file_path: Union[str, Path],
                treated_countries: Optional[List[str]] = None,
                treatment_years: Optional[Dict[str, int]] = None,
                treatment_col: Optional[str] = 'has_platform',
                **kwargs) -> Dict:
        """
        Build data dictionary from CSV file.

        Parameters:
        -----------
        file_path : str or Path
            Path to CSV
        treated_countries : list, optional
            Treated countries (if None, identified from treatment_col)
        treatment_years : dict, optional
            Treatment years (if None, identified from treatment_col)
        treatment_col : str, optional
            Treatment indicator column (default: 'has_platform')
        **kwargs
            Additional arguments for data loading

        Returns:
        --------
        dict : Standardized data dictionary

        Example:
        --------
        >>> data_dict = builder.from_csv('data.csv')
        """
        # Load data
        df = self.loader.load_from_csv(file_path, **kwargs)

        # Validate
        self.loader.validate_structure(df)

        # Identify treatment if not provided
        if treated_countries is None or treatment_years is None:
            treated, years = self.identifier.identify_from_indicator(df, treatment_col)
            if treated_countries is None:
                treated_countries = treated
            if treatment_years is None:
                treatment_years = years

        # Process
        data_dict = self.preprocessor.process(df, treated_countries, treatment_years)

        return data_dict
