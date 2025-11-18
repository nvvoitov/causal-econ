"""
Unit tests for data processing classes.

Tests PanelDataLoader, PanelDataPreprocessor, TreatmentIdentifier,
FeatureExtractor, and DataDictBuilder classes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import warnings

from causal_econ.core.data_processing import (
    PanelDataLoader,
    PanelDataPreprocessor,
    TreatmentIdentifier,
    FeatureExtractor,
    DataDictBuilder
)


@pytest.fixture
def sample_panel_csv(tmp_path):
    """Create a sample CSV file for testing."""
    data = {
        'country': ['USA', 'USA', 'Canada', 'Canada', 'UK', 'UK'],
        'year': [2000, 2001, 2000, 2001, 2000, 2001],
        'gdp_growth_annual_share': [3.5, 4.0, 2.5, 3.0, 2.0, 2.5],
        'has_platform': [0, 1, 0, 0, 0, 1],
        'gdp_per_capita': [35000, 36000, 32000, 33000, 30000, 31000],
        'population': [300, 305, 35, 36, 60, 61]
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_dataframe():
    """Create sample panel dataframe."""
    data = {
        'country': ['USA'] * 5 + ['Canada'] * 5,
        'year': list(range(2000, 2005)) * 2,
        'gdp_growth_annual_share': [3.5, 4.0, 3.8, 4.2, 4.5, 2.5, 3.0, 2.8, 3.2, 3.5],
        'has_platform': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        'gdp_per_capita': np.random.randn(10) * 5000 + 35000,
        'trade_openness': np.random.rand(10) * 50
    }
    return pd.DataFrame(data)


# ==============================================================================
# PanelDataLoader Tests
# ==============================================================================

class TestPanelDataLoader:
    """Test PanelDataLoader class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        loader = PanelDataLoader()

        assert loader.country_col == 'country'
        assert loader.year_col == 'year'
        assert loader.outcome_col == 'gdp_growth_annual_share'
        assert 'country' in loader.required_columns
        assert 'year' in loader.required_columns

    def test_initialization_custom(self):
        """Test custom initialization."""
        loader = PanelDataLoader(
            required_columns=['country', 'year', 'gdp'],
            country_col='nation',
            year_col='time',
            outcome_col='gdp'
        )

        assert loader.country_col == 'nation'
        assert loader.year_col == 'time'
        assert loader.outcome_col == 'gdp'
        assert loader.required_columns == ['country', 'year', 'gdp']

    def test_load_from_csv(self, sample_panel_csv):
        """Test loading CSV file."""
        loader = PanelDataLoader()
        df = loader.load_from_csv(sample_panel_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert 'country' in df.columns
        assert 'year' in df.columns

    def test_validate_structure_success(self, sample_dataframe):
        """Test successful validation."""
        loader = PanelDataLoader()
        result = loader.validate_structure(sample_dataframe)

        assert result is True

    def test_validate_structure_missing_columns(self):
        """Test validation with missing columns."""
        loader = PanelDataLoader()

        # Missing 'year' column
        df = pd.DataFrame({
            'country': ['USA', 'Canada'],
            'gdp_growth_annual_share': [3.5, 2.5]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            loader.validate_structure(df, raise_error=True)

    def test_validate_structure_duplicates(self):
        """Test validation with duplicate country-year."""
        loader = PanelDataLoader()

        df = pd.DataFrame({
            'country': ['USA', 'USA', 'Canada'],
            'year': [2000, 2000, 2000],  # Duplicate USA-2000
            'gdp_growth_annual_share': [3.5, 4.0, 2.5]
        })

        with pytest.raises(ValueError, match="duplicate country-year"):
            loader.validate_structure(df, raise_error=True)

    def test_validate_structure_non_numeric_year(self):
        """Test validation with non-numeric year."""
        loader = PanelDataLoader()

        df = pd.DataFrame({
            'country': ['USA', 'Canada'],
            'year': ['2000', '2001'],  # String years
            'gdp_growth_annual_share': [3.5, 2.5]
        })

        with pytest.raises(ValueError, match="must be numeric"):
            loader.validate_structure(df, raise_error=True)

    def test_validate_structure_warning_mode(self):
        """Test validation in warning mode."""
        loader = PanelDataLoader()

        df = pd.DataFrame({
            'country': ['USA', 'Canada']
            # Missing required columns
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = loader.validate_structure(df, raise_error=False)

            assert result is False
            assert len(w) > 0


# ==============================================================================
# PanelDataPreprocessor Tests
# ==============================================================================

class TestPanelDataPreprocessor:
    """Test PanelDataPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = PanelDataPreprocessor()

        assert preprocessor.country_col == 'country'
        assert preprocessor.year_col == 'year'
        assert preprocessor.outcome_col == 'gdp_growth_annual_share'

    def test_process_success(self, sample_dataframe):
        """Test successful data processing."""
        preprocessor = PanelDataPreprocessor()

        result = preprocessor.process(
            df=sample_dataframe,
            treated_countries=['USA'],
            treatment_years={'USA': 2002}
        )

        assert 'df' in result
        assert 'panel' in result
        assert 'treated_countries' in result
        assert 'treatment_years' in result

        assert isinstance(result['panel'], pd.DataFrame)
        assert 'USA' in result['panel'].columns
        assert 'Canada' in result['panel'].columns

    def test_process_missing_treated_country(self, sample_dataframe):
        """Test processing with missing treated country."""
        preprocessor = PanelDataPreprocessor()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = preprocessor.process(
                df=sample_dataframe,
                treated_countries=['USA', 'Germany'],  # Germany not in data
                treatment_years={'USA': 2002, 'Germany': 2003}
            )

            assert 'Germany' not in result['treated_countries']
            assert len(w) > 0

    def test_handle_missing_values_ffill(self):
        """Test forward fill for missing values."""
        preprocessor = PanelDataPreprocessor()

        df = pd.DataFrame({
            'country': ['USA', 'USA', 'USA'],
            'year': [2000, 2001, 2002],
            'gdp_growth_annual_share': [3.5, np.nan, 4.5]
        })

        result = preprocessor._handle_missing_values(df, 'ffill')

        # NaN at index 1 should be filled with 3.5
        assert result.loc[1, 'gdp_growth_annual_share'] == 3.5

    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        preprocessor = PanelDataPreprocessor()

        df = pd.DataFrame({
            'country': ['USA', 'USA', 'USA'],
            'year': [2000, 2001, 2002],
            'gdp_growth_annual_share': [3.5, np.nan, 4.5]
        })

        result = preprocessor._handle_missing_values(df, 'drop')

        assert len(result) == 2  # One row dropped


# ==============================================================================
# TreatmentIdentifier Tests
# ==============================================================================

class TestTreatmentIdentifier:
    """Test TreatmentIdentifier class."""

    def test_initialization(self):
        """Test identifier initialization."""
        identifier = TreatmentIdentifier()

        assert identifier.country_col == 'country'
        assert identifier.year_col == 'year'

    def test_identify_from_indicator_success(self, sample_dataframe):
        """Test treatment identification from indicator."""
        identifier = TreatmentIdentifier()

        treated, years = identifier.identify_from_indicator(
            df=sample_dataframe,
            treatment_col='has_platform'
        )

        assert 'USA' in treated
        assert 'USA' in years
        assert years['USA'] == 2002  # First year with has_platform=1

    def test_identify_from_indicator_no_treated(self):
        """Test identification with no treated units."""
        identifier = TreatmentIdentifier()

        df = pd.DataFrame({
            'country': ['USA', 'Canada'],
            'year': [2000, 2000],
            'has_platform': [0, 0]  # No treatment
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            treated, years = identifier.identify_from_indicator(df)

            assert len(treated) == 0
            assert len(years) == 0
            assert len(w) > 0

    def test_identify_from_indicator_missing_column(self, sample_dataframe):
        """Test identification with missing treatment column."""
        identifier = TreatmentIdentifier()

        with pytest.raises(ValueError, match="not found"):
            identifier.identify_from_indicator(
                sample_dataframe,
                treatment_col='nonexistent_column'
            )

    def test_create_indicators(self, sample_dataframe):
        """Test creating treatment indicators."""
        identifier = TreatmentIdentifier()

        treatment_years = {'USA': 2002}

        result = identifier.create_indicators(
            df=sample_dataframe,
            treatment_years=treatment_years
        )

        assert 'treated' in result.columns
        assert 'post_treatment' in result.columns

        # Check USA is marked as treated
        usa_data = result[result['country'] == 'USA']
        assert all(usa_data['treated'])

        # Check Canada is not treated
        canada_data = result[result['country'] == 'Canada']
        assert not any(canada_data['treated'])

        # Check post-treatment for USA
        usa_post = usa_data[usa_data['year'] >= 2002]
        assert all(usa_post['post_treatment'])


# ==============================================================================
# FeatureExtractor Tests
# ==============================================================================

class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        extractor = FeatureExtractor()

        assert 'country' in extractor.exclude_cols
        assert 'year' in extractor.exclude_cols

    def test_initialization_custom(self):
        """Test custom initialization."""
        extractor = FeatureExtractor(exclude_cols=['id', 'date'])

        assert extractor.exclude_cols == ['id', 'date']

    def test_get_feature_columns(self, sample_dataframe):
        """Test getting feature columns."""
        extractor = FeatureExtractor()

        features = extractor.get_feature_columns(sample_dataframe)

        assert 'gdp_per_capita' in features
        assert 'trade_openness' in features
        assert 'country' not in features
        assert 'year' not in features
        assert 'gdp_growth_annual_share' not in features

    def test_aggregate_pretreatment(self, sample_dataframe):
        """Test pre-treatment aggregation."""
        extractor = FeatureExtractor()

        treatment_years = {'USA': 2002}

        result = extractor.aggregate_pretreatment(
            df=sample_dataframe,
            country_col='country',
            year_col='year',
            treatment_years=treatment_years,
            agg_func='mean'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'USA' in result.index
        assert 'Canada' in result.index

        # USA should only use years < 2002
        # Canada (control) should use all years

    def test_aggregate_pretreatment_median(self, sample_dataframe):
        """Test pre-treatment aggregation with median."""
        extractor = FeatureExtractor()

        treatment_years = {'USA': 2002}

        result = extractor.aggregate_pretreatment(
            df=sample_dataframe,
            country_col='country',
            year_col='year',
            treatment_years=treatment_years,
            agg_func='median'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ==============================================================================
# DataDictBuilder Tests
# ==============================================================================

class TestDataDictBuilder:
    """Test DataDictBuilder class."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = DataDictBuilder()

        assert builder.loader is not None
        assert builder.preprocessor is not None
        assert builder.identifier is not None

    def test_from_csv_success(self, sample_panel_csv):
        """Test building data dict from CSV."""
        builder = DataDictBuilder()

        result = builder.from_csv(
            file_path=sample_panel_csv,
            treatment_col='has_platform'
        )

        assert 'df' in result
        assert 'panel' in result
        assert 'treated_countries' in result
        assert 'treatment_years' in result

        assert len(result['treated_countries']) > 0

    def test_from_csv_explicit_treatment(self, sample_panel_csv):
        """Test building with explicitly specified treatment."""
        builder = DataDictBuilder()

        result = builder.from_csv(
            file_path=sample_panel_csv,
            treated_countries=['USA'],
            treatment_years={'USA': 2001}
        )

        assert 'USA' in result['treated_countries']
        assert result['treatment_years']['USA'] == 2001

    def test_from_csv_partial_explicit(self, sample_panel_csv):
        """Test with partially explicit treatment specification."""
        builder = DataDictBuilder()

        # Provide countries but not years
        result = builder.from_csv(
            file_path=sample_panel_csv,
            treated_countries=['USA', 'UK'],
            treatment_col='has_platform'
        )

        assert 'USA' in result['treated_countries']
        assert 'treatment_years' in result


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestDataProcessingIntegration:
    """Test integration of data processing components."""

    def test_full_pipeline(self, sample_panel_csv):
        """Test complete data processing pipeline."""
        # Build data dict
        builder = DataDictBuilder()
        data_dict = builder.from_csv(sample_panel_csv)

        # Identify treatment
        identifier = TreatmentIdentifier()
        df_with_indicators = identifier.create_indicators(
            data_dict['df'],
            data_dict['treatment_years']
        )

        # Extract features
        extractor = FeatureExtractor()
        features = extractor.get_feature_columns(df_with_indicators)

        assert len(features) > 0
        assert 'treated' in df_with_indicators.columns
        assert 'post_treatment' in df_with_indicators.columns

    def test_loader_preprocessor_chain(self, sample_panel_csv):
        """Test chaining loader and preprocessor."""
        loader = PanelDataLoader()
        df = loader.load_from_csv(sample_panel_csv)

        # Validate
        is_valid = loader.validate_structure(df)
        assert is_valid

        # Identify treatment
        identifier = TreatmentIdentifier()
        treated, years = identifier.identify_from_indicator(df)

        # Preprocess
        preprocessor = PanelDataPreprocessor()
        data_dict = preprocessor.process(df, treated, years)

        assert 'panel' in data_dict
        assert data_dict['panel'].shape[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
