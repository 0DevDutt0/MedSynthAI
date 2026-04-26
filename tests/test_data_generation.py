import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.generate_synthetic_data import generate_synthetic_data
from src.data_validation import DataValidator

def test_dataset_shape():
    """Test that the dataset has exactly 10,000 rows."""
    df = generate_synthetic_data(10000)
    assert len(df) == 10000, f"Expected 10000 rows, got {len(df)}"

def test_dataset_columns():
    """Test that the dataset has the correct columns."""
    df = generate_synthetic_data(10000)

    from src.generate_synthetic_data import NUMERICAL_FEATURES, TEXT_FEATURES, LABELS
    expected_features = NUMERICAL_FEATURES + TEXT_FEATURES + LABELS

    assert set(df.columns) == set(expected_features), \
        f"Columns mismatch. Expected {len(expected_features)} features, got {len(df.columns)}"

def test_label_noise():
    """Test that label noise is within expected tolerance."""
    df = generate_synthetic_data(10000)

    # Test sepsis_risk noise (should be around 5%)
    sepsis_conditions = (df['hr'] > 100) & (df['temp'] > 38) & (df['wbc'] > 12) & (df['lactate'] > 2) & (df['crp'] > 5)
    expected_sepsis = sepsis_conditions.astype(int)

    # Count how many times the noise flipped labels
    actual_sepsis = df['sepsis_risk']
    noise_sepsis = (expected_sepsis != actual_sepsis).sum()
    noise_rate_sepsis = noise_sepsis / len(df)

    # Should be around 5% noise
    assert 0.03 <= noise_rate_sepsis <= 0.07, \
        f"Sepsis noise rate {noise_rate_sepsis:.3f} not in expected range [0.03, 0.07]"

    # Test cardiac_event noise (should be around 4%)
    cardiac_conditions = ((df['troponin'] > 0.5) | (df['bnp'] > 100)) & (df['sbp'] < 90)
    expected_cardiac = cardiac_conditions.astype(int)

    # Add chest pain logic
    chest_pain_mention = df['nursing_notes'].str.contains('chest pain', case=False, na=False)
    expected_cardiac |= chest_pain_mention

    # Count noise
    actual_cardiac = df['cardiac_event']
    noise_cardiac = (expected_cardiac != actual_cardiac).sum()
    noise_rate_cardiac = noise_cardiac / len(df)

    # Should be around 4% noise
    assert 0.02 <= noise_rate_cardiac <= 0.06, \
        f"Cardiac noise rate {noise_rate_cardiac:.3f} not in expected range [0.02, 0.06]"

def test_missing_values():
    """Test that missing value percentages are correct."""
    df = generate_synthetic_data(10000)

    from src.generate_synthetic_data import NUMERICAL_FEATURES, TEXT_FEATURES

    # Test MCAR (8% for numerical features)
    numerical_missing = df[NUMERICAL_FEATURES].isnull().sum()
    total_missing = numerical_missing.sum()
    total_values = len(df) * len(NUMERICAL_FEATURES)
    mcar_rate = total_missing / total_values

    assert 0.07 <= mcar_rate <= 0.09, \
        f"MCAR rate {mcar_rate:.3f} not in expected range [0.07, 0.09]"

    # Test MAR for lactate (missing more when sepsis_risk=1)
    lactate_missing = df['lactate'].isnull()
    sepsis_risk_1 = df['sepsis_risk'] == 1
    lactate_missing_sepsis_1 = lactate_missing & sepsis_risk_1
    lactate_missing_sepsis_0 = lactate_missing & ~sepsis_risk_1

    # Calculate missing rates
    missing_rate_sepsis_1 = lactate_missing_sepsis_1.sum() / sepsis_risk_1.sum() if sepsis_risk_1.sum() > 0 else 0
    missing_rate_sepsis_0 = lactate_missing_sepsis_0.sum() / (~sepsis_risk_1).sum() if (~sepsis_risk_1).sum() > 0 else 0

    # MAR pattern: missing rate should be higher when sepsis_risk=1
    assert missing_rate_sepsis_1 > missing_rate_sepsis_0, \
        f"MAR pattern not satisfied: sepsis=1 rate ({missing_rate_sepsis_1:.3f}) should be > sepsis=0 rate ({missing_rate_sepsis_0:.3f})"

    # Test MNAR for troponin (missing when extremely high)
    troponin_high = df['troponin'] > df['troponin'].quantile(0.99)
    troponin_missing = df['troponin'].isnull()
    troponin_missing_high = troponin_missing & troponin_high
    mnar_rate = troponin_missing_high.sum() / troponin_high.sum() if troponin_high.sum() > 0 else 0

    assert mnar_rate > 0.005, \
        f"MNAR rate {mnar_rate:.3f} not sufficient (>0.005)"

    # Test text fields missing (1%)
    text_missing = df[TEXT_FEATURES].isnull().sum()
    text_missing_rate = text_missing.sum() / (len(df) * len(TEXT_FEATURES))

    assert 0.005 <= text_missing_rate <= 0.015, \
        f"Text missing rate {text_missing_rate:.3f} not in expected range [0.005, 0.015]"

def test_text_fields():
    """Test that text fields contain expected keywords."""
    df = generate_synthetic_data(10000)

    # Test nursing notes contain expected elements
    assert df['nursing_notes'].notnull().sum() > 0, "Nursing notes should not be all null"

    # Check for common keywords
    keywords = ['patient', 'complains', 'intervention', 'awaiting']
    for keyword in keywords:
        count = df['nursing_notes'].str.contains(keyword, case=False, na=False).sum()
        assert count > 0, f"No occurrences of keyword '{keyword}' in nursing notes"

    # Test radiology impressions
    assert df['radiology_impression'].notnull().sum() > 0, "Radiology impressions should not be all null"

    # Test patient history
    assert df['patient_history'].notnull().sum() > 0, "Patient history should not be all null"

def test_reproducibility():
    """Test that the same random seed produces identical datasets."""
    df1 = generate_synthetic_data(10000)
    df2 = generate_synthetic_data(10000)

    # Check if datasets are identical
    pd.testing.assert_frame_equal(df1, df2)

def test_data_validation():
    """Test that the generated data passes validation."""
    df = generate_synthetic_data(10000)

    # Test that validation passes
    validator = DataValidator(df)
    assert validator.validate(), "Data validation failed"

if __name__ == "__main__":
    # Run tests directly
    test_dataset_shape()
    test_dataset_columns()
    test_label_noise()
    test_missing_values()
    test_text_fields()
    test_reproducibility()
    test_data_validation()
    print("All tests passed!")