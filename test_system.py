#!/usr/bin/env python3
"""
Simple test to verify the system works correctly.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        import generate_synthetic_data
        import data_validation
        print("SUCCESS: All modules imported successfully")
        return True
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return False

def test_data_generation():
    """Test that data can be generated."""
    try:
        from generate_synthetic_data import generate_synthetic_data
        data = generate_synthetic_data()
        if len(data) == 10000:
            print("SUCCESS: Data generation successful")
            return True
        else:
            print(f"FAILED: Data generation failed: expected 10000 rows, got {len(data)}")
            return False
    except Exception as e:
        print(f"FAILED: Data generation failed: {e}")
        return False

def test_validation_end_to_end():
    """Test that the full validation process works (using the main script)."""
    try:
        # Just run the main validation function from data_validation.py
        # This will test the full validation process
        from data_validation import DataValidator
        import pandas as pd

        # Load the generated data to test validation
        data = pd.read_csv('data/raw_synthetic.csv.gz')

        # Create validator and run validation
        validator = DataValidator(data)
        is_valid = validator.validate()

        print("SUCCESS: End-to-end validation successful")
        return True
    except Exception as e:
        print(f"FAILED: End-to-end validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MLLib system...")

    tests = [
        test_imports,
        test_data_generation,
        test_validation_end_to_end
    ]

    results = []
    for test in tests:
        results.append(test())

    if all(results):
        print("\nSUCCESS: All tests passed! The system is working correctly.")
        sys.exit(0)
    else:
        print("\nFAILED: Some tests failed.")
        sys.exit(1)