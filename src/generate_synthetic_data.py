import pandas as pd
import numpy as np
import random
import gzip
from pathlib import Path
from typing import Tuple, List
from loguru import logger
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Define numerical features with their distributions
NUMERICAL_FEATURES = [
    'hr', 'resp_rate', 'spo2', 'temp', 'sbp', 'dbp', 'map', 'glucose', 'lactate',
    'creatinine', 'bun', 'wbc', 'plt', 'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p',
    'albumin', 'bilirubin', 'ast', 'alt', 'alp', 'ggt', 'ptt', 'inr', 'troponin',
    'bnp', 'procalcitonin', 'crp', 'd_dimer', 'fibrinogen', 'ph', 'pco2', 'po2',
    'hco3', 'base_excess', 'urine_output'
]

# Define text features
TEXT_FEATURES = ['nursing_notes', 'radiology_impression', 'patient_history']

# Define labels
LABELS = ['sepsis_risk', 'cardiac_event', 'stable']

# Define all features
ALL_FEATURES = NUMERICAL_FEATURES + TEXT_FEATURES + LABELS

def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic ICU step-down ward dataset."""

    logger.info(f"Generating synthetic dataset with {n_samples} samples...")

    # Create base dataframe
    data = pd.DataFrame(index=range(n_samples))

    # Generate numerical features with realistic distributions
    for feature in NUMERICAL_FEATURES:
        if feature in ['hr', 'resp_rate', 'spo2', 'temp']:
            # Vital signs - normal distributions
            if feature == 'hr':
                data[feature] = np.random.normal(75, 12, n_samples)
            elif feature == 'resp_rate':
                data[feature] = np.random.normal(16, 4, n_samples)
            elif feature == 'spo2':
                data[feature] = np.random.normal(98, 2, n_samples)
            elif feature == 'temp':
                data[feature] = np.random.normal(37.0, 0.8, n_samples)
        elif feature in ['sbp', 'dbp', 'map']:
            # Blood pressure - truncated normal
            if feature == 'sbp':
                data[feature] = np.random.normal(120, 15, n_samples)
            elif feature == 'dbp':
                data[feature] = np.random.normal(80, 10, n_samples)
            elif feature == 'map':
                data[feature] = np.random.normal(90, 12, n_samples)
        elif feature in ['glucose', 'lactate', 'creatinine', 'bun']:
            # Lab values - log-normal distributions
            if feature == 'glucose':
                data[feature] = np.random.lognormal(3.2, 0.5, n_samples)
            elif feature == 'lactate':
                data[feature] = np.random.lognormal(1.5, 0.8, n_samples)
            elif feature == 'creatinine':
                data[feature] = np.random.lognormal(1.8, 0.6, n_samples)
            elif feature == 'bun':
                data[feature] = np.random.lognormal(2.0, 0.7, n_samples)
        elif feature in ['wbc', 'plt', 'hgb', 'na', 'k', 'cl', 'ca', 'mg', 'p']:
            # Blood counts and electrolytes
            if feature == 'wbc':
                data[feature] = np.random.lognormal(3.0, 0.8, n_samples)
            elif feature == 'plt':
                data[feature] = np.random.lognormal(4.0, 0.6, n_samples)
            elif feature == 'hgb':
                data[feature] = np.random.normal(14.0, 2.0, n_samples)
            elif feature in ['na', 'k', 'cl', 'ca', 'mg', 'p']:
                # Electrolytes
                if feature == 'na':
                    data[feature] = np.random.normal(140, 5, n_samples)
                elif feature == 'k':
                    data[feature] = np.random.normal(4.0, 0.5, n_samples)
                elif feature == 'cl':
                    data[feature] = np.random.normal(105, 5, n_samples)
                elif feature == 'ca':
                    data[feature] = np.random.normal(9.0, 0.5, n_samples)
                elif feature == 'mg':
                    data[feature] = np.random.normal(2.0, 0.3, n_samples)
                elif feature == 'p':
                    data[feature] = np.random.normal(3.0, 0.5, n_samples)
        elif feature in ['albumin', 'bilirubin', 'ast', 'alt', 'alp', 'ggt']:
            # Liver function tests
            if feature == 'albumin':
                data[feature] = np.random.normal(4.0, 0.5, n_samples)
            elif feature == 'bilirubin':
                data[feature] = np.random.lognormal(0.5, 0.8, n_samples)
            elif feature in ['ast', 'alt']:
                if feature == 'ast':
                    data[feature] = np.random.lognormal(2.5, 1.0, n_samples)
                else:  # alt
                    data[feature] = np.random.lognormal(2.5, 1.0, n_samples)
            elif feature in ['alp', 'ggt']:
                if feature == 'alp':
                    data[feature] = np.random.lognormal(3.0, 0.8, n_samples)
                else:  # ggt
                    data[feature] = np.random.lognormal(2.0, 1.0, n_samples)
        elif feature in ['ptt', 'inr', 'troponin', 'bnp', 'procalcitonin', 'crp', 'd_dimer', 'fibrinogen']:
            # Coagulation and cardiac markers
            if feature == 'ptt':
                data[feature] = np.random.lognormal(2.0, 0.5, n_samples)
            elif feature == 'inr':
                data[feature] = np.random.lognormal(0.5, 0.3, n_samples)
            elif feature == 'troponin':
                data[feature] = np.random.lognormal(0.5, 1.0, n_samples)
            elif feature == 'bnp':
                data[feature] = np.random.lognormal(1.5, 1.0, n_samples)
            elif feature == 'procalcitonin':
                data[feature] = np.random.lognormal(0.5, 1.0, n_samples)
            elif feature == 'crp':
                data[feature] = np.random.lognormal(1.5, 1.0, n_samples)
            elif feature == 'd_dimer':
                data[feature] = np.random.lognormal(2.5, 1.0, n_samples)
            elif feature == 'fibrinogen':
                data[feature] = np.random.lognormal(3.0, 0.8, n_samples)
        elif feature in ['ph', 'pco2', 'po2', 'hco3', 'base_excess']:
            # Blood gas values
            if feature == 'ph':
                data[feature] = np.random.normal(7.4, 0.05, n_samples)
            elif feature == 'pco2':
                data[feature] = np.random.normal(40, 5, n_samples)
            elif feature == 'po2':
                data[feature] = np.random.normal(80, 10, n_samples)
            elif feature == 'hco3':
                data[feature] = np.random.normal(24, 3, n_samples)
            elif feature == 'base_excess':
                data[feature] = np.random.normal(0, 2, n_samples)
        elif feature == 'urine_output':
            # Urine output
            data[feature] = np.random.lognormal(2.0, 0.8, n_samples)

    # Ensure realistic ranges for some features
    data['hr'] = np.clip(data['hr'], 40, 180)
    data['resp_rate'] = np.clip(data['resp_rate'], 8, 35)
    data['spo2'] = np.clip(data['spo2'], 85, 100)
    data['temp'] = np.clip(data['temp'], 35, 42)
    data['sbp'] = np.clip(data['sbp'], 60, 220)
    data['dbp'] = np.clip(data['dbp'], 40, 130)
    data['map'] = np.clip(data['map'], 50, 150)

    # Add outliers (2% of values)
    outlier_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    for feature in NUMERICAL_FEATURES:
        # Add outliers for some features
        if feature in ['hr', 'spo2', 'temp', 'sbp', 'dbp', 'map']:
            # Add more extreme outliers to vital signs
            outlier_values = np.random.choice([data[feature].min(), data[feature].max()],
                                             size=len(outlier_indices), replace=True)
            data.loc[outlier_indices, feature] = outlier_values

    # Generate text features
    nursing_note_templates = [
        "Patient alert, oriented x3. Complains of {symptom}. {intervention} provided. Awaiting {lab} results.",
        "Patient {symptom} with {severity} severity. {intervention} initiated. {lab} pending.",
        "Patient {symptom} and {symptom2}. {intervention} administered. {lab} results expected soon.",
        "Patient {symptom} and {symptom3}. {intervention} provided. {lab} pending.",
        "Patient {symptom}. {intervention} performed. {lab} results awaited."
    ]

    symptom_list = ["fever", "shortness of breath", "chest pain", "abdominal pain",
                   "headache", "dizziness", "fatigue", "nausea", "vomiting", "cough"]

    intervention_list = ["IV fluids", "oxygen therapy", "pain medication", "antibiotics",
                        "antipyretics", "antihypertensives", "bronchodilators", "antacids"]

    lab_list = ["CBC", "CMP", "Lactate", "CRP", "Troponin", "BNP", "Blood cultures",
               "Urinalysis", "Coagulation panel", "Liver function tests"]

    radiology_templates = [
        "No acute cardiopulmonary process.",
        "Bilateral infiltrates suggestive of pneumonia.",
        "Mild pulmonary edema.",
        "Normal chest x-ray.",
        "Right lower lobe consolidation.",
        "Pleural effusion.",
        "Cardiomegaly noted.",
        "No evidence of acute process.",
        "Mild emphysema.",
        "COPD exacerbation."
    ]

    history_templates = [
        "History of {condition} and {condition2}.",
        "Previously diagnosed with {condition}.",
        "History of {condition} and {condition3}.",
        "No significant past medical history.",
        "Patient with {condition} and {condition2}.",
        "History of {condition}, {condition2}, and {condition3}.",
        "Past medical history includes {condition}."
    ]

    conditions = ["hypertension", "diabetes mellitus", "copd", "asthma", "heart failure",
                  "chronic kidney disease", "cirrhosis", "pulmonary embolism",
                  "sepsis", "stroke"]

    for i in range(n_samples):
        # Generate nursing notes
        symptom = random.choice(symptom_list)
        symptom2 = random.choice(symptom_list)
        symptom3 = random.choice(symptom_list)
        intervention = random.choice(intervention_list)
        lab = random.choice(lab_list)
        severity = random.choice(["mild", "moderate", "severe"])

        template = random.choice(nursing_note_templates)
        note = template.format(symptom=symptom, symptom2=symptom2, symptom3=symptom3,
                              intervention=intervention, lab=lab, severity=severity)
        data.loc[i, 'nursing_notes'] = note

        # Generate radiology impressions
        impression = random.choice(radiology_templates)
        data.loc[i, 'radiology_impression'] = impression

        # Generate patient history
        condition = random.choice(conditions)
        condition2 = random.choice(conditions)
        condition3 = random.choice(conditions)

        template = random.choice(history_templates)
        history = template.format(condition=condition, condition2=condition2, condition3=condition3)
        data.loc[i, 'patient_history'] = history

    # Generate labels with noise
    # sepsis_risk: depends on hr, temp, wbc, lactate, crp
    data['sepsis_risk'] = 0
    sepsis_conditions = (data['hr'] > 100) & (data['temp'] > 38) & (data['wbc'] > 12) & (data['lactate'] > 2) & (data['crp'] > 5)
    data.loc[sepsis_conditions, 'sepsis_risk'] = 1

    # Add 5% label noise to sepsis_risk
    noise_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data.loc[noise_indices, 'sepsis_risk'] = 1 - data.loc[noise_indices, 'sepsis_risk']

    # cardiac_event: depends on troponin, bnp, sbp, chest pain mention in nursing note
    data['cardiac_event'] = 0

    # High troponin or BNP + low SBP suggests cardiac event
    cardiac_conditions = ((data['troponin'] > 0.5) | (data['bnp'] > 100)) & (data['sbp'] < 90)
    data.loc[cardiac_conditions, 'cardiac_event'] = 1

    # Add chest pain mention to nursing notes for more realistic cardiac events
    chest_pain_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    for idx in chest_pain_indices:
        if pd.isna(data.loc[idx, 'nursing_notes']):
            data.loc[idx, 'nursing_notes'] = "Patient chest pain. Oxygen therapy initiated. ECG pending."
        else:
            data.loc[idx, 'nursing_notes'] += " Patient chest pain. Oxygen therapy initiated. ECG pending."

    # Add cardiac event if chest pain is mentioned
    chest_pain_mention = data['nursing_notes'].str.contains('chest pain', case=False, na=False)
    data.loc[chest_pain_mention, 'cardiac_event'] = 1

    # Add 4% label noise to cardiac_event
    noise_indices = np.random.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    data.loc[noise_indices, 'cardiac_event'] = 1 - data.loc[noise_indices, 'cardiac_event']

    # stable: 1 if no risk; derived as NOT(sepsis OR cardiac) with 3% noise
    data['stable'] = 1 - (data['sepsis_risk'] | data['cardiac_event'])

    # Add 3% label noise to stable
    noise_indices = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    data.loc[noise_indices, 'stable'] = 1 - data.loc[noise_indices, 'stable']

    # Add missing values
    # MCAR missing values (8%)
    for feature in NUMERICAL_FEATURES:
        missing_indices = np.random.choice(n_samples, size=int(0.08 * n_samples), replace=False)
        data.loc[missing_indices, feature] = np.nan

    # MAR missing values for lactate (missing more often when sepsis_risk=1)
    lactate_missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    # Make missingness more likely when sepsis_risk=1
    lactate_sepsis_indices = data[data['sepsis_risk'] == 1].index
    lactate_sepsis_missing = np.random.choice(lactate_sepsis_indices,
                                             size=int(0.03 * len(lactate_sepsis_indices)),
                                             replace=False)
    data.loc[lactate_sepsis_missing, 'lactate'] = np.nan
    data.loc[lactate_missing_indices, 'lactate'] = np.nan

    # MNAR missing values for troponin (missing when extremely high)
    high_troponin_indices = data[data['troponin'] > data['troponin'].quantile(0.99)].index
    troponin_missing_indices = np.random.choice(high_troponin_indices,
                                               size=int(0.02 * len(high_troponin_indices)),
                                               replace=False)
    data.loc[troponin_missing_indices, 'troponin'] = np.nan

    # Text fields missing (1%)
    text_missing_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    for feature in TEXT_FEATURES:
        data.loc[text_missing_indices, feature] = np.nan

    # Ensure all features are in the correct order
    data = data[ALL_FEATURES]

    logger.info("Synthetic dataset generation complete!")
    return data

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Generate dataset
    df = generate_synthetic_data(10000)

    # Save to compressed CSV
    output_file = "data/raw_synthetic.csv.gz"
    df.to_csv(output_file, compression='gzip', index=False)

    logger.info(f"Dataset saved to {output_file}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Missing value counts:\n{df.isnull().sum()}")