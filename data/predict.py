import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def predict_stability(hr, resp_rate, spo2, temp, sbp, dbp, map, glucose, lactate, creatinine, bun, wbc, plt, hgb, na, k, cl, ca, mg, p, albumin, bilirubin, ast, alt, alp, ggt, ptt, inr, troponin, bnp, procalcitonin, crp, d_dimer, fibrinogen, ph, pco2, po2, hco3, base_excess, urine_output, sepsis_risk, cardiac_event):
    """
    Predict patient stability using the trained model
    """
    # Load the trained model and scaler
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Create a DataFrame with the input data
    data = pd.DataFrame({
        'hr': [hr],
        'resp_rate': [resp_rate],
        'spo2': [spo2],
        'temp': [temp],
        'sbp': [sbp],
        'dbp': [dbp],
        'map': [map],
        'glucose': [glucose],
        'lactate': [lactate],
        'creatinine': [creatinine],
        'bun': [bun],
        'wbc': [wbc],
        'plt': [plt],
        'hgb': [hgb],
        'na': [na],
        'k': [k],
        'cl': [cl],
        'ca': [ca],
        'mg': [mg],
        'p': [p],
        'albumin': [albumin],
        'bilirubin': [bilirubin],
        'ast': [ast],
        'alt': [alt],
        'alp': [alp],
        'ggt': [ggt],
        'ptt': [ptt],
        'inr': [inr],
        'troponin': [troponin],
        'bnp': [bnp],
        'procalcitonin': [procalcitonin],
        'crp': [crp],
        'd_dimer': [d_dimer],
        'fibrinogen': [fibrinogen],
        'ph': [ph],
        'pco2': [pco2],
        'po2': [po2],
        'hco3': [hco3],
        'base_excess': [base_excess],
        'urine_output': [urine_output],
        'sepsis_risk': [sepsis_risk],
        'cardiac_event': [cardiac_event]
    })

    # Fill missing values with median (same approach used during training)
    data = data.fillna(data.median())

    # Scale the features
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0]

    return {
        'prediction': int(prediction),
        'probability': {
            'stable': float(probability[0]),
            'unstable': float(probability[1])
        }
    }

# Example usage
if __name__ == "__main__":
    # Example patient data (this is just sample data for demonstration)
    example_patient = {
        'hr': 80,
        'resp_rate': 16,
        'spo2': 98,
        'temp': 37.0,
        'sbp': 120,
        'dbp': 80,
        'map': 95,
        'glucose': 100,
        'lactate': 1.0,
        'creatinine': 1.0,
        'bun': 15,
        'wbc': 8000,
        'plt': 250000,
        'hgb': 14,
        'na': 140,
        'k': 4.0,
        'cl': 100,
        'ca': 9.0,
        'mg': 2.0,
        'p': 4.0,
        'albumin': 4.0,
        'bilirubin': 1.0,
        'ast': 20,
        'alt': 15,
        'alp': 80,
        'ggt': 5,
        'ptt': 25,
        'inr': 1.0,
        'troponin': 0.01,
        'bnp': 50,
        'procalcitonin': 0.1,
        'crp': 5,
        'd_dimer': 200,
        'fibrinogen': 300,
        'ph': 7.4,
        'pco2': 40,
        'po2': 80,
        'hco3': 24,
        'base_excess': 0,
        'urine_output': 1000,
        'sepsis_risk': 0,
        'cardiac_event': 0
    }

    result = predict_stability(**example_patient)
    print("Patient Stability Prediction:")
    print(f"Prediction: {'Stable' if result['prediction'] == 1 else 'Unstable'}")
    print(f"Probability of Stable: {result['probability']['stable']:.4f}")
    print(f"Probability of Unstable: {result['probability']['unstable']:.4f}")