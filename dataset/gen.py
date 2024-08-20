import numpy as np
import pandas as pd

np.random.seed(42)

gender = np.random.choice(['Male', 'Female'], size=1500)

diabetes_type = np.random.choice(['Type 1', 'Type 2'], size=1500)

retinopathy_levels = ['No', 'Mid Non-life Proliferative Diabetic Retinopathy', 
                      'Moderate Non-proliferative Diabetic Retinopathy']
retinopathy_probabilities = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
retinopathy_status = np.random.choice(retinopathy_levels, size=1500)

retinopathy_prob_map = {level: prob for level, prob in zip(retinopathy_levels, retinopathy_probabilities)}

retinopathy_prob = np.array([retinopathy_prob_map[level] for level in retinopathy_status])

diagnosis_year = np.random.randint(1990, 2022, size=1500)

systolic_bp = np.random.randint(90, 180, size=1500)
diastolic_bp = np.random.randint(60, 110, size=1500)

hba1c = np.random.randint(50, 130, size=1500)

avg_glucose = np.random.randint(80, 300, size=1500)

data = pd.DataFrame({
    'Gender': gender,
    'Diabetes Type': diabetes_type,
    'Retinopathy Status': retinopathy_status,
    'Retinopathy Probability': retinopathy_prob,
    'Diagnosis Year': diagnosis_year,
    'Systolic BP': systolic_bp,
    'Diastolic BP': diastolic_bp,
    'HbA1c (mmol/mol)': hba1c,
    'Estimated Avg Glucose (mg/dL)': avg_glucose
})

data.to_csv('diabetes_retinopathy_dataset.csv', index=False)
