import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create lists of possible values
first_names = ['John', 'Michael', 'Robert', 'David', 'James', 'William', 'Richard',
               'Thomas', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan']
last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller',
              'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris']
diagnoses = ['Benign Prostatic Hyperplasia (BPH)', 'Urinary Tract Infection (UTI)', 'Kidney Stones', 'Prostate Cancer', 'Overactive Bladder',
             'Urinary Incontinence', 'Interstitial Cystitis', 'Neurogenic Bladder', 'Testicular Cancer', 'Erectile Dysfunction']
medications = ['Tamsulosin', 'Finasteride', 'Ciprofloxacin', 'Oxybutynin',
               'Mirabegron', 'Sildenafil', 'Tadalafil', 'Leuprolide', 'Hydrochlorothiazide', 'None']
symptoms = ['Urinary frequency', 'Nocturia', 'Hematuria', 'Dysuria', 'Urinary retention',
            'Flank pain', 'Lower abdominal pain', 'Erectile dysfunction', 'Scrotal swelling', 'None']
procedures = ['Cystoscopy', 'Transurethral Resection of Prostate (TURP)', 'Lithotripsy',
              'Radical Prostatectomy', 'Nephrectomy', 'Ureteroscopy', 'Vasectomy', 'Orchiopexy', 'None', 'None']

# Generate random dates within the last year


def random_date():
    today = datetime.now()
    days_back = random.randint(1, 365)
    return (today - timedelta(days=days_back)).strftime('%Y-%m-%d')


# Create data
data = []
for i in range(10):
    patient_id = f"URO-{2023000 + i}"
    age = random.randint(35, 85)
    gender = random.choice(['Male', 'Female'])
    diagnosis = random.choice(diagnoses)
    medication = random.choice(medications)
    symptom = random.choice(symptoms)
    procedure = random.choice(procedures)
    psa_level = round(random.uniform(0.5, 15.0),
                      2) if gender == 'Male' and age > 45 else None
    visit_date = random_date()
    follow_up = random.choice([True, False])

    # Clinical notes - slightly more detailed description
    notes = f"Patient presents with {symptom.lower() if symptom != 'None' else 'routine follow-up'}. "

    if diagnosis == 'Benign Prostatic Hyperplasia (BPH)':
        notes += "Enlarged prostate on digital rectal exam. Patient reports difficulty initiating urination."
    elif diagnosis == 'Urinary Tract Infection (UTI)':
        notes += "Urine culture positive. Prescribed antibiotics for 7 days."
    elif diagnosis == 'Kidney Stones':
        notes += "5mm stone visualized on CT scan in left ureter. Advised increased fluid intake."
    elif diagnosis == 'Prostate Cancer':
        notes += f"PSA elevated at {psa_level}. Gleason score 3+4. Discussing treatment options."
    elif diagnosis == 'Overactive Bladder':
        notes += "Urodynamic studies show detrusor overactivity. Started on anticholinergic therapy."
    else:
        notes += "Treatment plan discussed. Patient understanding and in agreement with proposed management."

    data.append({
        'Patient_ID': patient_id,
        'Age': age,
        'Gender': gender,
        'Diagnosis': diagnosis,
        'Symptoms': symptom,
        'Medication': medication,
        'Procedure': procedure,
        'PSA_Level': psa_level,
        'Visit_Date': visit_date,
        'Follow_Up_Required': follow_up,
        'Clinical_Notes': notes
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('urology_patient_data.csv', index=False)

print("Fake urology patient data CSV created successfully:")
print(df)
