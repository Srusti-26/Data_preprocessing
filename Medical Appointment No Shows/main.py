import pandas as pd
from pipeline import create_data_cleaning_pipeline  

# Load dataset
df = pd.read_csv('raw.csv')
print("Raw dataset loaded!")

# Define cleaning parameters
params = {
    'rename_columns': {
        'PatientId': 'patient_id',
        'AppointmentID': 'appointment_id',
        'Gender': 'gender',
        'ScheduledDay': 'scheduled_day',
        'AppointmentDay': 'appointment_day',
        'Age': 'age',
        'Neighbourhood': 'neighbourhood',
        'Scholarship': 'scholarship',
        'Hipertension': 'hypertension',
        'Diabetes': 'diabetes',
        'Alcoholism': 'alcoholism',
        'Handcap': 'handicap',
        'SMS_received': 'sms_received',
        'No-show': 'no_show'
    },
    'date_columns': ['scheduled_day', 'appointment_day'],
    'type_conversions': {
        'age': 'int',
        'scholarship': 'int',
        'sms_received': 'int'
    },
    'category_mappings': {
        'gender': {'M': 'Male', 'F': 'Female'},
        'no_show': {'Yes': 'NoShow', 'No': 'Show'}
    },
    'custom_fill_values': {
        'age': df['Age'].median()
    },
    'drop_threshold': 0.6
}

# Run cleaning
pipeline = create_data_cleaning_pipeline(**params)
df_cleaned = pipeline.fit_transform(df)

# Drop 'unnamed:_0' if it exists
if 'unnamed:_0' in df_cleaned.columns:
    df_cleaned.drop(columns=['unnamed:_0'], inplace=True)
    print("Dropped 'unnamed:_0' column.")

# Sort 
df_cleaned = df_cleaned.sort_values(by='appointment_day')

# Save cleaned data
df_cleaned.to_csv('cleaned.csv', index=False)
print("Cleaned dataset saved to 'cleaned.csv'")

# Summary
print("\n=== Cleaning Summary ===")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
print(f"Columns removed: {df.shape[1] - df_cleaned.shape[1]}")
print("\nMissing values:")
missing = df_cleaned.isnull().sum()
print(missing[missing > 0] if missing.any() > 0 else "None")
