import pandas as pd
from pipeline import create_data_cleaning_pipeline

# Load raw dataset
df = pd.read_csv('Customer Personality Analysis Dataset.csv')
print("Raw dataset loaded!")

# Define cleaning parameters
params = {
    'rename_columns': {
        'ID': 'customer_id',
        'Year_Birth': 'birth_year',
        'Education': 'education',
        'Marital_Status': 'marital_status',
        'Income': 'income',
        'Kidhome': 'kids_home',
        'Teenhome': 'teens_home',
        'Dt_Customer': 'signup_date',
        'Recency': 'recency',
        'MntWines': 'mnt_wines',
        'MntFruits': 'mnt_fruits',
        'MntMeatProducts': 'mnt_meat',
        'MntFishProducts': 'mnt_fish',
        'MntSweetProducts': 'mnt_sweets',
        'MntGoldProds': 'mnt_gold',
        'NumDealsPurchases': 'deal_purchases',
        'NumWebPurchases': 'web_purchases',
        'NumCatalogPurchases': 'catalog_purchases',
        'NumStorePurchases': 'store_purchases',
        'NumWebVisitsMonth': 'web_visits',
        'AcceptedCmp1': 'accepted_cmp1',
        'AcceptedCmp2': 'accepted_cmp2',
        'AcceptedCmp3': 'accepted_cmp3',
        'AcceptedCmp4': 'accepted_cmp4',
        'AcceptedCmp5': 'accepted_cmp5',
        'Response': 'response',
        'Complain': 'complain',
        'Z_CostContact': 'cost_contact',
        'Z_Revenue': 'revenue'
    },
    'date_columns': ['signup_date'],
    'type_conversions': {
        'birth_year': 'int',
        'income': 'float',
        'signup_date': 'datetime64[ns]'
    },
    'category_mappings': {
        'marital_status': {
            'Single': 'Single', 'Married': 'Married', 'Divorced': 'Divorced',
            'Widow': 'Widow', 'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'
        },
        'education': {
            'PhD': 'PhD', 'Master': 'Masters', 'Graduation': 'Graduate',
            '2n Cycle': 'Undergraduate', 'Basic': 'Basic'
        }
    },
    'custom_fill_values': {
        'income': df['Income'].median()
    },
    'drop_threshold': 0.7
}

# Run the cleaning pipeline
cleaning_pipeline = create_data_cleaning_pipeline(**params)
df_cleaned = cleaning_pipeline.fit_transform(df)

# Drop unnecessary index column if it exists
if 'unnamed:_0' in df_cleaned.columns:
    df_cleaned.drop(columns=['unnamed:_0'], inplace=True)
    print("Dropped 'unnamed:_0' column.")

# Sort by signup_date
df_cleaned = df_cleaned.sort_values(by='signup_date')

# Save the cleaned dataset
df_cleaned.to_csv('cleaned_customer_data_new.csv', index=False)
print("Cleaned dataset saved to 'cleaned_customer_data_new.csv'")

# Final Summary
print("\n=== Data Cleaning Summary ===")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
print(f"Columns removed: {df.shape[1] - df_cleaned.shape[1]}")
print("\nRemaining missing values:")
missing = df_cleaned.isnull().sum()
print(missing[missing > 0] if missing.any() > 0 else "None")
print("\nData cleaning completed successfully!")