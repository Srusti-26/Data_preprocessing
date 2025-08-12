# Data Cleaning Pipeline Project

This project contains a reusable data preprocessing pipeline built with Python and scikit-learn. It is designed to clean, standardize, and prepare various real-world datasets for analysis or machine learning tasks.
##  Features

- Modular pipeline using custom transformers
- Handles:
  - Column renaming
  - Missing values
  - Duplicates
  - Outliers
  - Data type conversion
  - Text and category standardization
  - Date formatting
- Scalable design to handle multiple datasets
##  Folder Structure
```text
data_preprocessor/
├── pipeline.py # Main cleaning logic
├── datasets/
│ ├── customer/
│ │ ├── raw.csv
│ │ ├── cleaned.csv
│ │ └── main.py
│ └── medical_no_show/
│ ├── raw.csv
│ ├── cleaned.csv
│ └── main.py
├── README.md
└── .gitignore
```

##  Datasets Processed

1. **Customer Personality Analysis**
   - Source: [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
   - Preprocessing includes renaming, missing value imputation, and outlier handling.

2. **Medical Appointment No-Show**
   - Source: [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
   - Cleaned and standardized to improve usability for modeling.



##  How to Use

1. Place your dataset in a new folder inside `datasets/`
2. Create a `main.py` script using `pipeline.py`
3. Run `main.py` to generate the cleaned file



##  Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install them via:


pip install -r requirements.txt
 Author
Created by Srusti 
