import pandas as pd
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Find the dataset folder
DATA_DIR = BASE_DIR / "data" / "raw"

# Load customer data
customers = pd.read_csv(
    DATA_DIR / "olist_customers_dataset.csv"
)

# Display the first five rows
print("First 5 customer records:")
print(customers.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(customers.shape)

# Display column names
print("\nColumn names:")
print(customers.columns.tolist())

# Display dataset information
print("\nDataset information:")
customers.info()
