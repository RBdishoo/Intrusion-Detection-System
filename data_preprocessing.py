import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler # NEW: Import StandardScaler

# --- Configuration ---
data_file_path = 'Monday-WorkingHours.pcap_ISCX.csv'

# --- Load the data ---
print(f"\nAttempting to load data from: {data_file_path}")
try:
    df = pd.read_csv(data_file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{data_file_path}' was not found.")
    print("Please ensure the path is correct and the file exists in that location.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- Data Cleaning Steps ---

# 1. Clean Column Names: Remove leading/trailing spaces
print("\n--- Cleaning Column Names ---")
# df.columns.tolist() before stripping to show original names
original_columns_for_display = df.columns.tolist()
df.columns = df.columns.str.strip()
print("Original Columns (first 5):", original_columns_for_display[:5], "...")
print("Cleaned Columns (first 5):", df.columns.tolist()[:5], "...")

# 2. Handle Infinite Values (Inf) and NaN values
print("\n--- Handling Infinite and NaN values ---")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_nan_count = df.isnull().sum().sum()
print(f"Initial NaN count after converting Inf: {initial_nan_count}")
df.fillna(0, inplace=True)
print("NaN values filled with 0.")
final_nan_count = df.isnull().sum().sum()
print(f"Final NaN count: {final_nan_count}")


# --- NEW: Feature Engineering and Selection ---

print("\n--- Feature Engineering and Selection ---")

# Identify the Label column (after cleaning names)
label_column = 'Label' # Now we can assume it's simply 'Label' after stripping spaces

# Identify columns to drop
# 'Fwd Header Length.1' is a duplicate of 'Fwd Header Length' in some CIC-IDS2017 datasets.
# 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp' are typically dropped for ML models
# that learn numerical patterns, as they are identifiers/timestamps, not behavioral features.
# You might keep them for forensic analysis, but not for training the anomaly detection core.
columns_to_drop = [
    'Fwd Header Length.1', # Known duplicate
    'Flow ID',             # Unique identifier, not a feature
    'Source IP',           # Specific to source, not general behavior
    'Destination IP',      # Specific to destination, not general behavior
    'Timestamp'            # Time of flow, not a behavioral feature
]

# Check if columns exist before dropping
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_columns_to_drop:
    print(f"Dropping columns: {existing_columns_to_drop}")
    df.drop(columns=existing_columns_to_drop, inplace=True)
else:
    print("No specified columns to drop were found in the DataFrame (they might have been handled already or are not in this specific file).")


# Identify constant columns (columns with only one unique value)
# These provide no information to a machine learning model.
constant_columns = [col for col in df.columns if df[col].nunique() == 1 and col != label_column]
if constant_columns:
    print(f"Dropping constant columns: {constant_columns}")
    df.drop(columns=constant_columns, inplace=True)
else:
    print("No constant columns found (excluding the Label column).")


# Separate features (X) and target (y)
# For unsupervised learning, X will be all numerical features.
# y will be the Label column, which we'll use for evaluation later.
X = df.drop(columns=[label_column])
y = df[label_column]
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")


# Feature Scaling (Standardization)
# Scale numerical features. This is important for distance-based algorithms.
print("\n--- Scaling Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Convert back to DataFrame for easier inspection
print("Features scaled successfully.")
print("First 5 rows of scaled features:")
print(X_scaled_df.head())


# --- Initial Data Exploration (Keep these for re-verification) ---
print("\n--- DataFrame Information After Cleaning & Feature Selection ---")
# Use the X_scaled_df for info, as this is our final feature set
X_scaled_df.info()

print("\n--- First 5 Rows After Cleaning & Feature Selection ---")
print(X_scaled_df.head())

# We don't need describe() on scaled data as much, but you can keep if curious.
# print("\n--- Basic Statistics After Cleaning & Feature Selection ---")
# print(X_scaled_df.describe())

print("\n--- Column Names After Cleaning & Feature Selection ---")
print(X_scaled_df.columns.tolist())

print("\n--- Value Counts for the 'Label' column (Original Labels) ---")
print(y.value_counts()) # Show the original label counts from 'y'
