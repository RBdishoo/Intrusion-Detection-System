import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os

print("--- Anomaly Detection Model Evaluation Script ---")

# Define paths for test data and saved models/scaler
TEST_DATA_PATH = 'Tuesday-WorkingHours.pcap_ISCX.csv'
SCALER_PATH = 'scaler.joblib'
DROPPED_COLUMNS_PATH = 'dropped_columns.joblib'
AUTOENCODER_MODEL_PATH = 'autoencoder_model.h5'
AUTOENCODER_THRESHOLD_PATH = 'autoencoder_threshold.joblib'
ISOLATION_FOREST_MODEL_PATH = 'isolation_forest_model.joblib'

# --- 1. Load Data ---
print(f"Attempting to load test data from: {TEST_DATA_PATH}")
try:
    df_test = pd.read_csv(TEST_DATA_PATH)
    print("Test data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The test data file {TEST_DATA_PATH} was not found. Please ensure it's in the correct directory.")
    exit()

# --- 2. Apply Identical Preprocessing ---
print("\n--- Preprocessing Test Data ---")

# Clean column names (MUST be identical)
original_columns = df_test.columns
new_columns = []
for col in original_columns:
    new_col = col.strip().replace(' ', '_').replace('/', '_').replace('.', '').replace('-', '_')
    new_columns.append(new_col)
df_test.columns = new_columns
print("Column names cleaned.")

# Handle infinite and NaN values (MUST be identical)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Infinite values replaced with NaN.")
df_test.fillna(0, inplace=True)
print("NaN values filled with 0.")

# Separate features and labels
if 'Label' in df_test.columns:
    true_labels = df_test['Label']
    features_df_test = df_test.drop(columns=['Label']).copy()
    print("Labels separated from features.")
else:
    print("Error: 'Label' column not found in test data. Cannot evaluate without true labels.")
    exit()

# Load dropped columns list and apply
print(f"Attempting to load dropped columns list from: {DROPPED_COLUMNS_PATH}")
try:
    columns_to_drop_from_features = joblib.load(DROPPED_COLUMNS_PATH)
    print("Dropped columns list loaded successfully!")
except FileNotFoundError:
    print(f"Error: {DROPPED_COLUMNS_PATH} not found. Ensure preprocessing_training_data.py was run.")
    exit()

existing_cols_to_drop = [col for col in columns_to_drop_from_features if col in features_df_test.columns]
if existing_cols_to_drop:
    features_df_test.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Dropped columns from test data: {existing_cols_to_drop}")
else:
    print("No specified columns from dropped_columns.joblib found to drop in test data.")

# Ensure all feature columns are numeric, coerce errors, fill new NaNs (Crucial after dropping)
for col in features_df_test.columns:
    features_df_test[col] = pd.to_numeric(features_df_test[col], errors='coerce').fillna(0)

print(f"Test data features shape after preprocessing: {features_df_test.shape}")

# Load the pre-fitted scaler
print(f"Attempting to load scaler from: {SCALER_PATH}")
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully!")
except FileNotFoundError:
    print(f"Error: {SCALER_PATH} not found. Ensure preprocessing_training_data.py was run.")
    exit()

# Scale the test data using the loaded scaler
X_test_scaled = scaler.transform(features_df_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_df_test.columns, index=features_df_test.index)
print("Test data scaled successfully!")
print("Shape of scaled test data:", X_test_scaled_df.shape)

# Convert true labels to binary (1 for BENIGN/Normal, 0 for ATTACK/Anomaly)
binary_true_labels = true_labels.apply(lambda x: 1 if x == 'BENIGN' else 0)
print("\nTrue Labels distribution in Test Data:")
print(binary_true_labels.value_counts())

# --- 3. Evaluate Autoencoder ---
print("\n--- Evaluating Autoencoder Model ---")
try:
    autoencoder = keras.models.load_model(AUTOENCODER_MODEL_PATH)
    print(f"Autoencoder model loaded from: {AUTOENCODER_MODEL_PATH}")

    # Load the calculated threshold
    print(f"Attempting to load Autoencoder threshold from: {AUTOENCODER_THRESHOLD_PATH}")
    try:
        autoencoder_threshold = joblib.load(AUTOENCODER_THRESHOLD_PATH)
        print(f"Autoencoder threshold loaded successfully: {autoencoder_threshold:.4f}")
    except FileNotFoundError:
        print(f"Error: {AUTOENCODER_THRESHOLD_PATH} not found. Run autoencoder_training.py first to generate it.")
        exit()

    # Get reconstructions
    reconstructions = autoencoder.predict(X_test_scaled_df)
    reconstruction_errors = np.mean(np.square(X_test_scaled_df - reconstructions), axis=1)

    # Classify predictions
    # If reconstruction_error > threshold, it's an anomaly (0); else it's normal (1)
    ae_binary_predictions = (reconstruction_errors <= autoencoder_threshold).astype(int) # 1 if normal, 0 if anomaly

    # Evaluate metrics
    print("\n--- Autoencoder Performance Metrics ---")
    print(f"Accuracy: {accuracy_score(binary_true_labels, ae_binary_predictions):.4f}")
    print(f"Precision: {precision_score(binary_true_labels, ae_binary_predictions, pos_label=0):.4f} (Anomalies)") # pos_label=0 for anomalies
    print(f"Recall: {recall_score(binary_true_labels, ae_binary_predictions, pos_label=0):.4f} (Anomalies)")
    print(f"F1-Score: {f1_score(binary_true_labels, ae_binary_predictions, pos_label=0):.4f} (Anomalies)")
    print("Confusion Matrix:\n", confusion_matrix(binary_true_labels, ae_binary_predictions))

    roc_auc = roc_auc_score(binary_true_labels, reconstruction_errors)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(binary_true_labels, reconstruction_errors, pos_label=0) # pos_label=0 for anomalies
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Autoencoder')
    plt.legend(loc="lower right")
    plt.show()

except Exception as e:
    print(f"Error during Autoencoder evaluation: {e}")


# --- 4. Evaluate Isolation Forest ---
print("\n--- Evaluating Isolation Forest Model ---")
try:
    isolation_forest_model = joblib.load(ISOLATION_FOREST_MODEL_PATH)
    print(f"Isolation Forest model loaded from: {ISOLATION_FOREST_MODEL_PATH}")

    if_anomaly_scores = isolation_forest_model.decision_function(X_test_scaled_df)

    # --- Manually calculate Isolation Forest threshold ---
    # Note: Lower scores mean more anomalous for Isolation Forest's decision_function
    # To improve precision, we need to be more selective, i.e., increase the threshold score (make it less negative).
    # This means selecting a *lower* percentile of the original if_anomaly_scores.
    # We are aiming to balance Recall (catching anomalies) with Precision (reducing false alarms).
    if_threshold = np.percentile(if_anomaly_scores, 1.5) # Changed to 1.5th percentile

    # Classify predictions based on this manual threshold
    # If score < if_threshold, it's an anomaly (0); else it's normal (1)
    if_binary_predictions = (if_anomaly_scores < if_threshold).astype(int) # 0 if anomaly, 1 if normal

    print(f"Calculated Isolation Forest anomaly threshold: {if_threshold:.4f}")

    # Evaluate metrics
    print("\n--- Isolation Forest Performance Metrics (with custom threshold) ---")
    print(f"Accuracy: {accuracy_score(binary_true_labels, if_binary_predictions):.4f}")
    print(f"Precision: {precision_score(binary_true_labels, if_binary_predictions, pos_label=0):.4f} (Anomalies)")
    print(f"Recall: {recall_score(binary_true_labels, if_binary_predictions, pos_label=0):.4f} (Anomalies)")
    print(f"F1-Score: {f1_score(binary_true_labels, if_binary_predictions, pos_label=0):.4f} (Anomalies)")
    print("Confusion Matrix:\n", confusion_matrix(binary_true_labels, if_binary_predictions))

    # Correct ROC AUC calculation for Isolation Forest:
    # Since pos_label=0 (anomaly) and lower decision_function scores mean more anomalous,
    # we use if_anomaly_scores directly.
    roc_auc_if = roc_auc_score(binary_true_labels, if_anomaly_scores) # This line is correct
    print(f"ROC AUC: {roc_auc_if:.4f}")

    # Plot ROC curve
    # For plotting, we want higher scores to be to the right (more anomalous), so use -if_anomaly_scores for X-axis.
    # IMPORTANT: roc_curve does NOT take 'pos_label' argument. It assumes positive class is 1.
    # To make it work with anomaly=0 and higher scores = more anomalous, we pass -if_anomaly_scores.
    fpr_if, tpr_if, thresholds_if = roc_curve(binary_true_labels, -if_anomaly_scores) 
    plt.figure()
    plt.plot(fpr_if, tpr_if, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_if:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Isolation Forest')
    plt.legend(loc="lower right")
    plt.show()

    # Plotting Isolation Forest anomaly score distributions
    plt.figure(figsize=(12, 7))
    # Plot benign scores (true_label == 1)
    # Negate scores so higher means more anomalous for visualization consistency
    plt.hist(-if_anomaly_scores[binary_true_labels == 1], bins=50, alpha=0.5, label='Benign Traffic Scores (Higher = More Anomalous)', color='blue', density=True)
    # Plot anomaly scores (true_label == 0)
    plt.hist(-if_anomaly_scores[binary_true_labels == 0], bins=50, alpha=0.5, label='Anomaly Traffic Scores (Higher = More Anomalous)', color='red', density=True)
    plt.title('Distribution of Isolation Forest Anomaly Scores for Benign vs. Anomaly')
    plt.xlabel('Anomaly Score (Higher = More Anomalous)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Error during Isolation Forest evaluation: {e}")


print("\n--- Anomaly Detection Model Evaluation Script Finished ---")
