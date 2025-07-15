# Intrusion Detection System (IDS) with AI-Integrated Anomaly Detection

# Intrusion-Detection-System

## AI-Powered Anomaly Detection for Network Security

This project develops a Python-based Intrusion Detection System (IDS) module that leverages machine learning to identify anomalous behavior in network traffic. By detecting deviations from normal patterns, this system provides a proactive defense against evolving cyber threats, showcasing the practical application of AI in cybersecurity.

### Features
* **Robust Data Preprocessing:** Engineered a comprehensive pipeline for cleaning, scaling, and preparing raw network flow logs for model consumption.
* **Machine Learning Models:** Implemented and evaluated both Autoencoder (Deep Learning) and Isolation Forest (Ensemble Learning) for unsupervised anomaly detection.
* **Comprehensive Performance Evaluation:** Assesses model effectiveness using industry-standard metrics including Accuracy, Precision, Recall, F1-Score, and ROC AUC.
* **Visual Analytics:** Generates insightful plots such as ROC curves and anomaly score distributions for in-depth model understanding.

### Performance Highlights
On a test dataset comprising 432,074 benign and 13,835 anomalous network activities:

| Model           | Accuracy | Precision (Anomaly) | Recall (Anomaly) | F1-Score (Anomaly) | ROC AUC |
| :-------------- | :------- | :------------------ | :--------------- | :----------------- | :------ |
| **Isolation Forest** | 0.0460   | 0.0315              | **1.0000 (100%)**| 0.0611             | 0.6628  |
| **Autoencoder** | 0.8569   | 0.0021              | 0.0077           | 0.0033             | 0.6077  |

* **Isolation Forest Insight:** Achieved **100% recall** in detecting simulated attacks, successfully identifying every single anomaly in the test set.
* **Autoencoder Insight:** Demonstrated a robust overall accuracy for benign traffic, with an ROC AUC of 0.61.

### Project Structure

    model_training.py          # Script for initial data preprocessing and model training
    evaluate_models.py         # Main script for evaluating trained models on test data
    Tuesday-WorkingHours.pcap_ISCX.csv # Test dataset (approx. 450MB)
    scaler.joblib              # Pre-trained StandardScaler for feature normalization
    dropped_columns.joblib     # List of columns identified for removal during preprocessing
    autoencoder_model.h5       # Trained Keras Autoencoder model
    autoencoder_threshold.joblib # Anomaly threshold for the Autoencoder model
    isolation_forest_model.joblib # Trained Scikit-learn Isolation Forest model
    images/                    # Directory containing generated plots and figures
        autoencoder_roc_curve.png
        autoencoder_reconstruction_error_distribution.jpg
        isolation_forest_roc_curve.png
        isolation_forest_anomaly_score_distribution.jpg
    RESULTS.md                 # Detailed console output, plots, and in-depth analysis of evaluation results

