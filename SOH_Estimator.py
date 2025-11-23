import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Constants
# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "soh_training_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "soh_model.pkl")
OUTPUT_DIR = BASE_DIR

def load_data(csv_path):
    """Load and preprocess data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples.")
    return df

def train_model(df):
    """Train Random Forest model for SOH prediction."""
    # Features: Impedance magnitude and phase at different frequencies
    # Targets: SOH
    
    # Identify feature columns (all columns except SOH)
    feature_cols = [c for c in df.columns if c != 'SOH']
    X = df[feature_cols]
    y = df['SOH']
    
    print(f"Features: {feature_cols}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance:")
    for f in range(X.shape[1]):
        print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"SOH Prediction (RMSE={rmse:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "soh_prediction_accuracy.png"))
    print(f"Accuracy plot saved to {os.path.join(OUTPUT_DIR, 'soh_prediction_accuracy.png')}")
    
    return model

def save_model(model, path):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def predict_soh(model, impedance_features):
    """Predict SOH for new impedance measurements."""
    # impedance_features should be a dict or dataframe with same columns as training
    prediction = model.predict([impedance_features])
    return prediction[0]

def main():
    print("SOH ESTIMATOR TRAINING")
    print("="*50)
    
    try:
        # 1. Load Data
        df = load_data(DATA_PATH)
        
        # 2. Train Model
        model = train_model(df)
        
        # 3. Save Model
        save_model(model, MODEL_PATH)
        
        # 4. Validation Test (Single Prediction)
        print("\nValidation Test:")
        test_sample = df.iloc[0]
        features = test_sample.drop('SOH').values
        actual = test_sample['SOH']
        predicted = model.predict([features])[0]
        
        print(f"Actual SOH: {actual:.4f}")
        print(f"Predicted SOH: {predicted:.4f}")
        print(f"Error: {abs(actual - predicted):.4f}")
        
        print("\n[OK] SOH Estimator implementation complete.")
        
    except Exception as e:
        print(f"\n[ERR] Error: {e}")

if __name__ == "__main__":
    main()
