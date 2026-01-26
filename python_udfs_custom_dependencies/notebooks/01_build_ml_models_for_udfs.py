# Databricks notebook source
# MAGIC %md
# MAGIC # Build ML Models for Unity Catalog UDFs
# MAGIC
# MAGIC **Author**: Prasad Kona  
# MAGIC **Last Updated**: January 26, 2026  
# MAGIC **Notebook**: 01 - Build ML Models for UDFs
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC 1. Training two scikit-learn models on NYC taxi data:
# MAGIC    - **Regression Model**: Predict trip fare amount
# MAGIC    - **Classification Model**: Predict tip category (low/medium/high)
# MAGIC 2. Packaging models as Python wheel files
# MAGIC 3. Uploading wheels to Unity Catalog volumes for use in UDFs
# MAGIC
# MAGIC **Dataset**: NYC Taxi Trip data (available in Databricks samples)
# MAGIC
# MAGIC **Requirements**:
# MAGIC - Databricks Runtime 18.1 or above
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Permissions: CREATE VOLUME, READ/WRITE VOLUME

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
from datetime import datetime
import os
import shutil

# COMMAND ----------

# Configuration - Update these based on your Unity Catalog setup
CATALOG = "main"  # Replace with your catalog name
SCHEMA = "default"  # Replace with your schema name
VOLUME = "ml_models"  # Volume name to create/use
FOLDER_NAME = "taxi_models"  # Folder within volume to organize wheel files

# Package configuration
PACKAGE_VERSION = "1.0.0"
REGRESSION_PACKAGE_NAME = "nyc_taxi_fare_predictor"
CLASSIFICATION_PACKAGE_NAME = "nyc_taxi_tip_classifier"

# Display configuration
print("=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"Catalog:  {CATALOG}")
print(f"Schema:   {SCHEMA}")
print(f"Volume:   {VOLUME}")
print(f"Folder:   {FOLDER_NAME}")
print(f"Full UC Path: /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FOLDER_NAME}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and Explore NYC Taxi Data

# COMMAND ----------

# Load sample NYC taxi data from Databricks datasets
# Using the samples.nyctaxi.trips table that's available in most Databricks workspaces

# If the table doesn't exist, we'll create sample data
try:
    df = spark.table("samples.nyctaxi.trips").limit(100000).toPandas()
    print(f"Loaded {len(df)} records from samples.nyctaxi.trips")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Handle different column naming conventions in the sample table
    # Common variations: passenger_count vs passengerCount vs num_passengers
    if 'passenger_count' not in df.columns:
        if 'passengerCount' in df.columns:
            df['passenger_count'] = df['passengerCount']
        elif 'num_passengers' in df.columns:
            df['passenger_count'] = df['num_passengers']
        else:
            # Default to 1 passenger if column doesn't exist
            df['passenger_count'] = 1
            print("⚠️  'passenger_count' column not found, using default value of 1")
    
    # Handle fare_amount variations
    if 'fare_amount' not in df.columns:
        if 'fareAmount' in df.columns:
            df['fare_amount'] = df['fareAmount']
        elif 'total_amount' in df.columns:
            df['fare_amount'] = df['total_amount']
        else:
            raise ValueError("Cannot find fare amount column")
    
    # Handle trip_distance variations
    if 'trip_distance' not in df.columns:
        if 'tripDistance' in df.columns:
            df['trip_distance'] = df['tripDistance']
        else:
            raise ValueError("Cannot find trip distance column")
    
    # Handle tip_amount variations
    if 'tip_amount' not in df.columns:
        if 'tipAmount' in df.columns:
            df['tip_amount'] = df['tipAmount']
        else:
            df['tip_amount'] = 0
            print("⚠️  'tip_amount' column not found, using default value of 0")
    
    # Extract time-based features if timestamp columns exist
    if 'tpep_pickup_datetime' in df.columns:
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
        df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
        print("✓ Extracted time features from tpep_pickup_datetime")
    elif 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour_of_day'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        print("✓ Extracted time features from pickup_datetime")
    else:
        # Use defaults if no timestamp column
        df['hour_of_day'] = np.random.randint(0, 24, len(df))
        df['day_of_week'] = np.random.randint(0, 7, len(df))
        print("⚠️  No timestamp column found, using random time features")
    
except Exception as e:
    print(f"Could not load samples.nyctaxi.trips: {e}")
    print("Creating synthetic NYC taxi data for demonstration...")
    # Create synthetic data if sample table not available
    np.random.seed(42)
    n_samples = 50000
    
    df = pd.DataFrame({
        'trip_distance': np.random.exponential(3, n_samples),
        'pickup_zip': np.random.randint(10001, 10280, n_samples),
        'dropoff_zip': np.random.randint(10001, 10280, n_samples),
        'fare_amount': np.random.uniform(5, 100, n_samples),
        'tip_amount': np.random.uniform(0, 20, n_samples),
        'passenger_count': np.random.randint(1, 7, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
    })
    
    # Make fare somewhat correlated with distance
    df['fare_amount'] = 2.5 + df['trip_distance'] * 3.5 + np.random.normal(0, 2, n_samples)
    df['fare_amount'] = df['fare_amount'].clip(2.5, 200)
    
    # Make tip somewhat correlated with fare
    df['tip_amount'] = df['fare_amount'] * 0.15 + np.random.normal(0, 1, n_samples)
    df['tip_amount'] = df['tip_amount'].clip(0, None)

# Display basic statistics
print("\nDataset shape:", df.shape)
print("\nRequired columns present:")
print(f"  ✓ trip_distance: {df['trip_distance'].notna().sum()} non-null values")
print(f"  ✓ fare_amount: {df['fare_amount'].notna().sum()} non-null values")
print(f"  ✓ passenger_count: {df['passenger_count'].notna().sum()} non-null values")
print(f"  ✓ tip_amount: {df['tip_amount'].notna().sum()} non-null values")
print(f"  ✓ hour_of_day: {df['hour_of_day'].notna().sum()} non-null values")
print(f"  ✓ day_of_week: {df['day_of_week'].notna().sum()} non-null values")
print("\nFirst few rows:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

def engineer_features(df):
    """
    Create features for model training
    Handles missing columns and various data quality issues
    """
    df_features = df.copy()
    
    # Ensure all required columns exist
    required_columns = ['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount', 'hour_of_day', 'day_of_week']
    for col in required_columns:
        if col not in df_features.columns:
            if col == 'trip_distance':
                df_features[col] = 5.0  # Default 5 miles
            elif col == 'fare_amount':
                df_features[col] = 15.0  # Default $15
            elif col == 'passenger_count':
                df_features[col] = 1  # Default 1 passenger
            elif col == 'tip_amount':
                df_features[col] = 0.0  # Default no tip
            elif col == 'hour_of_day':
                df_features[col] = 12  # Default noon
            elif col == 'day_of_week':
                df_features[col] = 3  # Default Wednesday
            print(f"⚠️  Added missing column '{col}' with default values")
    
    # Handle missing values
    df_features = df_features.fillna({
        'trip_distance': df_features['trip_distance'].median(),
        'passenger_count': 1,
        'tip_amount': 0,
        'fare_amount': df_features['fare_amount'].median()
    })
    
    # Ensure passenger_count is numeric
    df_features['passenger_count'] = pd.to_numeric(df_features['passenger_count'], errors='coerce').fillna(1)
    
    # Remove outliers and invalid data
    df_features = df_features[
        (df_features['trip_distance'] > 0) & 
        (df_features['trip_distance'] < 100) &
        (df_features['fare_amount'] > 0) &
        (df_features['fare_amount'] < 500) &
        (df_features['passenger_count'] > 0) &
        (df_features['passenger_count'] <= 6)
    ]
    
    # Ensure time features are in valid ranges
    df_features['hour_of_day'] = df_features['hour_of_day'].clip(0, 23)
    df_features['day_of_week'] = df_features['day_of_week'].clip(0, 6)
    
    # Create categorical features
    df_features['is_rush_hour'] = df_features['hour_of_day'].apply(
        lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
    )
    df_features['is_weekend'] = df_features['day_of_week'].apply(
        lambda x: 1 if x >= 5 else 0
    )
    
    # Create tip percentage for classification target
    df_features['tip_percentage'] = (
        df_features['tip_amount'] / df_features['fare_amount'] * 100
    ).fillna(0).clip(0, 100)  # Clip to reasonable range
    
    # Analyze tip percentage distribution
    print(f"ℹ️  Tip percentage statistics:")
    print(f"   - Min: {df_features['tip_percentage'].min():.2f}%")
    print(f"   - Median: {df_features['tip_percentage'].median():.2f}%")
    print(f"   - Max: {df_features['tip_percentage'].max():.2f}%")
    print(f"   - % of zero tips: {(df_features['tip_percentage'] == 0).sum() / len(df_features) * 100:.1f}%")
    
    # Check if we have enough variation in tip percentages
    tip_pct_25 = df_features['tip_percentage'].quantile(0.33)
    tip_pct_75 = df_features['tip_percentage'].quantile(0.67)
    
    print(f"\n   - 33rd percentile: {tip_pct_25:.2f}%")
    print(f"   - 67th percentile: {tip_pct_75:.2f}%")
    
    # Handle case where percentiles are too similar (e.g., many zero tips)
    if tip_pct_25 == tip_pct_75 or (tip_pct_75 - tip_pct_25) < 0.01:
        print(f"\n⚠️  Warning: Insufficient variation in tip percentages")
        print(f"   Using fixed boundaries instead of percentiles")
        
        # Use fixed, well-separated boundaries
        # Based on common NYC taxi tipping patterns
        tip_pct_25 = 5.0   # Low: 0-5%
        tip_pct_75 = 15.0  # Medium: 5-15%, High: >15%
        
        print(f"   - Low tip boundary: {tip_pct_25:.1f}%")
        print(f"   - High tip boundary: {tip_pct_75:.1f}%")
    
    # Create tip category with guaranteed unique bins
    # Use duplicates='drop' to handle edge case of identical boundaries
    df_features['tip_category'] = pd.cut(
        df_features['tip_percentage'],
        bins=[-np.inf, tip_pct_25, tip_pct_75, np.inf],
        labels=[0, 1, 2],
        duplicates='drop'  # Drop duplicate bin edges if they exist
    )
    
    # Handle any NaN values from pd.cut (shouldn't happen but be safe)
    if df_features['tip_category'].isna().any():
        print(f"\n⚠️  Warning: Found {df_features['tip_category'].isna().sum()} NaN values in tip_category")
        print(f"   Filling with category 0 (Low)")
        df_features['tip_category'] = df_features['tip_category'].fillna(0)
    
    # Convert to int
    df_features['tip_category'] = df_features['tip_category'].astype(int)
    
    # Show category distribution
    category_dist = df_features['tip_category'].value_counts().sort_index()
    print(f"\nℹ️  Tip category distribution:")
    for cat, count in category_dist.items():
        pct = (count / len(df_features)) * 100
        cat_name = ['Low', 'Medium', 'High'][cat]
        print(f"   - Category {cat} ({cat_name}): {count} samples ({pct:.1f}%)")
    
    # Drop timestamp columns that we no longer need
    # These can cause Arrow serialization issues when displaying
    timestamp_columns = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'pickup_datetime', 'dropoff_datetime',
        'lpep_pickup_datetime', 'lpep_dropoff_datetime'
    ]
    columns_to_drop = [col for col in timestamp_columns if col in df_features.columns]
    if columns_to_drop:
        df_features = df_features.drop(columns=columns_to_drop)
        print(f"\nℹ️  Dropped timestamp columns (no longer needed): {columns_to_drop}")
    
    return df_features

# Apply feature engineering
df_processed = engineer_features(df)
print(f"Processed dataset shape: {df_processed.shape}")
print("\nFeature statistics:")
display(df_processed.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train Regression Model (Fare Prediction)

# COMMAND ----------

# Prepare features for regression (predicting fare amount)
regression_features = [
    'trip_distance', 
    'passenger_count', 
    'hour_of_day', 
    'day_of_week',
    'is_rush_hour', 
    'is_weekend'
]

X_reg = df_processed[regression_features]
y_reg = df_processed['fare_amount']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train_reg)}")
print(f"Test set size: {len(X_test_reg)}")

# COMMAND ----------

# Train Random Forest Regressor
print("Training Random Forest Regressor...")
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Evaluate
y_pred_reg = rf_regressor.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== Regression Model Performance ===")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': regression_features,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
display(feature_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Classification Model (Tip Category Prediction)

# COMMAND ----------

# Prepare features for classification (predicting tip category)
classification_features = [
    'fare_amount',
    'trip_distance',
    'passenger_count',
    'hour_of_day',
    'day_of_week',
    'is_rush_hour',
    'is_weekend'
]

X_clf = df_processed[classification_features]
y_clf = df_processed['tip_category']

# Check class distribution before splitting
print("=" * 70)
print("Checking class distribution before train/test split...")
print("=" * 70)
class_counts = y_clf.value_counts().sort_index()
print("\nClass distribution in full dataset:")
for cat, count in class_counts.items():
    pct = (count / len(y_clf)) * 100
    cat_name = ['Low', 'Medium', 'High'][cat]
    print(f"  Category {cat} ({cat_name}): {count} samples ({pct:.1f}%)")

# Check if we have enough samples in each class for stratification
min_class_count = class_counts.min()
test_size = 0.2
min_samples_needed = 2  # Need at least 2 samples per class in each split

if min_class_count < (min_samples_needed / test_size):
    print(f"\n⚠️  Warning: Smallest class has only {min_class_count} samples")
    print(f"   This may cause issues with stratified splitting")
    print(f"   Using stratify=None instead")
    stratify_param = None
else:
    stratify_param = y_clf
    print(f"\n✓ All classes have sufficient samples for stratification")

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=test_size, random_state=42, stratify=stratify_param
)

print(f"\nTraining set size: {len(X_train_clf)}")
print(f"Test set size: {len(X_test_clf)}")
print("\nClass distribution in training set:")
train_dist = y_train_clf.value_counts().sort_index()
for cat, count in train_dist.items():
    pct = (count / len(y_train_clf)) * 100
    cat_name = ['Low', 'Medium', 'High'][cat]
    print(f"  Category {cat} ({cat_name}): {count} samples ({pct:.1f}%)")

print("\nClass distribution in test set:")
test_dist = y_test_clf.value_counts().sort_index()
for cat, count in test_dist.items():
    pct = (count / len(y_test_clf)) * 100
    cat_name = ['Low', 'Medium', 'High'][cat]
    print(f"  Category {cat} ({cat_name}): {count} samples ({pct:.1f}%)")
print("=" * 70)

# COMMAND ----------

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_classifier.fit(X_train_clf, y_train_clf)

# Evaluate
y_pred_clf = rf_classifier.predict(X_test_clf)
y_pred_proba_clf = rf_classifier.predict_proba(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)

print("\n=== Classification Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")

# Check how many unique classes are in test set
n_classes_test = len(np.unique(y_test_clf))
n_classes_pred = len(np.unique(y_pred_clf))

print(f"\nClasses in test set: {n_classes_test}")
print(f"Classes in predictions: {n_classes_pred}")

if n_classes_test >= 2 and n_classes_pred >= 2:
    print("\nClassification Report:")
    # Use labels parameter to only show classes that exist in the data
    labels = sorted(np.unique(np.concatenate([y_test_clf, y_pred_clf])))
    target_names = [['Low Tip', 'Medium Tip', 'High Tip'][i] for i in labels]
    print(classification_report(y_test_clf, y_pred_clf, 
                              labels=labels,
                              target_names=target_names,
                              zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_clf, y_pred_clf, labels=labels)
    print(cm)
    
    # Create a nice display of confusion matrix
    cm_df = pd.DataFrame(cm, 
                         index=[f"Actual {tn}" for tn in target_names],
                         columns=[f"Pred {tn}" for tn in target_names])
    print("\nConfusion Matrix (formatted):")
    display(cm_df)
else:
    print("\n⚠️  Warning: Test set or predictions have only one class")
    print("   Classification metrics may not be meaningful")
    print(f"   Test set classes: {np.unique(y_test_clf)}")
    print(f"   Predicted classes: {np.unique(y_pred_clf)}")
    print("\n   This model will still work but may benefit from:")
    print("   - More diverse training data")
    print("   - Different tip category boundaries")
    print("   - Larger dataset")

# Feature importance
feature_importance_clf = pd.DataFrame({
    'feature': classification_features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
display(feature_importance_clf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Python Package Structure for Regression Model

# COMMAND ----------

# Create package directory structure
regression_pkg_dir = f"/tmp/{REGRESSION_PACKAGE_NAME}"
regression_src_dir = f"{regression_pkg_dir}/{REGRESSION_PACKAGE_NAME}"

# Clean up if exists
if os.path.exists(regression_pkg_dir):
    shutil.rmtree(regression_pkg_dir)

os.makedirs(regression_src_dir, exist_ok=True)

print(f"Created package directory: {regression_pkg_dir}")

# COMMAND ----------

# Create __init__.py for regression package
init_code_reg = '''"""
NYC Taxi Fare Predictor
A scikit-learn based model for predicting taxi fare amounts.
Version: {version}
"""

from .predictor import predict_fare, FarePredictor

__version__ = "{version}"
__all__ = ["predict_fare", "FarePredictor"]
'''.format(version=PACKAGE_VERSION)

with open(f"{regression_src_dir}/__init__.py", "w") as f:
    f.write(init_code_reg)

print("Created __init__.py for regression package")

# COMMAND ----------

# Save regression model and create predictor module
model_path_reg = f"{regression_src_dir}/model.pkl"
metadata_path_reg = f"{regression_src_dir}/metadata.json"

# Save model
with open(model_path_reg, 'wb') as f:
    pickle.dump(rf_regressor, f)

# Save metadata
metadata_reg = {
    'model_type': 'RandomForestRegressor',
    'features': regression_features,
    'version': PACKAGE_VERSION,
    'training_date': datetime.now().isoformat(),
    'metrics': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }
}

with open(metadata_path_reg, 'w') as f:
    json.dump(metadata_reg, f, indent=2)

print(f"Saved regression model to: {model_path_reg}")
print(f"Saved metadata to: {metadata_path_reg}")

# COMMAND ----------

# Create predictor.py for regression package
predictor_code_reg = '''"""
Fare prediction module
"""
import pickle
import json
import os
from typing import Dict, List, Union
import numpy as np

# Get the directory where this module is located
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and metadata at module import time
with open(os.path.join(_MODULE_DIR, "model.pkl"), "rb") as f:
    _MODEL = pickle.load(f)

with open(os.path.join(_MODULE_DIR, "metadata.json"), "r") as f:
    _METADATA = json.load(f)

_FEATURES = _METADATA["features"]


class FarePredictor:
    """
    NYC Taxi Fare Predictor
    
    Features required:
    - trip_distance: Distance in miles
    - passenger_count: Number of passengers (1-6)
    - hour_of_day: Hour (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_rush_hour: 1 if rush hour (7-9am or 5-7pm), 0 otherwise
    - is_weekend: 1 if weekend (Sat/Sun), 0 otherwise
    """
    
    def __init__(self):
        self.model = _MODEL
        self.features = _FEATURES
        self.metadata = _METADATA
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict fare amount for a single trip
        
        Args:
            features: Dictionary with feature names as keys
            
        Returns:
            Predicted fare amount in dollars
        """
        # Extract features in correct order
        feature_values = [features.get(f, 0) for f in self.features]
        
        # Predict
        prediction = self.model.predict([feature_values])[0]
        
        # Ensure non-negative
        return max(0, float(prediction))
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """
        Predict fare amounts for multiple trips
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predicted fare amounts
        """
        # Extract features for all samples
        feature_matrix = [
            [features.get(f, 0) for f in self.features]
            for features in features_list
        ]
        
        # Predict
        predictions = self.model.predict(feature_matrix)
        
        # Ensure non-negative
        return [max(0, float(p)) for p in predictions]


def predict_fare(
    trip_distance: float,
    passenger_count: int = 1,
    hour_of_day: int = 12,
    day_of_week: int = 3,
    is_rush_hour: int = 0,
    is_weekend: int = 0
) -> float:
    """
    Simple function to predict taxi fare
    
    Args:
        trip_distance: Distance in miles
        passenger_count: Number of passengers (default: 1)
        hour_of_day: Hour of day 0-23 (default: 12)
        day_of_week: Day of week 0-6 (default: 3 = Wednesday)
        is_rush_hour: 1 if rush hour, 0 otherwise (default: 0)
        is_weekend: 1 if weekend, 0 otherwise (default: 0)
    
    Returns:
        Predicted fare amount in dollars
    """
    features = {
        'trip_distance': trip_distance,
        'passenger_count': passenger_count,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend
    }
    
    predictor = FarePredictor()
    return predictor.predict(features)
'''

with open(f"{regression_src_dir}/predictor.py", "w") as f:
    f.write(predictor_code_reg)

print("Created predictor.py for regression package")

# COMMAND ----------

# Create setup.py for regression package
setup_code_reg = f'''from setuptools import setup, find_packages

setup(
    name="{REGRESSION_PACKAGE_NAME}",
    version="{PACKAGE_VERSION}",
    description="NYC Taxi Fare Predictor using scikit-learn",
    author="Databricks",
    packages=find_packages(),
    package_data={{
        "{REGRESSION_PACKAGE_NAME}": ["model.pkl", "metadata.json"],
    }},
    install_requires=[
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
'''

with open(f"{regression_pkg_dir}/setup.py", "w") as f:
    f.write(setup_code_reg)

print("Created setup.py for regression package")

# COMMAND ----------

# Create README for regression package
readme_reg = f'''# NYC Taxi Fare Predictor

A scikit-learn based model for predicting NYC taxi fare amounts.

## Installation

```bash
pip install {REGRESSION_PACKAGE_NAME}-{PACKAGE_VERSION}-py3-none-any.whl
```

## Usage

### Simple Function

```python
from {REGRESSION_PACKAGE_NAME} import predict_fare

# Predict fare for a 5-mile trip with 2 passengers
fare = predict_fare(
    trip_distance=5.0,
    passenger_count=2,
    hour_of_day=17,  # 5 PM
    day_of_week=4,   # Friday
    is_rush_hour=1,
    is_weekend=0
)
print(f"Predicted fare: ${{fare:.2f}}")
```

### Class-based API

```python
from {REGRESSION_PACKAGE_NAME} import FarePredictor

predictor = FarePredictor()

# Single prediction
features = {{
    'trip_distance': 5.0,
    'passenger_count': 2,
    'hour_of_day': 17,
    'day_of_week': 4,
    'is_rush_hour': 1,
    'is_weekend': 0
}}
fare = predictor.predict(features)

# Batch prediction
features_list = [features, features, features]
fares = predictor.predict_batch(features_list)
```

## Model Information

- Model Type: Random Forest Regressor
- Version: {PACKAGE_VERSION}
- MAE: ${mae:.2f}
- RMSE: ${rmse:.2f}
- R² Score: {r2:.4f}

## Features

- trip_distance: Distance in miles
- passenger_count: Number of passengers (1-6)
- hour_of_day: Hour (0-23)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- is_rush_hour: 1 if rush hour (7-9am or 5-7pm), 0 otherwise
- is_weekend: 1 if weekend (Sat/Sun), 0 otherwise
'''

with open(f"{regression_pkg_dir}/README.md", "w") as f:
    f.write(readme_reg)

print("Created README.md for regression package")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create Python Package Structure for Classification Model

# COMMAND ----------

# Create package directory structure
classification_pkg_dir = f"/tmp/{CLASSIFICATION_PACKAGE_NAME}"
classification_src_dir = f"{classification_pkg_dir}/{CLASSIFICATION_PACKAGE_NAME}"

# Clean up if exists
if os.path.exists(classification_pkg_dir):
    shutil.rmtree(classification_pkg_dir)

os.makedirs(classification_src_dir, exist_ok=True)

print(f"Created package directory: {classification_pkg_dir}")

# COMMAND ----------

# Create __init__.py for classification package
init_code_clf = '''"""
NYC Taxi Tip Classifier
A scikit-learn based model for predicting tip categories.
Version: {version}
"""

from .classifier import predict_tip_category, TipClassifier

__version__ = "{version}"
__all__ = ["predict_tip_category", "TipClassifier"]
'''.format(version=PACKAGE_VERSION)

with open(f"{classification_src_dir}/__init__.py", "w") as f:
    f.write(init_code_clf)

print("Created __init__.py for classification package")

# COMMAND ----------

# Save classification model and create classifier module
model_path_clf = f"{classification_src_dir}/model.pkl"
metadata_path_clf = f"{classification_src_dir}/metadata.json"

# Save model
with open(model_path_clf, 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save metadata
metadata_clf = {
    'model_type': 'RandomForestClassifier',
    'features': classification_features,
    'version': PACKAGE_VERSION,
    'training_date': datetime.now().isoformat(),
    'classes': ['Low Tip', 'Medium Tip', 'High Tip'],
    'class_description': 'Categories based on 33rd and 67th percentiles of tip percentages',
    'metrics': {
        'accuracy': float(accuracy),
        'n_classes_train': int(len(np.unique(y_train_clf))),
        'n_classes_test': int(len(np.unique(y_test_clf)))
    }
}

with open(metadata_path_clf, 'w') as f:
    json.dump(metadata_clf, f, indent=2)

print(f"Saved classification model to: {model_path_clf}")
print(f"Saved metadata to: {metadata_path_clf}")

# COMMAND ----------

# Create classifier.py for classification package
classifier_code = '''"""
Tip category classification module
"""
import pickle
import json
import os
from typing import Dict, List, Tuple
import numpy as np

# Get the directory where this module is located
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and metadata at module import time
with open(os.path.join(_MODULE_DIR, "model.pkl"), "rb") as f:
    _MODEL = pickle.load(f)

with open(os.path.join(_MODULE_DIR, "metadata.json"), "r") as f:
    _METADATA = json.load(f)

_FEATURES = _METADATA["features"]
_CLASSES = _METADATA["classes"]


class TipClassifier:
    """
    NYC Taxi Tip Category Classifier
    
    Predicts tip category:
    - 0: Low Tip (bottom 33% of tip percentages)
    - 1: Medium Tip (middle 33% of tip percentages)  
    - 2: High Tip (top 33% of tip percentages)
    
    Features required:
    - fare_amount: Trip fare in dollars
    - trip_distance: Distance in miles
    - passenger_count: Number of passengers (1-6)
    - hour_of_day: Hour (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_rush_hour: 1 if rush hour (7-9am or 5-7pm), 0 otherwise
    - is_weekend: 1 if weekend (Sat/Sun), 0 otherwise
    """
    
    def __init__(self):
        self.model = _MODEL
        self.features = _FEATURES
        self.metadata = _METADATA
        self.classes = _CLASSES
    
    def predict(self, features: Dict[str, float]) -> Tuple[int, str, List[float]]:
        """
        Predict tip category for a single trip
        
        Args:
            features: Dictionary with feature names as keys
            
        Returns:
            Tuple of (category_id, category_name, probabilities)
        """
        # Extract features in correct order
        feature_values = [features.get(f, 0) for f in self.features]
        
        # Predict
        category_id = int(self.model.predict([feature_values])[0])
        probabilities = self.model.predict_proba([feature_values])[0].tolist()
        category_name = self.classes[category_id]
        
        return category_id, category_name, probabilities
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Tuple[int, str, List[float]]]:
        """
        Predict tip categories for multiple trips
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of tuples (category_id, category_name, probabilities)
        """
        # Extract features for all samples
        feature_matrix = [
            [features.get(f, 0) for f in self.features]
            for features in features_list
        ]
        
        # Predict
        category_ids = self.model.predict(feature_matrix)
        probabilities = self.model.predict_proba(feature_matrix)
        
        results = []
        for cat_id, probs in zip(category_ids, probabilities):
            cat_id = int(cat_id)
            results.append((cat_id, self.classes[cat_id], probs.tolist()))
        
        return results


def predict_tip_category(
    fare_amount: float,
    trip_distance: float,
    passenger_count: int = 1,
    hour_of_day: int = 12,
    day_of_week: int = 3,
    is_rush_hour: int = 0,
    is_weekend: int = 0
) -> str:
    """
    Simple function to predict tip category
    
    Args:
        fare_amount: Trip fare in dollars
        trip_distance: Distance in miles
        passenger_count: Number of passengers (default: 1)
        hour_of_day: Hour of day 0-23 (default: 12)
        day_of_week: Day of week 0-6 (default: 3 = Wednesday)
        is_rush_hour: 1 if rush hour, 0 otherwise (default: 0)
        is_weekend: 1 if weekend, 0 otherwise (default: 0)
    
    Returns:
        Tip category name (e.g., "Low Tip (<10%)")
    """
    features = {
        'fare_amount': fare_amount,
        'trip_distance': trip_distance,
        'passenger_count': passenger_count,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend
    }
    
    classifier = TipClassifier()
    _, category_name, _ = classifier.predict(features)
    return category_name
'''

with open(f"{classification_src_dir}/classifier.py", "w") as f:
    f.write(classifier_code)

print("Created classifier.py for classification package")

# COMMAND ----------

# Create setup.py for classification package
setup_code_clf = f'''from setuptools import setup, find_packages

setup(
    name="{CLASSIFICATION_PACKAGE_NAME}",
    version="{PACKAGE_VERSION}",
    description="NYC Taxi Tip Category Classifier using scikit-learn",
    author="Databricks",
    packages=find_packages(),
    package_data={{
        "{CLASSIFICATION_PACKAGE_NAME}": ["model.pkl", "metadata.json"],
    }},
    install_requires=[
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
'''

with open(f"{classification_pkg_dir}/setup.py", "w") as f:
    f.write(setup_code_clf)

print("Created setup.py for classification package")

# COMMAND ----------

# Create README for classification package
readme_clf = f'''# NYC Taxi Tip Classifier

A scikit-learn based model for predicting NYC taxi tip categories.

## Installation

```bash
pip install {CLASSIFICATION_PACKAGE_NAME}-{PACKAGE_VERSION}-py3-none-any.whl
```

## Usage

### Simple Function

```python
from {CLASSIFICATION_PACKAGE_NAME} import predict_tip_category

# Predict tip category for a trip
category = predict_tip_category(
    fare_amount=25.50,
    trip_distance=5.0,
    passenger_count=2,
    hour_of_day=17,  # 5 PM
    day_of_week=4,   # Friday
    is_rush_hour=1,
    is_weekend=0
)
print(f"Predicted tip category: {{{{category}}}}")
```

### Class-based API

```python
from {CLASSIFICATION_PACKAGE_NAME} import TipClassifier

classifier = TipClassifier()

# Single prediction with probabilities
features = {{
    'fare_amount': 25.50,
    'trip_distance': 5.0,
    'passenger_count': 2,
    'hour_of_day': 17,
    'day_of_week': 4,
    'is_rush_hour': 1,
    'is_weekend': 0
}}
category_id, category_name, probabilities = classifier.predict(features)
print(f"Category: {{{{category_name}}}}")
print(f"Probabilities: {{{{probabilities}}}}")

# Batch prediction
features_list = [features, features, features]
results = classifier.predict_batch(features_list)
```

## Model Information

- Model Type: Random Forest Classifier
- Version: {PACKAGE_VERSION}
- Accuracy: {accuracy:.4f}

## Classes

- 0: Low Tip (bottom 33% of tip percentages)
- 1: Medium Tip (middle 33% of tip percentages)
- 2: High Tip (top 33% of tip percentages)

Note: Categories are determined using 33rd and 67th percentiles of the training data 
to ensure balanced class distribution.

## Features

- fare_amount: Trip fare in dollars
- trip_distance: Distance in miles
- passenger_count: Number of passengers (1-6)
- hour_of_day: Hour (0-23)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- is_rush_hour: 1 if rush hour (7-9am or 5-7pm), 0 otherwise
- is_weekend: 1 if weekend (Sat/Sun), 0 otherwise
'''

with open(f"{classification_pkg_dir}/README.md", "w") as f:
    f.write(readme_clf)

print("Created README.md for classification package")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Build Wheel Files

# COMMAND ----------

# Build regression wheel
print("=" * 60)
print("Building Regression Model Wheel File...")
print("=" * 60)

import subprocess
import sys

# Build wheel for regression model
result = subprocess.run(
    [sys.executable, "setup.py", "bdist_wheel"],
    cwd=regression_pkg_dir,
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✓ Regression wheel built successfully!")
    
    # Find the wheel file
    dist_dir = f"{regression_pkg_dir}/dist"
    wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
    if wheel_files:
        regression_wheel_path = f"{dist_dir}/{wheel_files[0]}"
        print(f"Wheel file: {regression_wheel_path}")
        
        # Display wheel file info
        wheel_size = os.path.getsize(regression_wheel_path) / 1024
        print(f"Wheel size: {wheel_size:.2f} KB")
else:
    print("✗ Error building regression wheel:")
    print(result.stderr)

# COMMAND ----------

# Build classification wheel
print("=" * 60)
print("Building Classification Model Wheel File...")
print("=" * 60)

# Build wheel for classification model
result = subprocess.run(
    [sys.executable, "setup.py", "bdist_wheel"],
    cwd=classification_pkg_dir,
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✓ Classification wheel built successfully!")
    
    # Find the wheel file
    dist_dir = f"{classification_pkg_dir}/dist"
    wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
    if wheel_files:
        classification_wheel_path = f"{dist_dir}/{wheel_files[0]}"
        print(f"Wheel file: {classification_wheel_path}")
        
        # Display wheel file info
        wheel_size = os.path.getsize(classification_wheel_path) / 1024
        print(f"Wheel size: {wheel_size:.2f} KB")
else:
    print("✗ Error building classification wheel:")
    print(result.stderr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Create Unity Catalog Volume (if needed)

# COMMAND ----------

# Create volume if it doesn't exist
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
    print(f"✓ Volume {CATALOG}.{SCHEMA}.{VOLUME} is ready")
except Exception as e:
    print(f"Note: Could not create volume. You may need to create it manually:")
    print(f"  CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
    print(f"\nError: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Upload Wheel Files to Unity Catalog Volume

# COMMAND ----------

# DBTITLE 1,Cell 35
# Define volume path with folder
volume_base_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
volume_path = f"{volume_base_path}/{FOLDER_NAME}"

print(f"Uploading wheel files to: {volume_path}")
print("=" * 60)

# Create folder in volume if it doesn't exist
try:
    dbutils.fs.mkdirs(volume_path)
    print(f"✓ Folder created/verified: {FOLDER_NAME}")
except Exception as e:
    print(f"Note: Folder may already exist or error creating: {e}")

print()

# On Shared clusters, we need to copy files to /Workspace first, then to UC Volume
import shutil

# Get current username
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Create a temporary workspace directory for wheel files
workspace_temp_dir = f"/Workspace/Users/{current_user}/temp_wheels"
os.makedirs(workspace_temp_dir, exist_ok=True)
print(f"Using temporary workspace directory: {workspace_temp_dir}")
print()

# Upload regression wheel
try:
    regression_wheel_name = os.path.basename(regression_wheel_path)
    
    # Copy to workspace first
    workspace_regression_path = f"{workspace_temp_dir}/{regression_wheel_name}"
    shutil.copy2(regression_wheel_path, workspace_regression_path)
    
    # Now upload from workspace to UC volume
    dbutils.fs.cp(
        f"file:{workspace_regression_path}",
        f"{volume_path}/{regression_wheel_name}",
        recurse=False
    )
    print(f"✓ Uploaded: {regression_wheel_name}")
    regression_wheel_uc_path = f"{volume_path}/{regression_wheel_name}"
except Exception as e:
    print(f"✗ Error uploading regression wheel: {e}")
    regression_wheel_uc_path = None

# Upload classification wheel
try:
    classification_wheel_name = os.path.basename(classification_wheel_path)
    
    # Copy to workspace first
    workspace_classification_path = f"{workspace_temp_dir}/{classification_wheel_name}"
    shutil.copy2(classification_wheel_path, workspace_classification_path)
    
    # Now upload from workspace to UC volume
    dbutils.fs.cp(
        f"file:{workspace_classification_path}",
        f"{volume_path}/{classification_wheel_name}",
        recurse=False
    )
    print(f"✓ Uploaded: {classification_wheel_name}")
    classification_wheel_uc_path = f"{volume_path}/{classification_wheel_name}"
except Exception as e:
    print(f"✗ Error uploading classification wheel: {e}")
    classification_wheel_uc_path = None

# Clean up temporary workspace directory
try:
    shutil.rmtree(workspace_temp_dir)
    print(f"\n✓ Cleaned up temporary workspace directory")
except Exception as e:
    print(f"\nNote: Could not clean up temp directory: {e}")

# COMMAND ----------

# List files in volume folder
print("\nFiles in Unity Catalog Volume Folder:")
print("=" * 60)
print(f"Location: {volume_path}")
print()
try:
    files = dbutils.fs.ls(volume_path)
    for file_info in files:
        size_kb = file_info.size / 1024
        print(f"  {file_info.name} ({size_kb:.2f} KB)")
except Exception as e:
    print(f"Could not list volume contents: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Test the Packages Locally

# COMMAND ----------

# Test regression package
print("Testing Regression Package...")
print("=" * 60)

# Install and import
%pip install {regression_wheel_path} --quiet

from nyc_taxi_fare_predictor import predict_fare, FarePredictor

# Test simple function
test_fare = predict_fare(
    trip_distance=5.0,
    passenger_count=2,
    hour_of_day=17,
    day_of_week=4,
    is_rush_hour=1,
    is_weekend=0
)
print(f"Simple function test - Predicted fare: ${test_fare:.2f}")

# Test class-based API
predictor = FarePredictor()
test_features = {
    'trip_distance': 3.5,
    'passenger_count': 1,
    'hour_of_day': 10,
    'day_of_week': 2,
    'is_rush_hour': 0,
    'is_weekend': 0
}
class_fare = predictor.predict(test_features)
print(f"Class-based API test - Predicted fare: ${class_fare:.2f}")

print("✓ Regression package working correctly!")

# COMMAND ----------

# DBTITLE 1,Cell 39
# Test classification package
print("\nTesting Classification Package...")
print("=" * 60)

# Install and import
%pip install {classification_wheel_path} --quiet

from nyc_taxi_tip_classifier import predict_tip_category, TipClassifier

# Test simple function
test_category = predict_tip_category(
    fare_amount=25.50,
    trip_distance=5.0,
    passenger_count=2,
    hour_of_day=17,
    day_of_week=4,
    is_rush_hour=1,
    is_weekend=0
)
print(f"Simple function test - Predicted category: {test_category}")

# Test class-based API
classifier = TipClassifier()
test_features_clf = {
    'fare_amount': 15.00,
    'trip_distance': 3.5,
    'passenger_count': 1,
    'hour_of_day': 10,
    'day_of_week': 2,
    'is_rush_hour': 0,
    'is_weekend': 0
}
cat_id, cat_name, probs = classifier.predict(test_features_clf)
print(f"Class-based API test - Category: {cat_name}")
print(f"Category ID: {cat_id}")
print(f"Number of probability values: {len(probs)}")
print(f"Probabilities: {probs}")

# Display probabilities based on how many classes we have
if len(probs) == 3:
    print(f"Probabilities: Low={probs[0]:.2%}, Medium={probs[1]:.2%}, High={probs[2]:.2%}")
elif len(probs) == 2:
    print(f"Probabilities: Class 0={probs[0]:.2%}, Class 1={probs[1]:.2%}")
else:
    print(f"Probabilities: {[f'{p:.2%}' for p in probs]}")

print("\u2713 Classification package working correctly!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary and Next Steps

# COMMAND ----------

print("=" * 70)
print("SUMMARY - ML Models for Unity Catalog UDFs")
print("=" * 70)
print("\n✓ Successfully created and packaged two ML models:")
print("\n1. REGRESSION MODEL (Fare Prediction)")
print(f"   - Package: {REGRESSION_PACKAGE_NAME}")
print(f"   - Version: {PACKAGE_VERSION}")
print(f"   - Performance: MAE=${mae:.2f}, RMSE=${rmse:.2f}, R²={r2:.4f}")
if regression_wheel_uc_path:
    print(f"   - Location: {regression_wheel_uc_path}")

print("\n2. CLASSIFICATION MODEL (Tip Category)")
print(f"   - Package: {CLASSIFICATION_PACKAGE_NAME}")
print(f"   - Version: {PACKAGE_VERSION}")
print(f"   - Performance: Accuracy={accuracy:.4f}")
if classification_wheel_uc_path:
    print(f"   - Location: {classification_wheel_uc_path}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\n1. Create Unity Catalog UDFs using these wheel files")
print("   - Use ENVIRONMENT clause to specify dependencies")
print(f"   - Reference wheels from: {volume_path}")
print("\n2. Example UDF creation (Regression Model):")
if regression_wheel_uc_path:
    print(f'''
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.predict_taxi_fare(
    trip_distance DOUBLE,
    passenger_count INT,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour INT,
    is_weekend INT
)
RETURNS DOUBLE
LANGUAGE PYTHON
ENVIRONMENT (
    dependencies = '["{regression_wheel_uc_path}"]',
    environment_version = 'None'
)
AS $$
from {REGRESSION_PACKAGE_NAME} import predict_fare
return predict_fare(trip_distance, passenger_count, hour_of_day, 
                   day_of_week, is_rush_hour, is_weekend)
$$;
''')
else:
    print("   (Wheel file path not available)")

print("\n3. Example UDF creation (Classification Model):")
if classification_wheel_uc_path:
    print(f'''
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.predict_tip_category(
    fare_amount DOUBLE,
    trip_distance DOUBLE,
    passenger_count INT,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour INT,
    is_weekend INT
)
RETURNS STRING
LANGUAGE PYTHON
ENVIRONMENT (
    dependencies = '["{classification_wheel_uc_path}"]',
    environment_version = 'None'
)
AS $$
from {CLASSIFICATION_PACKAGE_NAME} import predict_tip_category
return predict_tip_category(fare_amount, trip_distance, passenger_count, 
                            hour_of_day, day_of_week, is_rush_hour, is_weekend)
$$;
''')
else:
    print("   (Wheel file path not available)")

print("\n4. Test the UDFs in SQL:")
print(f'''
-- Test regression UDF
SELECT {CATALOG}.{SCHEMA}.predict_taxi_fare(5.0, 2, 17, 4, 1, 0) AS predicted_fare;

-- Test classification UDF
SELECT {CATALOG}.{SCHEMA}.predict_tip_category(25.50, 5.0, 2, 17, 4, 1, 0) AS tip_category;
''')

print("\n5. Use in DataFrame operations:")
print(f'''
from pyspark.sql.functions import expr

# Add fare predictions
df_with_predictions = df.withColumn("predicted_fare", 
    expr("{CATALOG}.{SCHEMA}.predict_taxi_fare(trip_distance, passenger_count, hour_of_day, day_of_week, is_rush_hour, is_weekend)")
)

# Add tip category predictions
df_with_tips = df.withColumn("tip_category",
    expr("{CATALOG}.{SCHEMA}.predict_tip_category(fare_amount, trip_distance, passenger_count, hour_of_day, day_of_week, is_rush_hour, is_weekend)")
)
''')

print("\n" + "=" * 70)
print("PACKAGE INFORMATION")
print("=" * 70)

# COMMAND ----------

# Display package information for reference
print("\n📦 Local Package Paths:")
print("-" * 70)
print(f"Regression wheel:      {regression_wheel_path if 'regression_wheel_path' in locals() else 'Not created'}")
print(f"Classification wheel:  {classification_wheel_path if 'classification_wheel_path' in locals() else 'Not created'}")

print("\n☁️  Unity Catalog Paths:")
print("-" * 70)
print(f"Volume location:       {volume_path}")
print(f"Regression wheel:      {regression_wheel_uc_path if regression_wheel_uc_path else 'Not uploaded'}")
print(f"Classification wheel:  {classification_wheel_uc_path if classification_wheel_uc_path else 'Not uploaded'}")

print("\n" + "=" * 70)
print("✅ SETUP COMPLETE!")
print("=" * 70)
print("\nFor complete UDF examples with these models, see the next notebook:")
print("  → 02_create_udfs_with_custom_dependencies.py")
print("=" * 70)