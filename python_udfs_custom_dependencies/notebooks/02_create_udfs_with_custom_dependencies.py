# Databricks notebook source
# MAGIC %md
# MAGIC # Create and Use Python UDFs with Custom Dependencies
# MAGIC 
# MAGIC **Author**: Prasad Kona  
# MAGIC **Last Updated**: January 26, 2026  
# MAGIC **Notebook**: 02 - Create and Use UDFs with Custom Dependencies
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook demonstrates how to create and use Python User-Defined Functions (UDFs) with custom dependencies in Unity Catalog.
# MAGIC 
# MAGIC ## What You'll Learn
# MAGIC 
# MAGIC 1. **Create UDFs** with custom wheel file dependencies from Unity Catalog volumes
# MAGIC 2. **Use UDFs in SQL** queries for ML inference at scale
# MAGIC 3. **Use UDFs in PySpark** DataFrames for batch processing
# MAGIC 4. **Manage UDFs** - permissions, versioning, and best practices
# MAGIC 5. **Performance optimization** with DETERMINISTIC and isolation modes
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC - Complete notebook `01_build_ml_models_for_udfs.py`
# MAGIC - Wheel files uploaded to Unity Catalog volume
# MAGIC - Databricks Runtime 18.1 or above
# MAGIC 
# MAGIC ## Documentation
# MAGIC 
# MAGIC For complete documentation on Python UDFs in Unity Catalog, see:
# MAGIC - **Official Docs**: https://docs.databricks.com/en/udf/unity-catalog.html
# MAGIC - **CREATE FUNCTION Reference**: https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html
# MAGIC - **Unity Catalog Functions**: https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC 
# MAGIC These should match the settings from notebook 01.

# COMMAND ----------

# Configuration - Update these based on your Unity Catalog setup
CATALOG = "main"  # Replace with your catalog name
SCHEMA = "default"  # Replace with your schema name
VOLUME = "ml_models"  # Volume name to create/use
FOLDER_NAME = "taxi_models"  # Folder within volume containing wheel files

# Package names and version (should match notebook 01)
PACKAGE_VERSION = "1.0.0"
REGRESSION_PACKAGE_NAME = "nyc_taxi_fare_predictor"
CLASSIFICATION_PACKAGE_NAME = "nyc_taxi_tip_classifier"

# Construct full paths
volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FOLDER_NAME}"
regression_wheel_path = f"{volume_path}/{REGRESSION_PACKAGE_NAME}-{PACKAGE_VERSION}-py3-none-any.whl"
classification_wheel_path = f"{volume_path}/{CLASSIFICATION_PACKAGE_NAME}-{PACKAGE_VERSION}-py3-none-any.whl"

# Display configuration
print("=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"Catalog:               {CATALOG}")
print(f"Schema:                {SCHEMA}")
print(f"Volume:                {VOLUME}")
print(f"Folder:                {FOLDER_NAME}")
print(f"Full UC Path:          {volume_path}")
print(f"\nRegression wheel:      {regression_wheel_path}")
print(f"Classification wheel:  {classification_wheel_path}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Wheel Files in Volume

# COMMAND ----------

print("Checking for wheel files in Unity Catalog volume...")
print("=" * 70)

try:
    files = dbutils.fs.ls(volume_path)
    print(f"\n✓ Found {len(files)} file(s) in {volume_path}:\n")
    
    regression_found = False
    classification_found = False
    
    for file_info in files:
        size_kb = file_info.size / 1024
        print(f"  📦 {file_info.name} ({size_kb:.2f} KB)")
        
        if REGRESSION_PACKAGE_NAME in file_info.name:
            regression_found = True
        if CLASSIFICATION_PACKAGE_NAME in file_info.name:
            classification_found = True
    
    print("\n" + "=" * 70)
    print("Verification Status:")
    print("=" * 70)
    print(f"  Regression wheel:      {'✓ Found' if regression_found else '✗ Not found'}")
    print(f"  Classification wheel:  {'✓ Found' if classification_found else '✗ Not found'}")
    
    if not (regression_found and classification_found):
        print("\n⚠️  Warning: Some wheel files are missing!")
        print("   Please run notebook 01_build_ml_models_for_udfs.py first.")
    else:
        print("\n✓ All wheel files are available!")
        
except Exception as e:
    print(f"\n✗ Error accessing volume: {e}")
    print("\nPlease ensure:")
    print("  1. The volume exists")
    print("  2. You have READ VOLUME permissions")
    print("  3. Wheel files were uploaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Python UDF for Fare Prediction (Regression)
# MAGIC 
# MAGIC This UDF uses the regression model to predict taxi fare amounts based on trip characteristics.

# COMMAND ----------

# Create the fare prediction UDF
udf_fare_prediction = f"{CATALOG}.{SCHEMA}.predict_taxi_fare"

print(f"Creating UDF: {udf_fare_prediction}")
print("=" * 70)

create_fare_udf_sql = f"""
CREATE OR REPLACE FUNCTION {udf_fare_prediction}(
    trip_distance DOUBLE,
    passenger_count INT,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour INT,
    is_weekend INT
)
RETURNS DOUBLE
COMMENT 'Predicts NYC taxi fare amount using a Random Forest model.

Parameters:
- trip_distance: Distance in miles (e.g., 5.0)
- passenger_count: Number of passengers 1-6 (e.g., 2)
- hour_of_day: Hour of day 0-23 (e.g., 17 for 5 PM)
- day_of_week: Day of week 0-6, where 0=Monday, 6=Sunday (e.g., 4 for Friday)
- is_rush_hour: 1 if during rush hour (7-9am or 5-7pm), 0 otherwise
- is_weekend: 1 if weekend (Saturday/Sunday), 0 otherwise

Returns:
- Predicted fare amount in dollars

Example:
SELECT predict_taxi_fare(5.0, 2, 17, 4, 1, 0) AS predicted_fare;

Model: Random Forest Regressor
Version: {PACKAGE_VERSION}
Package: {REGRESSION_PACKAGE_NAME}
'
LANGUAGE PYTHON
DETERMINISTIC
ENVIRONMENT (
    dependencies = '["{regression_wheel_path}"]',
    environment_version = 'None'
)
AS $$
from {REGRESSION_PACKAGE_NAME} import predict_fare
return predict_fare(trip_distance, passenger_count, hour_of_day, 
                   day_of_week, is_rush_hour, is_weekend)
$$;
"""

try:
    spark.sql(create_fare_udf_sql)
    print(f"✓ Successfully created UDF: {udf_fare_prediction}")
    print("\nUDF is now available for use in SQL queries and PySpark!")
except Exception as e:
    print(f"✗ Error creating UDF: {e}")
    print("\nPlease check:")
    print("  1. You have CREATE FUNCTION permission")
    print("  2. The wheel file path is correct")
    print("  3. The volume is accessible")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Python UDF for Tip Category Prediction (Classification)
# MAGIC 
# MAGIC This UDF uses the classification model to predict tip categories (Low/Medium/High).

# COMMAND ----------

# Create the tip category prediction UDF
udf_tip_category = f"{CATALOG}.{SCHEMA}.predict_tip_category"

print(f"Creating UDF: {udf_tip_category}")
print("=" * 70)

create_tip_udf_sql = f"""
CREATE OR REPLACE FUNCTION {udf_tip_category}(
    fare_amount DOUBLE,
    trip_distance DOUBLE,
    passenger_count INT,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour INT,
    is_weekend INT
)
RETURNS STRING
COMMENT 'Predicts NYC taxi tip category using a Random Forest classifier.

Parameters:
- fare_amount: Trip fare in dollars (e.g., 25.50)
- trip_distance: Distance in miles (e.g., 5.0)
- passenger_count: Number of passengers 1-6 (e.g., 2)
- hour_of_day: Hour of day 0-23 (e.g., 17 for 5 PM)
- day_of_week: Day of week 0-6, where 0=Monday, 6=Sunday (e.g., 4 for Friday)
- is_rush_hour: 1 if during rush hour (7-9am or 5-7pm), 0 otherwise
- is_weekend: 1 if weekend (Saturday/Sunday), 0 otherwise

Returns:
- Tip category as string: "Low Tip (<10%)", "Medium Tip (10-20%)", or "High Tip (>20%)"

Example:
SELECT predict_tip_category(25.50, 5.0, 2, 17, 4, 1, 0) AS tip_category;

Model: Random Forest Classifier
Version: {PACKAGE_VERSION}
Package: {CLASSIFICATION_PACKAGE_NAME}
'
LANGUAGE PYTHON
DETERMINISTIC
ENVIRONMENT (
    dependencies = '["{classification_wheel_path}"]',
    environment_version = 'None'
)
AS $$
from {CLASSIFICATION_PACKAGE_NAME} import predict_tip_category
return predict_tip_category(fare_amount, trip_distance, passenger_count, 
                            hour_of_day, day_of_week, is_rush_hour, is_weekend)
$$;
"""

try:
    spark.sql(create_tip_udf_sql)
    print(f"✓ Successfully created UDF: {udf_tip_category}")
    print("\nUDF is now available for use in SQL queries and PySpark!")
except Exception as e:
    print(f"✗ Error creating UDF: {e}")
    print("\nPlease check:")
    print("  1. You have CREATE FUNCTION permission")
    print("  2. The wheel file path is correct")
    print("  3. The volume is accessible")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify UDFs Were Created

# COMMAND ----------

print("Checking created UDFs...")
print("=" * 70)

# Set the catalog context first
spark.sql(f"USE CATALOG {CATALOG}")

# List all functions in the schema
functions_df = spark.sql(f"SHOW FUNCTIONS IN {SCHEMA}")

our_functions = functions_df.filter(
    functions_df.function.contains("predict_taxi") | 
    functions_df.function.contains("predict_tip")
)

print("\n📋 UDFs in current schema:\n")
display(our_functions)

# Get detailed information about each UDF
print("\n" + "=" * 70)
print("UDF Details")
print("=" * 70)

for udf_name in [udf_fare_prediction, udf_tip_category]:
    try:
        print(f"\n🔍 {udf_name}")
        print("-" * 70)
        udf_info = spark.sql(f"DESCRIBE FUNCTION EXTENDED {udf_name}")
        display(udf_info)
    except Exception as e:
        print(f"Could not get details for {udf_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test UDFs with Simple Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Fare Prediction UDF

# COMMAND ----------

print("Testing Fare Prediction UDF")
print("=" * 70)

# Test case 1: Short weekday trip
test_sql_1 = f"""
SELECT 
    '{udf_fare_prediction}' as udf_name,
    'Short weekday trip (2 miles, 1 passenger, 10am Wednesday)' as scenario,
    {udf_fare_prediction}(2.0, 1, 10, 2, 0, 0) as predicted_fare
"""

print("\n📊 Test Case 1: Short weekday trip")
display(spark.sql(test_sql_1))

# Test case 2: Long rush hour trip
test_sql_2 = f"""
SELECT 
    '{udf_fare_prediction}' as udf_name,
    'Long rush hour trip (8 miles, 3 passengers, 5pm Friday)' as scenario,
    {udf_fare_prediction}(8.0, 3, 17, 4, 1, 0) as predicted_fare
"""

print("\n📊 Test Case 2: Long rush hour trip")
display(spark.sql(test_sql_2))

# Test case 3: Weekend trip
test_sql_3 = f"""
SELECT 
    '{udf_fare_prediction}' as udf_name,
    'Weekend trip (5 miles, 2 passengers, 2pm Saturday)' as scenario,
    {udf_fare_prediction}(5.0, 2, 14, 5, 0, 1) as predicted_fare
"""

print("\n📊 Test Case 3: Weekend trip")
display(spark.sql(test_sql_3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Tip Category Prediction UDF

# COMMAND ----------

print("Testing Tip Category Prediction UDF")
print("=" * 70)

# Test case 1: Low fare, short trip
test_tip_sql_1 = f"""
SELECT 
    '{udf_tip_category}' as udf_name,
    'Low fare short trip ($10 fare, 2 miles)' as scenario,
    {udf_tip_category}(10.0, 2.0, 1, 10, 2, 0, 0) as predicted_tip_category
"""

print("\n📊 Test Case 1: Low fare, short trip")
display(spark.sql(test_tip_sql_1))

# Test case 2: High fare, long trip
test_tip_sql_2 = f"""
SELECT 
    '{udf_tip_category}' as udf_name,
    'High fare long trip ($50 fare, 10 miles)' as scenario,
    {udf_tip_category}(50.0, 10.0, 2, 17, 4, 1, 0) as predicted_tip_category
"""

print("\n📊 Test Case 2: High fare, long trip")
display(spark.sql(test_tip_sql_2))

# Test case 3: Medium fare
test_tip_sql_3 = f"""
SELECT 
    '{udf_tip_category}' as udf_name,
    'Medium fare trip ($25 fare, 5 miles)' as scenario,
    {udf_tip_category}(25.0, 5.0, 2, 14, 5, 0, 1) as predicted_tip_category
"""

print("\n📊 Test Case 3: Medium fare trip")
display(spark.sql(test_tip_sql_3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create Sample Data for Batch Testing

# COMMAND ----------

import pandas as pd
import numpy as np

# Create sample taxi trip data
np.random.seed(42)
n_samples = 100

sample_data = pd.DataFrame({
    'trip_id': range(1, n_samples + 1),
    'trip_distance': np.random.exponential(3, n_samples).clip(0.1, 50),
    'passenger_count': np.random.randint(1, 5, n_samples),
    'hour_of_day': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
})

# Calculate derived features
sample_data['is_rush_hour'] = sample_data['hour_of_day'].apply(
    lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
)
sample_data['is_weekend'] = sample_data['day_of_week'].apply(
    lambda x: 1 if x >= 5 else 0
)

# Simulate actual fare amounts (for comparison with predictions)
sample_data['actual_fare'] = (
    2.5 + 
    sample_data['trip_distance'] * 3.5 + 
    sample_data['is_rush_hour'] * 2 +
    np.random.normal(0, 2, n_samples)
).clip(2.5, 200)

# Create Spark DataFrame
df_trips = spark.createDataFrame(sample_data)

print("Created sample dataset with 100 taxi trips")
print("=" * 70)
print("\nSample data:")
display(df_trips.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Use UDFs in SQL Queries

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Add Fare Predictions to Trip Data

# COMMAND ----------

# Register the DataFrame as a temporary view
df_trips.createOrReplaceTempView("sample_trips")

# Use UDF in SQL query
sql_query_1 = f"""
SELECT 
    trip_id,
    trip_distance,
    passenger_count,
    hour_of_day,
    day_of_week,
    is_rush_hour,
    is_weekend,
    actual_fare,
    {udf_fare_prediction}(
        trip_distance, 
        passenger_count, 
        hour_of_day, 
        day_of_week,
        is_rush_hour,
        is_weekend
    ) as predicted_fare,
    actual_fare - {udf_fare_prediction}(
        trip_distance, 
        passenger_count, 
        hour_of_day, 
        day_of_week,
        is_rush_hour,
        is_weekend
    ) as prediction_error
FROM sample_trips
ORDER BY trip_id
LIMIT 20
"""

print("SQL Query with Fare Prediction UDF")
print("=" * 70)
result_df = spark.sql(sql_query_1)
display(result_df)

# Calculate prediction metrics
print("\n" + "=" * 70)
print("Prediction Performance Metrics")
print("=" * 70)

metrics_sql = f"""
SELECT 
    COUNT(*) as total_trips,
    ROUND(AVG(ABS(actual_fare - {udf_fare_prediction}(
        trip_distance, passenger_count, hour_of_day, 
        day_of_week, is_rush_hour, is_weekend
    ))), 2) as mean_absolute_error,
    ROUND(SQRT(AVG(POW(actual_fare - {udf_fare_prediction}(
        trip_distance, passenger_count, hour_of_day, 
        day_of_week, is_rush_hour, is_weekend
    ), 2))), 2) as root_mean_squared_error
FROM sample_trips
"""

display(spark.sql(metrics_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2: Predict Tip Categories for Trips

# COMMAND ----------

sql_query_2 = f"""
SELECT 
    trip_id,
    actual_fare,
    trip_distance,
    passenger_count,
    CASE 
        WHEN hour_of_day < 6 THEN 'Night'
        WHEN hour_of_day < 12 THEN 'Morning'
        WHEN hour_of_day < 18 THEN 'Afternoon'
        ELSE 'Evening'
    END as time_period,
    {udf_tip_category}(
        actual_fare,
        trip_distance, 
        passenger_count, 
        hour_of_day, 
        day_of_week,
        is_rush_hour,
        is_weekend
    ) as predicted_tip_category
FROM sample_trips
ORDER BY trip_id
LIMIT 20
"""

print("SQL Query with Tip Category Prediction UDF")
print("=" * 70)
display(spark.sql(sql_query_2))

# Analyze tip category distribution
print("\n" + "=" * 70)
print("Tip Category Distribution")
print("=" * 70)

distribution_sql = f"""
SELECT 
    {udf_tip_category}(
        actual_fare, trip_distance, passenger_count, 
        hour_of_day, day_of_week, is_rush_hour, is_weekend
    ) as tip_category,
    COUNT(*) as trip_count,
    ROUND(AVG(actual_fare), 2) as avg_fare,
    ROUND(AVG(trip_distance), 2) as avg_distance
FROM sample_trips
GROUP BY tip_category
ORDER BY tip_category
"""

display(spark.sql(distribution_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3: Combined Analysis with Both UDFs

# COMMAND ----------

combined_sql = f"""
SELECT 
    CASE 
        WHEN is_weekend = 1 THEN 'Weekend'
        ELSE 'Weekday'
    END as day_type,
    CASE 
        WHEN is_rush_hour = 1 THEN 'Rush Hour'
        ELSE 'Non-Rush Hour'
    END as rush_hour_status,
    COUNT(*) as trip_count,
    ROUND(AVG(actual_fare), 2) as avg_actual_fare,
    ROUND(AVG({udf_fare_prediction}(
        trip_distance, passenger_count, hour_of_day, 
        day_of_week, is_rush_hour, is_weekend
    )), 2) as avg_predicted_fare,
    {udf_tip_category}(
        AVG(actual_fare),
        AVG(trip_distance), 
        2,  -- avg passenger count
        12, -- midday
        3,  -- Wednesday
        MAX(is_rush_hour),
        MAX(is_weekend)
    ) as typical_tip_category
FROM sample_trips
GROUP BY day_type, rush_hour_status
ORDER BY day_type, rush_hour_status
"""

print("Combined Analysis with Both UDFs")
print("=" * 70)
print("Analyze fare predictions and tip categories by day type and rush hour\n")
display(spark.sql(combined_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Use UDFs in PySpark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Add Predictions Using expr()

# COMMAND ----------

from pyspark.sql.functions import expr, col, round as spark_round

print("Using UDFs in PySpark with expr()")
print("=" * 70)

# Add fare predictions
df_with_predictions = df_trips.withColumn(
    "predicted_fare",
    spark_round(
        expr(f"""{udf_fare_prediction}(
            trip_distance, 
            passenger_count, 
            hour_of_day, 
            day_of_week,
            is_rush_hour,
            is_weekend
        )"""),
        2
    )
)

# Add tip category predictions
df_with_predictions = df_with_predictions.withColumn(
    "predicted_tip_category",
    expr(f"""{udf_tip_category}(
        actual_fare,
        trip_distance, 
        passenger_count, 
        hour_of_day, 
        day_of_week,
        is_rush_hour,
        is_weekend
    )""")
)

# Add prediction error
df_with_predictions = df_with_predictions.withColumn(
    "prediction_error",
    spark_round(col("actual_fare") - col("predicted_fare"), 2)
)

print("\nDataFrame with predictions:")
display(df_with_predictions.select(
    "trip_id", "trip_distance", "actual_fare", 
    "predicted_fare", "prediction_error", "predicted_tip_category"
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2: Filter and Analyze Using UDF Results

# COMMAND ----------

print("Filtering trips with high predicted fares")
print("=" * 70)

# Find trips with predicted fare > $30
high_fare_trips = df_with_predictions.filter(col("predicted_fare") > 30)

print(f"\nFound {high_fare_trips.count()} trips with predicted fare > $30\n")
display(high_fare_trips.select(
    "trip_id", "trip_distance", "passenger_count", 
    "predicted_fare", "predicted_tip_category"
).limit(10))

# COMMAND ----------

print("\nAnalyze trips by predicted tip category")
print("=" * 70)

tip_analysis = df_with_predictions.groupBy("predicted_tip_category").agg(
    {"trip_distance": "avg", "actual_fare": "avg", "trip_id": "count"}
).withColumnRenamed("count(trip_id)", "trip_count") \
 .withColumnRenamed("avg(trip_distance)", "avg_distance") \
 .withColumnRenamed("avg(actual_fare)", "avg_fare") \
 .orderBy("predicted_tip_category")

display(tip_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3: Save Results to Delta Table

# COMMAND ----------

# Save predictions to a Delta table
output_table = f"{CATALOG}.{SCHEMA}.taxi_predictions"

print(f"Saving predictions to Delta table: {output_table}")
print("=" * 70)

try:
    df_with_predictions.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(output_table)
    
    print(f"✓ Successfully saved predictions to {output_table}")
    
    # Show table info
    print(f"\nTable details:")
    display(spark.sql(f"DESCRIBE TABLE EXTENDED {output_table}"))
    
    # Show sample records
    print(f"\nSample records from {output_table}:")
    display(spark.table(output_table).limit(10))
    
except Exception as e:
    print(f"✗ Error saving to Delta table: {e}")
    print("\nNote: You may need CREATE TABLE permission in the schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. UDF Management and Best Practices

# COMMAND ----------

# MAGIC %md
# MAGIC ### View UDF Permissions

# COMMAND ----------

print("Viewing UDF Permissions")
print("=" * 70)

for udf_name in [udf_fare_prediction, udf_tip_category]:
    print(f"\n🔐 Permissions for {udf_name}:")
    print("-" * 70)
    try:
        permissions = spark.sql(f"SHOW GRANTS ON FUNCTION {udf_name}")
        display(permissions)
    except Exception as e:
        print(f"Could not retrieve permissions: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant Permissions to Users/Groups
# MAGIC 
# MAGIC ```sql
# MAGIC -- Grant EXECUTE permission to a user
# MAGIC GRANT EXECUTE ON FUNCTION main.default.predict_taxi_fare TO `user@example.com`;
# MAGIC 
# MAGIC -- Grant EXECUTE permission to a group
# MAGIC GRANT EXECUTE ON FUNCTION main.default.predict_taxi_fare TO `data_scientists`;
# MAGIC 
# MAGIC -- Grant all privileges (EXECUTE + MANAGE)
# MAGIC GRANT ALL PRIVILEGES ON FUNCTION main.default.predict_taxi_fare TO `admin_group`;
# MAGIC 
# MAGIC -- Revoke permissions
# MAGIC REVOKE EXECUTE ON FUNCTION main.default.predict_taxi_fare FROM `user@example.com`;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Advanced UDF Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: UDF with Multiple Dependencies
# MAGIC 
# MAGIC This example shows how to create a UDF that uses multiple packages (PyPI + UC Volume).

# COMMAND ----------

# Create a UDF that combines predictions with JSON output
udf_json_predictions = f"{CATALOG}.{SCHEMA}.predict_taxi_json"

print(f"Creating advanced UDF with multiple dependencies: {udf_json_predictions}")
print("=" * 70)

create_json_udf_sql = f"""
CREATE OR REPLACE FUNCTION {udf_json_predictions}(
    fare_amount DOUBLE,
    trip_distance DOUBLE,
    passenger_count INT,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour INT,
    is_weekend INT
)
RETURNS STRING
COMMENT 'Returns predictions for both fare and tip category as JSON string.

Uses multiple dependencies: PyPI package (simplejson) and custom wheel from UC volume.

Example:
SELECT predict_taxi_json(25.50, 5.0, 2, 17, 4, 1, 0);

Returns JSON like:
{{"predicted_fare": 28.5, "tip_category": "Medium Tip (10-20%)", "model_version": "1.0.0"}}
'
LANGUAGE PYTHON
DETERMINISTIC
ENVIRONMENT (
    dependencies = '["simplejson==3.19.3", "{regression_wheel_path}", "{classification_wheel_path}"]',
    environment_version = 'None'
)
AS $$
import simplejson as json
from {REGRESSION_PACKAGE_NAME} import predict_fare
from {CLASSIFICATION_PACKAGE_NAME} import predict_tip_category

# Get predictions
predicted_fare = predict_fare(trip_distance, passenger_count, hour_of_day, 
                              day_of_week, is_rush_hour, is_weekend)
tip_cat = predict_tip_category(fare_amount, trip_distance, passenger_count, 
                                hour_of_day, day_of_week, is_rush_hour, is_weekend)

# Return as JSON
result = {{
    "predicted_fare": round(predicted_fare, 2),
    "tip_category": tip_cat,
    "model_version": "{PACKAGE_VERSION}",
    "input": {{
        "fare_amount": fare_amount,
        "trip_distance": trip_distance,
        "passenger_count": passenger_count
    }}
}}

return json.dumps(result)
$$;
"""

try:
    spark.sql(create_json_udf_sql)
    print(f"✓ Successfully created UDF: {udf_json_predictions}")
    print("\nThis UDF demonstrates using multiple dependencies:")
    print("  1. PyPI package: simplejson")
    print("  2. Custom wheel: regression model")
    print("  3. Custom wheel: classification model")
except Exception as e:
    print(f"✗ Error creating UDF: {e}")

# COMMAND ----------

# Test the JSON UDF
print("\nTesting JSON Prediction UDF")
print("=" * 70)

test_json_sql = f"""
SELECT 
    {udf_json_predictions}(25.50, 5.0, 2, 17, 4, 1, 0) as prediction_json
"""

result = spark.sql(test_json_sql)
display(result)

# Parse and display the JSON result
import json

json_result = result.collect()[0]['prediction_json']
parsed = json.loads(json_result)

print("\nParsed JSON result:")
print(json.dumps(parsed, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: UDF for Batch Predictions with Array Input

# COMMAND ----------

udf_batch_predict = f"{CATALOG}.{SCHEMA}.predict_fare_batch"

print(f"Creating batch prediction UDF: {udf_batch_predict}")
print("=" * 70)

create_batch_udf_sql = f"""
CREATE OR REPLACE FUNCTION {udf_batch_predict}(
    trip_distances ARRAY<DOUBLE>,
    passenger_counts ARRAY<INT>,
    hours ARRAY<INT>
)
RETURNS ARRAY<DOUBLE>
COMMENT 'Predicts fares for multiple trips at once using array inputs.

All arrays must have the same length. Uses default values for rush hour detection.

Example:
SELECT predict_fare_batch(
    ARRAY(2.0, 5.0, 10.0), 
    ARRAY(1, 2, 4), 
    ARRAY(9, 17, 20)
) AS predicted_fares;
'
LANGUAGE PYTHON
DETERMINISTIC
ENVIRONMENT (
    dependencies = '["{regression_wheel_path}"]',
    environment_version = 'None'
)
AS $$
from {REGRESSION_PACKAGE_NAME} import predict_fare

if not trip_distances or len(trip_distances) == 0:
    return []

predictions = []
for i in range(len(trip_distances)):
    dist = trip_distances[i]
    passengers = passenger_counts[i] if i < len(passenger_counts) else 1
    hour = hours[i] if i < len(hours) else 12
    
    # Determine rush hour and weekend
    is_rush = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    day_of_week = 3  # Default to Wednesday
    is_weekend = 0
    
    fare = predict_fare(dist, passengers, hour, day_of_week, is_rush, is_weekend)
    predictions.append(round(fare, 2))

return predictions
$$;
"""

try:
    spark.sql(create_batch_udf_sql)
    print(f"✓ Successfully created batch UDF: {udf_batch_predict}")
except Exception as e:
    print(f"✗ Error creating UDF: {e}")

# Test batch UDF
print("\n" + "=" * 70)
print("Testing Batch Prediction UDF")
print("=" * 70)

test_batch_sql = f"""
SELECT 
    ARRAY(2.0, 5.0, 10.0, 15.0) as distances,
    ARRAY(1, 2, 3, 4) as passengers,
    ARRAY(9, 12, 17, 20) as hours,
    {udf_batch_predict}(
        ARRAY(2.0, 5.0, 10.0, 15.0), 
        ARRAY(1, 2, 3, 4), 
        ARRAY(9, 12, 17, 20)
    ) as predicted_fares
"""

display(spark.sql(test_batch_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Cleanup and Version Management

# COMMAND ----------

# MAGIC %md
# MAGIC ### List All Created UDFs

# COMMAND ----------

print("All UDFs created in this session:")
print("=" * 70)

created_udfs = [
    udf_fare_prediction,
    udf_tip_category,
    udf_json_predictions,
    udf_batch_predict
]

for udf in created_udfs:
    try:
        # Check if UDF exists
        spark.sql(f"DESCRIBE FUNCTION {udf}")
        print(f"✓ {udf}")
    except:
        print(f"✗ {udf} (not found)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop UDFs (Optional)
# MAGIC 
# MAGIC Uncomment and run to drop the UDFs:
# MAGIC 
# MAGIC ```python
# MAGIC # Drop individual UDFs
# MAGIC # spark.sql(f"DROP FUNCTION IF EXISTS {udf_fare_prediction}")
# MAGIC # spark.sql(f"DROP FUNCTION IF EXISTS {udf_tip_category}")
# MAGIC # spark.sql(f"DROP FUNCTION IF EXISTS {udf_json_predictions}")
# MAGIC # spark.sql(f"DROP FUNCTION IF EXISTS {udf_batch_predict}")
# MAGIC 
# MAGIC # Drop all created UDFs
# MAGIC # for udf in created_udfs:
# MAGIC #     try:
# MAGIC #         spark.sql(f"DROP FUNCTION IF EXISTS {udf}")
# MAGIC #         print(f"Dropped: {udf}")
# MAGIC #     except Exception as e:
# MAGIC #         print(f"Could not drop {udf}: {e}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary and Best Practices

# COMMAND ----------

print("=" * 70)
print("SUMMARY - Python UDFs with Custom Dependencies")
print("=" * 70)

print("\n✅ What We Accomplished:")
print("-" * 70)
print("1. ✓ Created UDFs with custom wheel file dependencies from UC volumes")
print("2. ✓ Tested fare prediction UDF (regression model)")
print("3. ✓ Tested tip category prediction UDF (classification model)")
print("4. ✓ Used UDFs in SQL queries for batch inference")
print("5. ✓ Used UDFs in PySpark DataFrames")
print("6. ✓ Created advanced UDFs with multiple dependencies")
print("7. ✓ Demonstrated batch prediction with array inputs")
print("8. ✓ Saved predictions to Delta tables")

print("\n📋 Created UDFs:")
print("-" * 70)
for idx, udf in enumerate(created_udfs, 1):
    print(f"{idx}. {udf}")

print("\n🎯 Best Practices:")
print("-" * 70)
print("1. ✓ Use DETERMINISTIC for functions with consistent outputs")
print("2. ✓ Add comprehensive COMMENT with examples")
print("3. ✓ Store wheel files in organized UC volume folders")
print("4. ✓ Version your models and track in metadata")
print("5. ✓ Grant appropriate EXECUTE permissions to users")
print("6. ✓ Test UDFs with sample data before production use")
print("7. ✓ Monitor UDF performance and optimize when needed")
print("8. ✓ Document dependencies and model information")

print("\n⚡ Performance Tips:")
print("-" * 70)
print("1. Use shared isolation (default) for standard ML inference UDFs")
print("2. Only use STRICT ISOLATION when executing arbitrary code")
print("3. Cache frequently accessed predictions in Delta tables")
print("4. Batch predictions when possible using array inputs")
print("5. Consider materializing predictions for large datasets")

print("\n🔒 Security & Governance:")
print("-" * 70)
print("1. Store models in UC volumes with proper access controls")
print("2. Grant EXECUTE permission only to authorized users/groups")
print("3. Use Unity Catalog lineage to track data usage")
print("4. Version models and maintain audit trail")
print("5. Document model training data and purpose")

print("\n📚 Additional Resources:")
print("-" * 70)
print("• Official Documentation:")
print("  https://docs.databricks.com/aws/en/udf/unity-catalog")
print("\n• CREATE FUNCTION Reference:")
print("  https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html")
print("\n• Unity Catalog Privileges:")
print("  https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/")
print("\n• Unity Catalog Volumes:")
print("  https://docs.databricks.com/connect/unity-catalog/volumes.html")

print("\n" + "=" * 70)
print("✅ TUTORIAL COMPLETE!")
print("=" * 70)
print("\nYou now have production-ready UDFs for ML inference at scale!")
print("Use these UDFs in your SQL queries, dashboards, and ETL pipelines.")
print("=" * 70)
