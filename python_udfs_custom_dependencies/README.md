# Python UDFs with Custom Dependencies on Databricks

This project demonstrates how to create and use Python User-Defined Functions (UDFs) with custom dependencies in Unity Catalog on Databricks.

## Overview

Learn how to package ML models as Python wheels, store them in Unity Catalog volumes, and create UDFs that use these custom dependencies for real-time inference at scale.

**What You'll Build**:
- Two scikit-learn models (fare prediction & tip classification) trained on NYC taxi data
- Python wheel packages containing trained models
- Unity Catalog UDFs that use custom wheel files as dependencies
- Production-ready SQL and PySpark examples

## Project Structure

```
python_udfs_custom_dependencies/
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
└── notebooks/
    ├── 00_README.py                                # Project documentation (workspace view)
    ├── 01_build_ml_models_for_udfs.py             # Train models & create wheels
    ├── 02_create_udfs_with_custom_dependencies.py # Create & test UDFs (Python)
    └── 03_sql_examples_for_udfs.sql               # SQL examples & queries
```

## Prerequisites

### Databricks Requirements
- **Runtime**: DBR 18.1+ (Python UDFs with custom dependencies)
- **Compute**: Serverless, Pro SQL warehouse, or cluster with standard access mode
- **Unity Catalog**: Enabled workspace

### Required Permissions
```
Catalog:  USE CATALOG
Schema:   USE SCHEMA, CREATE FUNCTION
Volume:   CREATE VOLUME, READ VOLUME, WRITE VOLUME
```

### Python Dependencies
```
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
```

## Quick Start

### Step 1: Build ML Models & Create Wheel Files

**Notebook**: `01_build_ml_models_for_udfs.py`

**What it does**:
1. Loads NYC taxi trip data
2. Engineers features (time-based, rush hour detection)
3. Trains two Random Forest models:
   - **Regression**: Predicts fare amount
   - **Classification**: Predicts tip category (Low/Medium/High)
4. Packages models as Python wheel files
5. Uploads wheels to Unity Catalog volume

**Configure at the top**:
```python
CATALOG = "main"              # Your catalog
SCHEMA = "default"            # Your schema
VOLUME = "ml_models"          # Your volume
FOLDER_NAME = "taxi_models"   # Subfolder for organization
```

**Outputs**:
- `/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FOLDER_NAME}/nyc_taxi_fare_predictor-1.0.0-py3-none-any.whl`
- `/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FOLDER_NAME}/nyc_taxi_tip_classifier-1.0.0-py3-none-any.whl`

---

### Step 2: Create Unity Catalog UDFs

**Notebook**: `02_create_udfs_with_custom_dependencies.py`

**What it does**:
1. Verifies wheel files exist in UC volume
2. Creates Python UDFs with custom dependencies:
   - `predict_taxi_fare()` - Regression model UDF
   - `predict_tip_category()` - Classification model UDF
   - `predict_taxi_json()` - Advanced UDF with multiple dependencies
   - `predict_fare_batch()` - Batch prediction with array inputs
3. Tests UDFs with sample data
4. Demonstrates SQL and PySpark usage
5. Shows performance patterns and best practices

**What Gets Created**:
```sql
-- Basic UDFs
{CATALOG}.{SCHEMA}.predict_taxi_fare(trip_distance, passenger_count, ...) → DOUBLE
{CATALOG}.{SCHEMA}.predict_tip_category(fare_amount, trip_distance, ...) → STRING

-- Advanced UDFs
{CATALOG}.{SCHEMA}.predict_taxi_json(...) → STRING (JSON output)
{CATALOG}.{SCHEMA}.predict_fare_batch(distances ARRAY, ...) → ARRAY<DOUBLE>
```

**Usage Examples**:

**SQL**:
```sql
-- Single prediction
SELECT predict_taxi_fare(5.0, 2, 17, 4, 1, 0) AS predicted_fare;

-- Batch inference
SELECT 
    trip_id,
    actual_fare,
    predict_taxi_fare(trip_distance, passenger_count, hour_of_day, 
                     day_of_week, is_rush_hour, is_weekend) AS predicted_fare
FROM trips_table;
```

**PySpark**:
```python
from pyspark.sql.functions import expr

df_with_predictions = df.withColumn(
    "predicted_fare",
    expr("predict_taxi_fare(trip_distance, passenger_count, hour_of_day, day_of_week, is_rush_hour, is_weekend)")
)
```

---

### Step 3: Use UDFs in Databricks SQL

**Notebook**: `03_sql_examples_for_udfs.sql`

**What it includes**:
- 40+ SQL query examples
- Sample test data (15 taxi trips)
- Basic predictions with hardcoded values
- Batch predictions on sample data
- Aggregations by time period, day type, rush hour
- Dashboard-ready queries (charts, KPIs)
- Performance optimization tips (CTEs, caching)

**Perfect For**:
- Databricks SQL Warehouses
- SQL Dashboards & visualizations
- BI Tools (Tableau, Power BI) via JDBC/ODBC
- Scheduled SQL queries
- SQL analysts & users

---

## ⚠️ Public Preview Limitations

Python UDFs in Unity Catalog with custom dependencies are currently in **Public Preview** with the following limitations:

### 5 UDF Invocations Per Query Limit

**What this means**:
- SQL queries can invoke UDFs **up to 5 times per query**
- Example: `SELECT udf(col) FROM table LIMIT 10` would call the UDF 10 times → ❌ Exceeds limit
- Aggregations often work because they group first: `SELECT AVG(udf(col)) FROM table GROUP BY category` → ✅ Likely OK

**Impact on Notebook 3** (SQL examples):
- Some queries use `LIMIT 5` to stay within this constraint
- This is temporary and will be removed at General Availability

### Workarounds for Production:

1. **Use PySpark DataFrames** (Recommended):
   ```python
   # No UDF invocation limit in PySpark!
   df.withColumn("prediction", expr("predict_taxi_fare(...)"))  # Works on millions of rows
   ```

2. **Batch Processing in SQL**:
   ```sql
   -- Process in chunks
   SELECT ... WHERE trip_id BETWEEN 1 AND 5;  -- Batch 1
   SELECT ... WHERE trip_id BETWEEN 6 AND 10; -- Batch 2
   ```

3. **Cache Predictions**:
   ```sql
   -- Pre-compute predictions
   CREATE TABLE predictions AS 
   SELECT *, predict_taxi_fare(...) as pred FROM trips LIMIT 5;
   -- Then query cached table (no limit!)
   SELECT * FROM predictions;
   ```

4. **Wait for General Availability** - The 5 UDF limit will be removed

### Documentation
For the latest information on Python UDFs and preview limitations:
- **Python UDFs in Unity Catalog**: https://docs.databricks.com/en/udf/unity-catalog.html
- **Python UDF Limitations**: https://docs.databricks.com/en/udf/unity-catalog.html#limitations

---

## How It Works

### 1. Model Training & Packaging

**Notebook 1** trains scikit-learn models and packages them:

```
NYC Taxi Data → Feature Engineering → Model Training → Python Package → Wheel File → UC Volume
```

**Package Structure**:
```
nyc_taxi_fare_predictor/
├── setup.py                    # Package metadata
├── nyc_taxi_fare_predictor/
│   ├── __init__.py            # Exports predict_fare()
│   ├── predictor.py           # FarePredictor class
│   └── model.pkl              # Trained model (pickled)
└── README.md                   # Usage instructions
```

### 2. UDF Creation with Custom Dependencies

**Notebook 2** creates UDFs that reference wheel files:

```sql
CREATE FUNCTION predict_taxi_fare(...)
RETURNS DOUBLE
LANGUAGE PYTHON
ENVIRONMENT (
    dependencies = '["/Volumes/catalog/schema/volume/folder/predictor-1.0.0-py3-none-any.whl"]'
)
AS $$
from nyc_taxi_fare_predictor import predict_fare
return predict_fare(trip_distance, passenger_count, ...)
$$;
```

**Key Points**:
- UDF code is executed in isolated Python environment
- Dependencies are installed when UDF is created
- Wheel files must be accessible from UC volumes
- Multiple dependencies can be specified (PyPI packages + wheels)

### 3. Inference at Scale

**Notebook 2 & 3** demonstrate inference:

```
SQL Query → UDF Invocation → Python Environment → Load Model → Predict → Return Result
```

**Performance**:
- First call: Environment setup (1-2 seconds)
- Subsequent calls: Cached environment (milliseconds)
- Deterministic UDFs are optimized by Spark

---

## Key Features

### Multiple Dependency Types
```sql
ENVIRONMENT (
    dependencies = '[
        "simplejson==3.19.3",                                    -- PyPI package
        "/Volumes/catalog/schema/volume/my_package.whl",        -- UC Volume wheel
        "https://example.com/package.whl"                        -- Public URL
    ]'
)
```

### Environment Isolation Modes

**Shared Isolation** (Default - Recommended):
```sql
CREATE FUNCTION my_udf(...) 
LANGUAGE PYTHON
AS $$ ... $$;
```
- UDFs with same owner/session share environment
- Lower memory usage, faster execution
- Use for ML inference, data transformations

**Strict Isolation** (Use with caution):
```sql
CREATE FUNCTION my_udf(...) 
LANGUAGE PYTHON
STRICT ISOLATION
AS $$ ... $$;
```
- Each UDF call gets isolated environment
- Higher overhead
- Use only for code that writes files, uses eval(), modifies globals

### Deterministic UDFs
```sql
CREATE FUNCTION my_udf(...)
RETURNS DOUBLE
LANGUAGE PYTHON
DETERMINISTIC  -- Spark can optimize/cache results
AS $$ ... $$;
```

---

## Models

### 1. Fare Prediction (Regression)
- **Algorithm**: Random Forest Regressor
- **Input Features**: trip_distance, passenger_count, hour_of_day, day_of_week, is_rush_hour, is_weekend
- **Output**: Predicted fare amount (DOUBLE)
- **Use Case**: Revenue forecasting, pricing optimization

### 2. Tip Category Prediction (Classification)
- **Algorithm**: Random Forest Classifier
- **Input Features**: fare_amount, trip_distance, passenger_count, hour_of_day, day_of_week, is_rush_hour, is_weekend
- **Output**: Tip category - "Low Tip (<10%)", "Medium Tip (10-20%)", "High Tip (>20%)"
- **Use Case**: Driver incentives, demand prediction

---

## Common Issues & Solutions

### Issue: "Module not found"
**Cause**: Wheel file path is incorrect or not accessible  
**Solution**: 
```python
# Verify wheel exists
dbutils.fs.ls("/Volumes/catalog/schema/volume/folder/")
# Ensure path in UDF matches exactly
```

### Issue: "Permission denied accessing volume"
**Cause**: Missing READ VOLUME permission  
**Solution**:
```sql
GRANT READ VOLUME ON VOLUME catalog.schema.volume TO `user@example.com`;
```

### Issue: "Maximum UDF invocations (5) exceeded"
**Cause**: Public preview limit  
**Solution**: Use PySpark DataFrames (no limit) or add `LIMIT 5` to SQL queries

### Issue: "dbutils.fs.cp() fails on shared cluster"
**Cause**: USER_ISOLATION mode restricts /tmp access  
**Solution**: Copy to /Workspace first (fixed in Notebook 1, Cell 35)

---

## Best Practices

### 1. Version Your Models
```python
PACKAGE_VERSION = "1.0.0"  # Update when model changes
```

### 2. Document Your UDFs
```sql
CREATE FUNCTION predict_fare(...)
COMMENT 'Predicts taxi fare using Random Forest model.
Version: 1.0.0
Trained: 2026-01-26
MAE: $2.50, RMSE: $3.80, R²: 0.85'
AS $$ ... $$;
```

### 3. Use DETERMINISTIC for Pure Functions
```sql
CREATE FUNCTION my_udf(...) DETERMINISTIC AS $$ ... $$;
```

### 4. Handle NULLs Explicitly
```python
AS $$
if trip_distance is None:
    return None
# ... rest of logic
$$;
```

### 5. Cache Predictions for Large Datasets
```python
# Pre-compute predictions
df_with_predictions.write.saveAsTable("predictions_cache")
# Query cached table
spark.table("predictions_cache")
```

---

## Additional Resources

### Official Documentation
- **Python UDFs in Unity Catalog**: https://docs.databricks.com/en/udf/unity-catalog.html
- **CREATE FUNCTION Reference**: https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html
- **Unity Catalog Privileges**: https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/
- **Unity Catalog Volumes**: https://docs.databricks.com/connect/unity-catalog/volumes.html

### Community & Support
- **Databricks Community**: https://community.databricks.com/
- **Knowledge Base**: https://kb.databricks.com/
- **Databricks Academy**: https://www.databricks.com/learn/training

---

## Author

**Prasad Kona**

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](../LICENSE) file for details.

**You are free to:**
- ✅ Use this code commercially
- ✅ Modify and distribute
- ✅ Use in private projects
- ✅ Use for any purpose

This is open-source software - feel free to reuse, adapt, and build upon this example!

---

**Last Updated**: January 26, 2026  
**Author**: Prasad Kona  
**Databricks Runtime**: 18.1+  
**Status**: Production-ready with public preview limitations

**Note**: Update `CATALOG`, `SCHEMA`, `VOLUME`, and `FOLDER_NAME` variables in notebooks to match your environment.
