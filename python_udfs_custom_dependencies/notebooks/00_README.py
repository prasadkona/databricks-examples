# Databricks notebook source
# MAGIC %md
# MAGIC # Python UDFs with Custom Dependencies on Databricks
# MAGIC 
# MAGIC **Author**: Prasad Kona  
# MAGIC **Last Updated**: January 26, 2026  
# MAGIC **Notebook**: 00 - Project README
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This project demonstrates how to use Python User-Defined Functions (UDFs) with custom dependencies in Unity Catalog on Databricks.
# MAGIC 
# MAGIC ## 📚 Official Documentation
# MAGIC 
# MAGIC - **Python UDFs in Unity Catalog**: https://docs.databricks.com/en/udf/unity-catalog.html
# MAGIC - **CREATE FUNCTION Reference**: https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html
# MAGIC - **Unity Catalog Privileges**: https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/
# MAGIC - **Unity Catalog Volumes**: https://docs.databricks.com/connect/unity-catalog/volumes.html
# MAGIC 
# MAGIC ## 📓 Notebooks
# MAGIC 
# MAGIC ### 1. Build ML Models for UDFs
# MAGIC 
# MAGIC **Notebook**: `01_build_ml_models_for_udfs`
# MAGIC 
# MAGIC **Purpose**: Train ML models, package as Python wheels, upload to Unity Catalog volumes
# MAGIC 
# MAGIC **What it does**:
# MAGIC - Loads NYC taxi trip data (or creates synthetic data)
# MAGIC - Engineers features for ML models
# MAGIC - Trains two models:
# MAGIC   - **Regression**: Predicts taxi fare amounts
# MAGIC   - **Classification**: Predicts tip categories (Low/Medium/High)
# MAGIC - Creates proper Python package structure
# MAGIC - Builds distributable wheel files
# MAGIC - Uploads wheels to Unity Catalog volume
# MAGIC - Tests packages locally
# MAGIC 
# MAGIC **Outputs**:
# MAGIC - `nyc_taxi_fare_predictor-1.0.0-py3-none-any.whl`
# MAGIC - `nyc_taxi_tip_classifier-1.0.0-py3-none-any.whl`
# MAGIC - Both uploaded to UC volume for use in UDFs
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ### 2. Create and Use UDFs with Custom Dependencies
# MAGIC 
# MAGIC **Notebook**: `02_create_udfs_with_custom_dependencies`
# MAGIC 
# MAGIC **Purpose**: Create UDFs using wheel files and demonstrate usage patterns
# MAGIC 
# MAGIC **What it does**:
# MAGIC - Verifies wheel files in UC volume
# MAGIC - Creates Python UDFs with custom dependencies:
# MAGIC   - `predict_taxi_fare()` - Fare prediction
# MAGIC   - `predict_tip_category()` - Tip category classification
# MAGIC   - `predict_taxi_json()` - Advanced UDF with multiple dependencies
# MAGIC   - `predict_fare_batch()` - Batch predictions with arrays
# MAGIC - Demonstrates SQL usage patterns
# MAGIC - Demonstrates PySpark DataFrame usage
# MAGIC - Shows permission management
# MAGIC - Provides performance benchmarking
# MAGIC - Includes best practices and production tips
# MAGIC 
# MAGIC **Examples Covered**:
# MAGIC - Single predictions in SQL
# MAGIC - Batch inference on tables
# MAGIC - Using UDFs in PySpark with `expr()`
# MAGIC - Filtering and aggregation with UDF results
# MAGIC - Saving predictions to Delta tables
# MAGIC - Multiple dependencies (PyPI + UC volumes)
# MAGIC - JSON output formatting
# MAGIC 
# MAGIC ## 🚀 Quick Start
# MAGIC 
# MAGIC ### Step 1: Configuration
# MAGIC 
# MAGIC Update these variables in both notebooks:
# MAGIC 
# MAGIC ```python
# MAGIC CATALOG = "main"              # Your catalog name
# MAGIC SCHEMA = "default"            # Your schema name
# MAGIC VOLUME = "ml_models"          # Volume name
# MAGIC FOLDER_NAME = "taxi_models"   # Folder within volume
# MAGIC ```
# MAGIC 
# MAGIC ### Step 2: Run Notebook 1
# MAGIC 
# MAGIC Run `01_build_ml_models_for_udfs` to:
# MAGIC - Train models on NYC taxi data
# MAGIC - Build and upload wheel files to UC volume
# MAGIC - Get ready-to-use paths for UDF creation
# MAGIC 
# MAGIC ### Step 3: Run Notebook 2
# MAGIC 
# MAGIC Run `02_create_udfs_with_custom_dependencies` to:
# MAGIC - Create UDFs with custom dependencies
# MAGIC - Test with sample data
# MAGIC - Learn SQL and PySpark usage patterns
# MAGIC - Explore advanced examples
# MAGIC 
# MAGIC ## 📦 Models
# MAGIC 
# MAGIC ### Regression Model - Fare Prediction
# MAGIC - **Purpose**: Predict NYC taxi fare amounts
# MAGIC - **Algorithm**: Random Forest Regressor
# MAGIC - **Features**: 
# MAGIC   - trip_distance
# MAGIC   - passenger_count
# MAGIC   - hour_of_day
# MAGIC   - day_of_week
# MAGIC   - is_rush_hour
# MAGIC   - is_weekend
# MAGIC - **Package**: `nyc_taxi_fare_predictor`
# MAGIC 
# MAGIC ### Classification Model - Tip Category Prediction
# MAGIC - **Purpose**: Classify tips into Low (<10%), Medium (10-20%), or High (>20%) categories
# MAGIC - **Algorithm**: Random Forest Classifier
# MAGIC - **Features**: 
# MAGIC   - fare_amount
# MAGIC   - trip_distance
# MAGIC   - passenger_count
# MAGIC   - hour_of_day
# MAGIC   - day_of_week
# MAGIC   - is_rush_hour
# MAGIC   - is_weekend
# MAGIC - **Package**: `nyc_taxi_tip_classifier`
# MAGIC 
# MAGIC ## ✅ Prerequisites
# MAGIC 
# MAGIC ### Databricks Requirements
# MAGIC - **Runtime**: Databricks Runtime 18.1 or above
# MAGIC - **Compute**: Serverless compute, Pro SQL warehouse, or cluster with standard access mode
# MAGIC - **Unity Catalog**: Enabled workspace
# MAGIC 
# MAGIC ### Permissions Required
# MAGIC - `USE CATALOG` on target catalog
# MAGIC - `USE SCHEMA` on target schema
# MAGIC - `CREATE FUNCTION` on target schema
# MAGIC - `CREATE VOLUME` on target schema
# MAGIC - `READ VOLUME` and `WRITE VOLUME` on target volume
# MAGIC 
# MAGIC ### Python Dependencies
# MAGIC ```
# MAGIC scikit-learn>=1.0.0
# MAGIC numpy>=1.21.0
# MAGIC pandas>=1.3.0
# MAGIC ```
# MAGIC 
# MAGIC ## 🎯 Key Features Demonstrated
# MAGIC 
# MAGIC ### 1. Multiple Dependency Sources
# MAGIC - **PyPI packages**: `simplejson==3.19.3`
# MAGIC - **UC Volume wheels**: `/Volumes/catalog/schema/volume/package.whl`
# MAGIC - **Public URLs**: `https://.../*.whl`
# MAGIC - **Mixed dependencies**: Combining all three sources
# MAGIC 
# MAGIC ### 2. Environment Isolation
# MAGIC - **Shared Isolation** (default): Better performance for standard UDFs
# MAGIC - **STRICT ISOLATION**: For UDFs that execute code, modify state, or write files
# MAGIC 
# MAGIC ### 3. Best Practices
# MAGIC - Using `DETERMINISTIC` for consistent outputs
# MAGIC - Proper error handling in UDFs
# MAGIC - Comprehensive documentation with `COMMENT`
# MAGIC - Version tracking and changelog
# MAGIC 
# MAGIC ### 4. Real-world Use Cases
# MAGIC - ML model inference at scale
# MAGIC - Custom data transformations
# MAGIC - External API integrations
# MAGIC - Security and compliance (data masking, encryption)
# MAGIC 
# MAGIC ## 📊 Example Usage
# MAGIC 
# MAGIC ### SQL Example
# MAGIC 
# MAGIC ```sql
# MAGIC -- Predict fare for a single trip
# MAGIC SELECT predict_taxi_fare(5.0, 2, 17, 4, 1, 0) AS predicted_fare;
# MAGIC 
# MAGIC -- Use with table data
# MAGIC SELECT 
# MAGIC     trip_id,
# MAGIC     trip_distance,
# MAGIC     actual_fare,
# MAGIC     predict_taxi_fare(
# MAGIC         trip_distance, 
# MAGIC         passenger_count, 
# MAGIC         hour_of_day, 
# MAGIC         day_of_week,
# MAGIC         is_rush_hour,
# MAGIC         is_weekend
# MAGIC     ) AS predicted_fare
# MAGIC FROM trips_table;
# MAGIC ```
# MAGIC 
# MAGIC ### PySpark Example
# MAGIC 
# MAGIC ```python
# MAGIC from pyspark.sql.functions import expr
# MAGIC 
# MAGIC df_with_predictions = df.withColumn(
# MAGIC     "predicted_fare",
# MAGIC     expr("""predict_taxi_fare(
# MAGIC         trip_distance, 
# MAGIC         passenger_count, 
# MAGIC         hour_of_day, 
# MAGIC         day_of_week,
# MAGIC         is_rush_hour,
# MAGIC         is_weekend
# MAGIC     )""")
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ## 💡 Performance Tips
# MAGIC 
# MAGIC 1. Use shared isolation (default) for standard ML inference UDFs
# MAGIC 2. Only use STRICT ISOLATION when executing arbitrary code
# MAGIC 3. Cache frequently accessed predictions in Delta tables
# MAGIC 4. Batch predictions when possible using array inputs
# MAGIC 5. Consider materializing predictions for large datasets
# MAGIC 
# MAGIC ## 🔒 Security & Governance
# MAGIC 
# MAGIC 1. Store models in UC volumes with proper access controls
# MAGIC 2. Grant EXECUTE permission only to authorized users/groups
# MAGIC 3. Use Unity Catalog lineage to track data usage
# MAGIC 4. Version models and maintain audit trail
# MAGIC 5. Document model training data and purpose
# MAGIC 
# MAGIC ## 🐛 Common Issues and Solutions
# MAGIC 
# MAGIC ### Issue: "Module not found"
# MAGIC **Solution**: Ensure wheel file path is correct and accessible in UC volume
# MAGIC 
# MAGIC ### Issue: "Permission denied"
# MAGIC **Solution**: Verify you have `READ VOLUME` permission on the source volume
# MAGIC 
# MAGIC ### Issue: Network connectivity errors
# MAGIC **Solution**: For serverless SQL warehouses, enable "Enable networking for UDFs in Serverless SQL Warehouses" in workspace Previews
# MAGIC 
# MAGIC ### Issue: Dependency conflicts
# MAGIC **Solution**: Pin exact versions in the `dependencies` list
# MAGIC 
# MAGIC ## 📝 Limitations
# MAGIC 
# MAGIC - Maximum 5 UDFs per query
# MAGIC - Python UDFs execute in isolated environment
# MAGIC - No access to internal services or file systems (unless using volumes)
# MAGIC - Must handle NULL values explicitly
# MAGIC - Type mappings must follow Databricks SQL language mappings
# MAGIC 
# MAGIC ## 📧 Support
# MAGIC 
# MAGIC For issues or questions:
# MAGIC 1. Check the [Databricks Community](https://community.databricks.com/)
# MAGIC 2. Review the [Knowledge Base](https://kb.databricks.com/)
# MAGIC 3. Contact Databricks Support
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Ready to get started?** Open notebook `01_build_ml_models_for_udfs` and let's build some ML-powered UDFs! 🚀
