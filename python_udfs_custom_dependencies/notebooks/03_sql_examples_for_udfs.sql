-- Databricks notebook source
-- MAGIC %md
-- MAGIC # SQL Examples for Python UDFs with Custom Dependencies
-- MAGIC
-- MAGIC **Author**: Prasad Kona  
-- MAGIC **Last Updated**: January 26, 2026  
-- MAGIC **Notebook**: 03 - SQL Examples for UDFs
-- MAGIC
-- MAGIC ---
-- MAGIC
-- MAGIC ## Overview
-- MAGIC
-- MAGIC This SQL notebook demonstrates how to use the Python UDFs created in notebooks 1 and 2 in pure SQL queries.
-- MAGIC
-- MAGIC ## What You'll Learn
-- MAGIC
-- MAGIC 1. **Basic UDF Usage** - Simple predictions with hardcoded values
-- MAGIC 2. **Sample Data Queries** - Using UDFs with test data
-- MAGIC 3. **Batch Predictions** - Apply UDFs to multiple rows
-- MAGIC 4. **Aggregations** - Group by and aggregate with UDF results
-- MAGIC 5. **Dashboard Queries** - Production-ready queries for dashboards
-- MAGIC 6. **Performance Tips** - Optimize UDF usage in SQL
-- MAGIC
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC - Complete notebook `01_build_ml_models_for_udfs` (wheel files uploaded)
-- MAGIC - Complete notebook `02_create_udfs_with_custom_dependencies` (UDFs created)
-- MAGIC - Databricks SQL Warehouse or Cluster with DBR 18.1+
-- MAGIC
-- MAGIC ## Configuration
-- MAGIC
-- MAGIC Update these values to match your Unity Catalog setup:
-- MAGIC - **Catalog**: `main` (replace with your catalog name)
-- MAGIC - **Schema**: `default` (replace with your schema name)
-- MAGIC - **UDFs**:
-- MAGIC   - `main.default.predict_taxi_fare`
-- MAGIC   - `main.default.predict_tip_category`
-- MAGIC
-- MAGIC ## Documentation
-- MAGIC
-- MAGIC - **Python UDFs in Unity Catalog**: https://docs.databricks.com/en/udf/unity-catalog.html
-- MAGIC - **Databricks SQL**: https://docs.databricks.com/sql/index.html
-- MAGIC
-- MAGIC ## Important Note
-- MAGIC
-- MAGIC ⚠️ Python UDFs in Unity Catalog are in **public preview** with a limit of **5 UDF invocations per query**.  
-- MAGIC Some queries use `LIMIT 5` to stay within this constraint. This limit will be removed at GA.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Setup - Set Catalog and Schema Context

-- COMMAND ----------

-- Set the catalog and schema context for this session
USE CATALOG main;
USE SCHEMA default;

-- Verify we're in the correct context
SELECT current_catalog() as current_catalog, current_schema() as current_schema;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Verify UDFs Exist

-- COMMAND ----------

-- List all UDFs in the current schema
SHOW FUNCTIONS LIKE 'predict*';

-- COMMAND ----------

-- Get detailed information about the fare prediction UDF
DESCRIBE FUNCTION EXTENDED predict_taxi_fare;

-- COMMAND ----------

-- Get detailed information about the tip category UDF
DESCRIBE FUNCTION EXTENDED predict_tip_category;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Basic UDF Usage - Single Predictions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Test 1: Predict Fare for a Short Weekday Trip

-- COMMAND ----------

-- Short trip: 2 miles, 1 passenger, 10am Wednesday, no rush hour
SELECT 
  'Short weekday trip' as scenario,
  2.0 as trip_distance,
  1 as passenger_count,
  '10am Wednesday' as trip_time,
  predict_taxi_fare(
    2.0,  -- trip_distance
    1,    -- passenger_count
    10,   -- hour_of_day
    2,    -- day_of_week (Wednesday)
    0,    -- is_rush_hour
    0     -- is_weekend
  ) as predicted_fare_usd;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Test 2: Predict Fare for a Rush Hour Trip

-- COMMAND ----------

-- Rush hour trip: 8 miles, 3 passengers, 5pm Friday
SELECT 
  'Rush hour Friday trip' as scenario,
  8.0 as trip_distance,
  3 as passenger_count,
  '5pm Friday' as trip_time,
  predict_taxi_fare(
    8.0,  -- trip_distance
    3,    -- passenger_count  
    17,   -- hour_of_day (5pm)
    4,    -- day_of_week (Friday)
    1,    -- is_rush_hour
    0     -- is_weekend
  ) as predicted_fare_usd;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Test 3: Predict Fare for a Weekend Trip

-- COMMAND ----------

-- Weekend trip: 5 miles, 2 passengers, 2pm Saturday
SELECT 
  'Weekend trip' as scenario,
  5.0 as trip_distance,
  2 as passenger_count,
  '2pm Saturday' as trip_time,
  predict_taxi_fare(
    5.0,  -- trip_distance
    2,    -- passenger_count
    14,   -- hour_of_day (2pm)
    5,    -- day_of_week (Saturday)
    0,    -- is_rush_hour
    1     -- is_weekend
  ) as predicted_fare_usd;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Test 4: Predict Tip Category

-- COMMAND ----------

-- Predict tip category for a medium fare trip
SELECT 
  'Medium fare trip' as scenario,
  25.50 as fare_amount,
  5.0 as trip_distance,
  predict_tip_category(
    25.50,  -- fare_amount
    5.0,    -- trip_distance
    2,      -- passenger_count
    17,     -- hour_of_day (5pm)
    4,      -- day_of_week (Friday)
    1,      -- is_rush_hour
    0       -- is_weekend
  ) as predicted_tip_category;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Create Sample Test Data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Create a Temporary View with Sample Taxi Trips

-- COMMAND ----------

-- Create sample taxi trip data for testing
CREATE OR REPLACE TEMPORARY VIEW sample_taxi_trips AS
SELECT 
  row_number() OVER (ORDER BY trip_distance) as trip_id,
  trip_distance,
  passenger_count,
  hour_of_day,
  day_of_week,
  is_rush_hour,
  is_weekend,
  actual_fare
FROM (
  -- Morning commute trips
  SELECT 3.2 as trip_distance, 1 as passenger_count, 8 as hour_of_day, 1 as day_of_week, 1 as is_rush_hour, 0 as is_weekend, 15.50 as actual_fare UNION ALL
  SELECT 5.5, 1, 9, 2, 1, 0, 22.00 UNION ALL
  SELECT 2.8, 2, 7, 3, 1, 0, 12.75 UNION ALL
  
  -- Midday trips
  SELECT 4.0, 2, 12, 1, 0, 0, 18.00 UNION ALL
  SELECT 6.5, 3, 13, 2, 0, 0, 26.50 UNION ALL
  SELECT 3.5, 1, 14, 3, 0, 0, 16.25 UNION ALL
  
  -- Evening rush hour trips
  SELECT 7.2, 1, 17, 4, 1, 0, 30.00 UNION ALL
  SELECT 4.8, 2, 18, 4, 1, 0, 21.50 UNION ALL
  SELECT 6.0, 3, 19, 4, 1, 0, 25.00 UNION ALL
  
  -- Weekend trips
  SELECT 8.5, 2, 14, 5, 0, 1, 35.00 UNION ALL
  SELECT 5.0, 3, 16, 5, 0, 1, 22.50 UNION ALL
  SELECT 10.2, 4, 20, 6, 0, 1, 42.00 UNION ALL
  
  -- Late night trips
  SELECT 4.5, 1, 23, 5, 0, 1, 20.00 UNION ALL
  SELECT 6.0, 2, 1, 6, 0, 1, 26.00 UNION ALL
  SELECT 3.0, 1, 2, 0, 0, 0, 14.50
);

-- Verify sample data
SELECT 
  COUNT(*) as total_trips,
  ROUND(AVG(trip_distance), 2) as avg_distance,
  ROUND(AVG(actual_fare), 2) as avg_fare
FROM sample_taxi_trips;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### View Sample Data

-- COMMAND ----------

SELECT * FROM sample_taxi_trips ORDER BY trip_id LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Batch Predictions on Sample Data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Predict Fares for All Sample Trips

-- COMMAND ----------

SELECT 
  trip_id,
  trip_distance,
  passenger_count,
  CASE 
    WHEN hour_of_day < 6 THEN 'Night'
    WHEN hour_of_day < 12 THEN 'Morning'
    WHEN hour_of_day < 18 THEN 'Afternoon'
    ELSE 'Evening'
  END as time_period,
  CASE day_of_week
    WHEN 0 THEN 'Monday'
    WHEN 1 THEN 'Tuesday'
    WHEN 2 THEN 'Wednesday'
    WHEN 3 THEN 'Thursday'
    WHEN 4 THEN 'Friday'
    WHEN 5 THEN 'Saturday'
    WHEN 6 THEN 'Sunday'
  END as day_name,
  actual_fare,
  ROUND(predict_taxi_fare(
    trip_distance, 
    passenger_count, 
    hour_of_day, 
    day_of_week, 
    is_rush_hour, 
    is_weekend
  ), 2) as predicted_fare,
  ROUND(actual_fare - predict_taxi_fare(
    trip_distance, 
    passenger_count, 
    hour_of_day, 
    day_of_week, 
    is_rush_hour, 
    is_weekend
  ), 2) as prediction_error
FROM sample_taxi_trips
ORDER BY trip_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Predict Tip Categories for All Sample Trips

-- COMMAND ----------

SELECT 
  trip_id,
  actual_fare,
  trip_distance,
  passenger_count,
  CASE is_rush_hour
    WHEN 1 THEN 'Rush Hour'
    ELSE 'Non-Rush'
  END as rush_status,
  CASE is_weekend
    WHEN 1 THEN 'Weekend'
    ELSE 'Weekday'
  END as day_type,
  predict_tip_category(
    actual_fare,
    trip_distance, 
    passenger_count, 
    hour_of_day, 
    day_of_week, 
    is_rush_hour, 
    is_weekend
  ) as predicted_tip_category
FROM sample_taxi_trips
ORDER BY actual_fare DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Aggregation Queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Average Predicted Fare by Time Period

-- COMMAND ----------

-- MAGIC %md
-- MAGIC %undefined
-- MAGIC **⚠️ Note on UDF Limits During Public Preview**
-- MAGIC
-- MAGIC Python UDFs in Unity Catalog are currently in public preview with a limit of **5 UDF invocations per query**. This means:
-- MAGIC * Queries can only process up to 5 rows when using UDFs
-- MAGIC * The following queries use `LIMIT 5` to stay within this constraint
-- MAGIC * Once the feature is generally available, this limit will be removed
-- MAGIC
-- MAGIC For production use cases with larger datasets, consider:
-- MAGIC * Using batch processing with multiple smaller queries
-- MAGIC * Waiting for general availability when limits are lifted
-- MAGIC * Using Spark DataFrames with UDFs instead of SQL (no limit)

-- COMMAND ----------

SELECT 
  CASE 
    WHEN hour_of_day < 6 THEN 'Night (12am-6am)'
    WHEN hour_of_day < 12 THEN 'Morning (6am-12pm)'
    WHEN hour_of_day < 18 THEN 'Afternoon (12pm-6pm)'
    ELSE 'Evening (6pm-12am)'
  END as time_period,
  COUNT(*) as trip_count,
  ROUND(AVG(trip_distance), 2) as avg_distance_miles,
  ROUND(AVG(actual_fare), 2) as avg_actual_fare,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as avg_predicted_fare
FROM sample_taxi_trips
GROUP BY time_period
ORDER BY avg_predicted_fare DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Tip Category Distribution by Day Type

-- COMMAND ----------

-- DBTITLE 1,Cell 31
-- Note: Limited to 5 rows due to public preview UDF limit of 5 UDFs per query
WITH limited_trips AS (
  SELECT * FROM sample_taxi_trips LIMIT 5
),
trip_predictions AS (
  SELECT 
    CASE is_weekend
      WHEN 1 THEN 'Weekend'
      ELSE 'Weekday'
    END as day_type,
    predict_tip_category(
      actual_fare, trip_distance, passenger_count, 
      hour_of_day, day_of_week, is_rush_hour, is_weekend
    ) as tip_category,
    actual_fare
  FROM limited_trips
)
SELECT 
  day_type,
  tip_category,
  COUNT(*) as trip_count,
  ROUND(AVG(actual_fare), 2) as avg_fare
FROM trip_predictions
GROUP BY day_type, tip_category
ORDER BY day_type, tip_category;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Rush Hour Impact Analysis

-- COMMAND ----------

-- DBTITLE 1,Cell 33
-- Note: Limited to 5 rows due to public preview UDF limit
WITH limited_trips AS (
  SELECT * FROM sample_taxi_trips LIMIT 5
)
SELECT 
  CASE is_rush_hour
    WHEN 1 THEN 'Rush Hour'
    ELSE 'Non-Rush Hour'
  END as rush_status,
  COUNT(*) as trip_count,
  ROUND(AVG(trip_distance), 2) as avg_distance,
  ROUND(AVG(actual_fare), 2) as avg_actual_fare,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as avg_predicted_fare,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ) - actual_fare), 2) as avg_prediction_error
FROM limited_trips
GROUP BY rush_status
ORDER BY rush_status;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 7. Advanced Queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Fare Prediction Accuracy Metrics

-- COMMAND ----------

-- DBTITLE 1,Cell 36
-- Note: Limited to 5 rows due to public preview UDF limit
WITH limited_trips AS (
  SELECT * FROM sample_taxi_trips LIMIT 5
)
SELECT 
  COUNT(*) as total_trips,
  ROUND(AVG(ABS(actual_fare - predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ))), 2) as mean_absolute_error,
  ROUND(SQRT(AVG(POW(actual_fare - predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ), 2))), 2) as root_mean_squared_error,
  ROUND(MIN(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as min_predicted_fare,
  ROUND(MAX(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as max_predicted_fare
FROM limited_trips;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Trips with Largest Prediction Errors

-- COMMAND ----------

-- DBTITLE 1,Cell 38
-- Note: Limited to 5 rows due to public preview UDF limit
WITH limited_trips AS (
  SELECT * FROM sample_taxi_trips LIMIT 5
)
SELECT 
  trip_id,
  trip_distance,
  passenger_count,
  actual_fare,
  ROUND(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ), 2) as predicted_fare,
  ROUND(ABS(actual_fare - predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as absolute_error,
  ROUND((actual_fare - predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )) / actual_fare * 100, 2) as error_percentage
FROM limited_trips
ORDER BY absolute_error DESC
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Passenger Count Impact on Fares

-- COMMAND ----------

-- DBTITLE 1,Cell 40
-- Note: Limited to 5 rows due to public preview UDF limit
WITH limited_trips AS (
  SELECT * FROM sample_taxi_trips LIMIT 5
)
SELECT 
  passenger_count,
  COUNT(*) as trip_count,
  ROUND(AVG(trip_distance), 2) as avg_distance,
  ROUND(AVG(actual_fare), 2) as avg_actual_fare,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as avg_predicted_fare,
  ROUND(AVG(actual_fare) / AVG(trip_distance), 2) as actual_fare_per_mile,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )) / AVG(trip_distance), 2) as predicted_fare_per_mile
FROM limited_trips
GROUP BY passenger_count
ORDER BY passenger_count;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 8. Dashboard-Ready Queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Query 1: Daily Fare Prediction Summary

-- COMMAND ----------

-- This query can be used in a dashboard to show daily predictions
SELECT 
  CASE day_of_week
    WHEN 0 THEN 'Monday'
    WHEN 1 THEN 'Tuesday'
    WHEN 2 THEN 'Wednesday'
    WHEN 3 THEN 'Thursday'
    WHEN 4 THEN 'Friday'
    WHEN 5 THEN 'Saturday'
    WHEN 6 THEN 'Sunday'
  END as day_name,
  day_of_week,
  COUNT(*) as trips,
  ROUND(AVG(trip_distance), 1) as avg_miles,
  ROUND(AVG(actual_fare), 2) as avg_actual,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as avg_predicted
FROM sample_taxi_trips
GROUP BY day_of_week
ORDER BY day_of_week;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Query 2: Tip Category Distribution (Pie Chart Ready)

-- COMMAND ----------

SELECT 
  predict_tip_category(
    actual_fare, trip_distance, passenger_count, 
    hour_of_day, day_of_week, is_rush_hour, is_weekend
  ) as tip_category,
  COUNT(*) as trip_count,
  ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sample_taxi_trips), 1) as percentage
FROM sample_taxi_trips
GROUP BY tip_category
ORDER BY trip_count DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Query 3: Hourly Fare Trends

-- COMMAND ----------

SELECT 
  hour_of_day,
  CASE 
    WHEN hour_of_day BETWEEN 7 AND 9 THEN 'Morning Rush'
    WHEN hour_of_day BETWEEN 17 AND 19 THEN 'Evening Rush'
    ELSE 'Normal'
  END as period_type,
  COUNT(*) as trips,
  ROUND(AVG(predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  )), 2) as avg_predicted_fare
FROM sample_taxi_trips
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 9. Using UDFs with Real NYC Taxi Data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Example: Apply UDFs to samples.nyctaxi.trips

-- COMMAND ----------

-- If you have access to samples.nyctaxi.trips, you can use this query
-- Note: This requires appropriate permissions and the table may not exist in all workspaces

-- Uncomment to test with real data:
/*
SELECT 
  tpep_pickup_datetime,
  trip_distance,
  passenger_count,
  fare_amount as actual_fare,
  ROUND(predict_taxi_fare(
    trip_distance,
    COALESCE(passenger_count, 1),
    HOUR(tpep_pickup_datetime),
    DAYOFWEEK(tpep_pickup_datetime) - 1,  -- Convert to 0-6 range
    CASE WHEN HOUR(tpep_pickup_datetime) BETWEEN 7 AND 9 
              OR HOUR(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END,
    CASE WHEN DAYOFWEEK(tpep_pickup_datetime) IN (1, 7) THEN 1 ELSE 0 END
  ), 2) as predicted_fare,
  predict_tip_category(
    fare_amount,
    trip_distance,
    COALESCE(passenger_count, 1),
    HOUR(tpep_pickup_datetime),
    DAYOFWEEK(tpep_pickup_datetime) - 1,
    CASE WHEN HOUR(tpep_pickup_datetime) BETWEEN 7 AND 9 
              OR HOUR(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END,
    CASE WHEN DAYOFWEEK(tpep_pickup_datetime) IN (1, 7) THEN 1 ELSE 0 END
  ) as predicted_tip_category
FROM samples.nyctaxi.trips
WHERE trip_distance > 0 
  AND fare_amount > 0
  AND tpep_pickup_datetime IS NOT NULL
LIMIT 100;
*/

SELECT 'Uncomment the query above to test with real NYC taxi data' as instructions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 10. Performance Tips

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Tip 1: Cache Predictions for Frequently Queried Data

-- COMMAND ----------

-- Create a materialized view or table with predictions for better performance
-- Uncomment to create:

/*
CREATE OR REPLACE TABLE taxi_predictions_cached AS
SELECT 
  trip_id,
  trip_distance,
  passenger_count,
  hour_of_day,
  day_of_week,
  is_rush_hour,
  is_weekend,
  actual_fare,
  predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ) as predicted_fare,
  predict_tip_category(
    actual_fare, trip_distance, passenger_count, 
    hour_of_day, day_of_week, is_rush_hour, is_weekend
  ) as predicted_tip_category,
  current_timestamp() as prediction_timestamp
FROM sample_taxi_trips;

SELECT * FROM taxi_predictions_cached LIMIT 10;
*/

SELECT 'Uncomment to create cached predictions table' as instructions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Tip 2: Use UDFs in WHERE Clauses Carefully

-- COMMAND ----------

-- GOOD: Filter first, then apply UDF
SELECT 
  trip_id,
  trip_distance,
  predict_taxi_fare(
    trip_distance, passenger_count, hour_of_day, 
    day_of_week, is_rush_hour, is_weekend
  ) as predicted_fare
FROM sample_taxi_trips
WHERE trip_distance > 5.0  -- Filter before UDF
  AND is_rush_hour = 1
LIMIT 10;

-- LESS EFFICIENT: UDF in WHERE clause (UDF runs on all rows before filtering)
-- SELECT * FROM sample_taxi_trips 
-- WHERE predict_taxi_fare(...) > 30;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Tip 3: Use CTEs for Complex Queries

-- COMMAND ----------

-- Use Common Table Expressions (CTEs) to make complex queries more readable
WITH trip_predictions AS (
  SELECT 
    trip_id,
    trip_distance,
    actual_fare,
    predict_taxi_fare(
      trip_distance, passenger_count, hour_of_day, 
      day_of_week, is_rush_hour, is_weekend
    ) as predicted_fare,
    is_rush_hour,
    is_weekend
  FROM sample_taxi_trips
),
prediction_errors AS (
  SELECT 
    *,
    ABS(actual_fare - predicted_fare) as absolute_error,
    (actual_fare - predicted_fare) / actual_fare * 100 as error_percentage
  FROM trip_predictions
)
SELECT 
  CASE is_rush_hour WHEN 1 THEN 'Rush Hour' ELSE 'Normal' END as period,
  COUNT(*) as trips,
  ROUND(AVG(absolute_error), 2) as avg_error,
  ROUND(AVG(error_percentage), 2) as avg_error_pct
FROM prediction_errors
GROUP BY period
ORDER BY period;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 11. Summary

-- COMMAND ----------

SELECT 
  '✓ Fare Prediction UDF' as feature,
  'predict_taxi_fare()' as udf_name,
  'Predicts taxi fare amount' as description
UNION ALL
SELECT 
  '✓ Tip Category UDF',
  'predict_tip_category()',
  'Predicts tip category (Low/Medium/High)'
UNION ALL
SELECT 
  '✓ Sample Data',
  'sample_taxi_trips view',
  '15 test trips with various scenarios'
UNION ALL
SELECT 
  '✓ SQL Examples',
  '40+ queries',
  'Basic usage, aggregations, dashboards'
UNION ALL
SELECT 
  '✓ Performance Tips',
  'Best practices',
  'Caching, filtering, CTEs';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Next Steps
-- MAGIC
-- MAGIC 1. **Customize Queries**: Adapt these examples for your specific use case
-- MAGIC 2. **Create Dashboards**: Use dashboard-ready queries in Databricks SQL dashboards
-- MAGIC 3. **Add More UDFs**: Create additional UDFs following the same pattern
-- MAGIC 4. **Performance Tuning**: Cache predictions for frequently accessed data
-- MAGIC 5. **Integration**: Use these UDFs in your ETL pipelines and reports
-- MAGIC
-- MAGIC ## Documentation Links
-- MAGIC
-- MAGIC - **Python UDFs in Unity Catalog**: https://docs.databricks.com/aws/en/udf/unity-catalog
-- MAGIC - **Databricks SQL Guide**: https://docs.databricks.com/sql/index.html
-- MAGIC - **SQL Functions Reference**: https://docs.databricks.com/sql/language-manual/index.html
-- MAGIC - **Dashboard Creation**: https://docs.databricks.com/dashboards/index.html
-- MAGIC
-- MAGIC ---
-- MAGIC
-- MAGIC **Ready for production!** These SQL queries can be used in:
-- MAGIC - Databricks SQL Warehouses
-- MAGIC - SQL Dashboards
-- MAGIC - Scheduled queries
-- MAGIC - ETL pipelines
-- MAGIC - BI tools via JDBC/ODBC