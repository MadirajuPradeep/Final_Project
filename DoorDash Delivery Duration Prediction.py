# Databricks notebook source
# MAGIC %md
# MAGIC ##Linear Regression to Estimate Delivery Duration for DoorDash
# MAGIC 
# MAGIC The dataset has about 197,429 rows and 16 features and contains a subset of deliveries received at DoorDash in early 2015 in a subset of the cities. Each row in this file corresponds to one unique delivery. The target value to predict here is the total seconds/minutes taken for delivery from when the order was created at and the delivery time.

# COMMAND ----------

# MAGIC %md ##### Create a spark session and load the DoorDash Data set

# COMMAND ----------

sc

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

file_location = "/FileStore/tables/historical_data.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data pre-processing

# COMMAND ----------

# Import the required libraries
from pyspark.sql.types import IntegerType,DoubleType
from pyspark.sql.functions import *

# COMMAND ----------

# Create new timestamp column for all the attributes that had timestamp details stored as string
# Convert the string values to int
# Create the duration column (duration between order created time and actual delivery time in minutes and seconds)

df=df.withColumn('actual_delivery_time_ts',to_timestamp(df.actual_delivery_time)).\
                withColumn('market_id_int',df["market_id"].cast(IntegerType())).\
                withColumn('order_protocol_int',df["order_protocol"].cast(IntegerType())).\
                withColumn('total_onshift_dashers_int',df["total_onshift_dashers"].cast(IntegerType())).\
                withColumn('total_busy_dashers_int',df["total_busy_dashers"].cast(IntegerType())).\
                withColumn('total_outstanding_orders_int',df["total_outstanding_orders"].cast(IntegerType())).\
                withColumn('estimated_store_to_consumer_driving_duration_int',df["estimated_store_to_consumer_driving_duration"].cast(IntegerType())).\
                withColumn('estimated_order_place_minutes',round(col('estimated_order_place_duration')/60)).\
                withColumn('estimated_driving_duration_minutes',round(col('estimated_store_to_consumer_driving_duration_int')/60)).\
                withColumn('DurationInSeconds',col("actual_delivery_time_ts").cast("long")- col("created_at").cast("long")).\
                withColumn('DurationInMinutes',round(col('DurationInSeconds')/60))
                

# COMMAND ----------

#Dropping old columns

df=df.drop("market_id","actual_delivery_time","order_protocol","total_onshift_dashers","total_busy_dashers","total_outstanding_orders","estimated_store_to_consumer_driving_duration")
display(df)

# COMMAND ----------

#create a new column to determine the time of the day the order was created at
df=df.withColumn('createdtime', date_format('created_at', 'HH:mm:ss'))

def timer(t):
    if t>= '6:00:00' and t<'12:00:00':
        return 'Morning'
    elif t>= '12:00:00' and t<'17:00:00':
        return 'Noon'
    elif t>= '17:00:00' and t<='23:59:59':
        return 'Evening'
    else:
        return 'Night'
    
func_udf = udf(timer)
df=df.withColumn("TimeOfDay",func_udf(df['createdtime']))

display(df)

# COMMAND ----------

#checking count of rows
df.count()

# COMMAND ----------

#dropping NA values
df=df.dropna()
df.count()

# COMMAND ----------

#create temporary table from the dataframe

temp_table_name = "temp" 
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

#create permanent table from the dataframe to access it from all sessions

permanent_table_name = "pt"
df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check if table is created
# MAGIC 
# MAGIC select * from temp limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check for outliers
# MAGIC select DurationInMinutes from temp order by 1 desc limit 10

# COMMAND ----------

#removing outliers

df = df.filter(df.DurationInMinutes!='6231')
df = df.filter(df.DurationInMinutes!='5541')
df = df.filter(df.DurationInMinutes!='951')
df = df.filter(df.DurationInMinutes!='907')
df = df.filter(df.DurationInMinutes!='803')
df = df.filter(df.DurationInMinutes!='761')
df = df.filter(df.DurationInMinutes!='656')
df = df.filter(df.DurationInMinutes!='603')
df = df.filter(df.DurationInMinutes!='549')
df = df.filter(df.DurationInMinutes!='545')
df = df.filter(df.DurationInMinutes!='536')
df = df.filter(df.DurationInMinutes!='472')
df = df.filter(df.DurationInMinutes!='470')
df = df.filter(df.DurationInMinutes!='451')
df = df.filter(df.DurationInMinutes!='427')
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Exploratory Data Analysis

# COMMAND ----------

#Summary statistics

df.select("DurationInMinutes","estimated_order_place_minutes","estimated_driving_duration_minutes").summary("mean","min","25%","50%","75%","max","stddev").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Avg Duration in Minutes by Market
# MAGIC 
# MAGIC select market_id_int as Market_ID,round(avg(DurationInMinutes),1) as Avg_Duration_In_Minutes from temp group by 1 order by 2 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Avg Duration in Minutes by Order Protocol
# MAGIC 
# MAGIC select order_protocol_int as Order_Protocol,round(avg(DurationInMinutes),1) as Avg_Duration_In_Minutes from temp group by 1 order by 2 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Number of Orders and Avg Duration in Minutes by Time of the Day
# MAGIC 
# MAGIC select TimeOfDay,count(*) as Total_Number_of_Orders,round(avg(DurationInMinutes),1) as Avg_Duration_In_Minutes from temp group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC ###Building the Linear Regression Model

# COMMAND ----------

# Import the required libraries

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler,StringIndexer,StandardScaler
from pyspark.ml import Pipeline

# COMMAND ----------

# Selecting the dependent and the independent variables that are identified as most useful attributes to estimate duration

data=df.select(['store_id','store_primary_category','subtotal','num_distinct_items','market_id_int','estimated_order_place_duration','total_busy_dashers_int',
                                 'order_protocol_int','total_onshift_dashers_int','estimated_store_to_consumer_driving_duration_int','TimeOfDay','DurationInMinutes'])

# COMMAND ----------

# Create a 70-30 train test split

train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

data.count()

# COMMAND ----------

# Use StringIndexer to convert the categorical columns to hold numerical data


store_category_indexer = StringIndexer(inputCol='store_primary_category',outputCol='store_category_index',handleInvalid='keep')
timeofday_indexer = StringIndexer(inputCol='TimeOfDay',outputCol='timeofday_index',handleInvalid='keep')



# COMMAND ----------

# Vector assembler is used to create a vector of input features

features = ["store_id","store_category_index","timeofday_index","num_distinct_items","market_id_int","subtotal","estimated_order_place_duration","total_onshift_dashers_int","estimated_store_to_consumer_driving_duration_int","order_protocol_int"]

assembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")

# COMMAND ----------

# StandardScaler is used to resize the distribution of values
standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")

# COMMAND ----------

# Pipeline is used to pass the data through indexer,assembler and StandardScalar simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data
pipe = Pipeline(stages=[store_category_indexer,timeofday_indexer, assembler, standardScaler])

# COMMAND ----------

fitted_pipe=pipe.fit(train_data)

# COMMAND ----------

train_data=fitted_pipe.transform(train_data)
display(train_data)


# COMMAND ----------

# Create an object for the Linear Regression model

lr_model = LinearRegression(labelCol='DurationInMinutes')

# COMMAND ----------

# Fit the model on the train data

fit_model = lr_model.fit(train_data.select(['features','DurationInMinutes']))

# COMMAND ----------

# Transform the test data using the model to predict the duration

test_data=fitted_pipe.transform(test_data)
display(test_data)

# COMMAND ----------

# Store the results in a dataframe

results = fit_model.transform(test_data)
display(results)

# COMMAND ----------

results.select(['DurationInMinutes','prediction']).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Evaluating the model

# COMMAND ----------

test_results = fit_model.evaluate(test_data)

# COMMAND ----------

test_results.residuals.show()

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

test_results.r2

# COMMAND ----------

# MAGIC %md
# MAGIC ######R2 value of 12.0% indicates that the model explains only about 12.0% variance in the Delivery Duration

# COMMAND ----------

 # Finding Beta Coefficients
fit_model.coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC ####Interpretation

# COMMAND ----------


As the number of distinct items increases by 1 count, the delivery duration increases by 0.5983 minutes
As the subtotal amount increases by 1 dollar, the delivery duration increases by 2.9516 minutes
As the estimated order place duration increases by 1 minute, the delivery duration increases by 2.0348 minutes
As the total onshift dashers increases by 1 person, the delivery duration increases by 0.3298 minutes
As the estimated store to consumer driving duration increases by 1 minute, the delivery duration increases by 4.2695 minutes.

The independent factors chosen for the model explain about 12% variance in the Delivery Duration. Although the model explains only about 12.0% variance, DoorDash can use this model to reasonably predict the delivery duration based on the above factors.

The average delivery duration is around 47 minutes.
The number of orders placed and average duration is highest for orders placed at night, compared to noon and evening. Hence, we can have more on-shift dashers during night to provide better customer service.
The average duration is highest for orders placed through protocol 6. Hence, we can have work on order protocol 6 to reduce average delivery duration.
The average duration is very higher for market ID 1. Hence, we can have more on-shift dashers in this region to provide better customer service.


