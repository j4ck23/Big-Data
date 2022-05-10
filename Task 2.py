import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import column, udf
import seaborn as sns

#Load data to dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, asc, desc
import pyspark.sql.functions as func 
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)
#df.show(1)

#Task 2
df_norm = df.where(df.Status == "Normal")#Selects data where the status is normal
#df_norm.show(5)

df_abnorm = df.where(df.Status =="Abnormal")#Selects data where the status is abnormal
#df_abnorm.show(5)

seq_of_columns = df.columns #fetches the headers of the columns

#max for each column in normal group
print("Max value for the normal data group")
df_norm_max = df_norm.select(seq_of_columns).describe().filter("summary = 'max'")
df_norm_max.show()

#max for each column in abnormal group
print("Max value for the abnormal data group")
df_abnorm_max = df_abnorm.select(seq_of_columns).describe().filter("summary = 'max'")
df_abnorm_max.show()

#min for each column in normal group
print("Min value for the normal data group")
df_norm_min = df_norm.select(seq_of_columns).describe().filter("summary = 'min'")
df_norm_min.show()

#min for each column in abnormal group
print("Min value for the abnormal data group")
df_abnorm_min = df_abnorm.select(seq_of_columns).describe().filter("summary = 'min'")
df_abnorm_min.show()

#mean for each column in normal group
print("Mean value for the normal data group")
df_norm_mean = df_norm.select(seq_of_columns).describe().filter("summary = 'mean'")
df_norm_mean.show()

#mean for each column in abnormal group
print("Mean value for the abnormal data group")
df_abnorm_mean = df_abnorm.select(seq_of_columns).describe().filter("summary = 'mean'")
df_abnorm_mean.show()

#mode for each column in normal group
print("Mode value for the normal data group")
#counts the values in the column and stores them in order from most to least then shows only the first data.
df_norm_mode = df_norm.groupBy('Power_range_sensor_1').count().sort(col("count").desc()).show(1)
df_norm_mode1 = df_norm.groupBy('Power_range_sensor_2').count().sort(col("count").desc()).show(1)
df_norm_mode2 = df_norm.groupBy('Power_range_sensor_3 ').count().sort(col("count").desc()).show(1)
df_norm_mode3 = df_norm.groupBy('Power_range_sensor_4').count().sort(col("count").desc()).show(1)
df_norm_mode4 = df_norm.groupBy('Pressure _sensor_1').count().sort(col("count").desc()).show(1)
df_norm_mode5 = df_norm.groupBy('Pressure _sensor_2').count().sort(col("count").desc()).show(1)
df_norm_mode6 = df_norm.groupBy('Pressure _sensor_3').count().sort(col("count").desc()).show(1)
df_norm_mode7 = df_norm.groupBy('Pressure _sensor_4').count().sort(col("count").desc()).show(1)
df_norm_mode8 = df_norm.groupBy('Vibration_sensor_1').count().sort(col("count").desc()).show(1)
df_norm_mode9 = df_norm.groupBy('Vibration_sensor_2').count().sort(col("count").desc()).show(1)
df_norm_mode10 = df_norm.groupBy('Vibration_sensor_3').count().sort(col("count").desc()).show(1)
df_norm_mode11 = df_norm.groupBy('Vibration_sensor_4').count().sort(col("count").desc()).show(1)


#mode for each column in abnormal group
print("Mode value for the abnormal data group")
#counts the values in the column and stores them in order from most to least then shows only the first data.
df_abnorm_mode = df_abnorm.groupBy('Power_range_sensor_1').count().sort(col("count").desc()).show(1)
df_abnorm_mode1 = df_abnorm.groupBy('Power_range_sensor_2').count().sort(col("count").desc()).show(1)
df_abnorm_mode2 = df_abnorm.groupBy('Power_range_sensor_3 ').count().sort(col("count").desc()).show(1)
df_abnorm_mode3 = df_abnorm.groupBy('Power_range_sensor_4').count().sort(col("count").desc()).show(1)
df_abnorm_mode4 = df_abnorm.groupBy('Pressure _sensor_1').count().sort(col("count").desc()).show(1)
df_abnorm_mode5 = df_abnorm.groupBy('Pressure _sensor_2').count().sort(col("count").desc()).show(1)
df_abnorm_mode6 = df_abnorm.groupBy('Pressure _sensor_3').count().sort(col("count").desc()).show(1)
df_abnorm_mode7 = df_abnorm.groupBy('Pressure _sensor_4').count().sort(col("count").desc()).show(1)
df_abnorm_mode8 = df_abnorm.groupBy('Vibration_sensor_1').count().sort(col("count").desc()).show(1)
df_abnorm_mode9 = df_abnorm.groupBy('Vibration_sensor_2').count().sort(col("count").desc()).show(1)
df_abnorm_mode10 = df_abnorm.groupBy('Vibration_sensor_3').count().sort(col("count").desc()).show(1)
df_abnorm_mode11 = df_abnorm.groupBy('Vibration_sensor_4').count().sort(col("count").desc()).show(1)


#Median for each column in normal group
print("Median value for the normal data group")
df_norm_median = df_norm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_1", 0.5).alias("median")).show(1)
df_norm_median1 = df_norm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_2", 0.5).alias("median")).show(1)
df_norm_median2 = df_norm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_3 ", 0.5).alias("median")).show(1)
df_norm_median3 = df_norm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_4", 0.5).alias("median")).show(1)
df_norm_median4 = df_norm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_1", 0.5).alias("median")).show(1)
df_norm_median5 = df_norm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_2", 0.5).alias("median")).show(1)
df_norm_median6 = df_norm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_3", 0.5).alias("median")).show(1)
df_norm_median7 = df_norm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_4", 0.5).alias("median")).show(1)
df_norm_median8 = df_norm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_1", 0.5).alias("median")).show(1)
df_norm_median9 = df_norm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_2", 0.5).alias("median")).show(1)
df_norm_median10 = df_norm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_3", 0.5).alias("median")).show(1)
df_norm_median11 = df_norm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_4", 0.5).alias("median")).show(1)

#Median for each column in abnormal group
print("Median value for the abnormal data group")
df_abnorm_median = df_abnorm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_1", 0.5).alias("median")).show(1)
df_abnorm_median1 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_2", 0.5).alias("median")).show(1)
df_abnorm_median2 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_3 ", 0.5).alias("median")).show(1)
df_abnorm_median3 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Power_range_sensor_4", 0.5).alias("median")).show(1)
df_abnorm_median4 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_1", 0.5).alias("median")).show(1)
df_abnorm_median5 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_2", 0.5).alias("median")).show(1)
df_abnorm_median6 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_3", 0.5).alias("median")).show(1)
df_abnorm_median7 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Pressure _sensor_4", 0.5).alias("median")).show(1)
df_abnorm_median8 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_1", 0.5).alias("median")).show(1)
df_abnorm_median9 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_2", 0.5).alias("median")).show(1)
df_abnorm_median10 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_3", 0.5).alias("median")).show(1)
df_abnorm_median11 = df_abnorm.groupBy("Status").agg(func.percentile_approx("Vibration_sensor_4", 0.5).alias("median")).show(1)

#variacne for each column in normal group
print("Variance value for the normal data group")
df_norm_variacne = df_norm.agg({'Power_range_sensor_1': 'variance'}).show()
df_norm_variacne1 = df_norm.agg({'Power_range_sensor_2': 'variance'}).show()
df_norm_variacne2 = df_norm.agg({'Power_range_sensor_3 ': 'variance'}).show()
df_norm_variacne3 = df_norm.agg({'Power_range_sensor_4': 'variance'}).show()
df_norm_variacne4 = df_norm.agg({'Pressure _sensor_1': 'variance'}).show()
df_norm_variacne5 = df_norm.agg({'Pressure _sensor_2': 'variance'}).show()
df_norm_variacne6 = df_norm.agg({'Pressure _sensor_3': 'variance'}).show()
df_norm_variacne7 = df_norm.agg({'Pressure _sensor_4': 'variance'}).show()
df_norm_variacne8 = df_norm.agg({'Vibration_sensor_1': 'variance'}).show()
df_norm_variacne9 = df_norm.agg({'Vibration_sensor_2': 'variance'}).show()
df_norm_variacne10 = df_norm.agg({'Vibration_sensor_3': 'variance'}).show()
df_norm_variacne11 = df_norm.agg({'Vibration_sensor_4': 'variance'}).show()

#variance for each column in abnormal group
print("Variance value for the abnormal data group")
df_abnorm_variacne = df_abnorm.agg({'Power_range_sensor_1': 'variance'}).show()
df_abnorm_variacne1 = df_abnorm.agg({'Power_range_sensor_2': 'variance'}).show()
df_abnorm_variacne2 = df_abnorm.agg({'Power_range_sensor_3 ': 'variance'}).show()
df_abnorm_variacne3 = df_abnorm.agg({'Power_range_sensor_4': 'variance'}).show()
df_abnorm_variacne4 = df_abnorm.agg({'Pressure _sensor_1': 'variance'}).show()
df_abnorm_variacne5 = df_abnorm.agg({'Pressure _sensor_2': 'variance'}).show()
df_abnorm_variacne6 = df_abnorm.agg({'Pressure _sensor_3': 'variance'}).show()
df_abnorm_variacne7 = df_abnorm.agg({'Pressure _sensor_4': 'variance'}).show()
df_abnorm_variacne8 = df_abnorm.agg({'Vibration_sensor_1': 'variance'}).show()
df_abnorm_variacne9 = df_abnorm.agg({'Vibration_sensor_2': 'variance'}).show()
df_abnorm_variacne10 = df_abnorm.agg({'Vibration_sensor_3': 'variance'}).show()
df_abnorm_variacne11 = df_abnorm.agg({'Vibration_sensor_4': 'variance'}).show()

#box plot for the normal group of data
dfNormBox = df_norm.toPandas()
boxplotNorm = dfNormBox.boxplot(column = ['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4',
                                          'Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4', 
                                          'Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'])
plt.title("Normal data")                                          
plt.show()

#box plot for the abnormal group of data
dfAbnormBox = df_abnorm.toPandas()
boxplotNorm = dfAbnormBox.boxplot(column = ['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3 ', 'Power_range_sensor_4',
                                          'Pressure _sensor_1', 'Pressure _sensor_2', 'Pressure _sensor_3', 'Pressure _sensor_4', 
                                          'Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4'])
plt.title("Abnormal data")                                          
plt.show()