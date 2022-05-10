import pyspark
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

#Load data to dataframe
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)
df.show(10)

#Task 1
from pyspark.sql.functions import when, count, col
df_missing = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]) #Finds any missing vlaues and counts them
df_missing.show() # displays a total for missing values