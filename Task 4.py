import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load data to dataframe
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)
df.show(1)

#splits the data randomly 70%, 30%
train,test = df.randomSplit([0.7,0.3])

#Counts the amount of data for both groups of data
print("training data")
train_amount = train.groupBy("Status").count().show()

#Counts the amount of data for both groups of data.
print("testing data")
test_amount = test.groupBy("Status").count().show()
