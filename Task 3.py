import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load data to dataframe
from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)
df.show(1)

#Task 3

#drops stats as it is a string
df1 = df.drop('Status')

#converts the spark dataframe to a panadas data frame
test = df1.toPandas()

#correlates the data
test1 = test.corr()
print(test1)

#plot the correlation in a heat map
plt.matshow(test.corr())
plt.show()