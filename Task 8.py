from typing import Counter
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
import pyspark
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from functools import reduce
from pyspark.sql.functions import when, regexp_replace, col, count
from pyspark.sql.types import *


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


data_set = pd.read_csv("nuclear_plants_big_dataset.csv", encoding='latin1', skiprows=0) #reads the data to a pandas dataframe.
data = data_set['Vibration_sensor_4']#selectst the column you want to analysis.
data.drop(index=data.index[0], axis=0, inplace=True) #drops the name of the column
data = data.astype(np.double)#changes the columns data type
#print(data)

n = 100000#sets the size of the chunks
Clusters = [data[i:i+n] for i in range(0,data.shape[0],n)]#splits the data into chunks depending on the set size,

Max = list(map(max, Clusters))#finds the max for each cluster and returns to a list
Min = list(map(min, Clusters))#finds the min for each cluster and returns to a list
Mean = list(map(mean, Clusters))#finds the mean for each cluster and returns to a list
print(Max)
print(Min)
print(Mean)

results_Max = reduce(max, Max)#finds the max value form the list of max values of the clusters
results_Min = reduce(min, Min)#finds the min value form the list of min values of the clusters
results_Mean = reduce(lambda a,b: a+b, Mean) / len(Mean)#finds the mean value form the list of mean values of the clusters

print("Max Value:")
print(results_Max)

print("Min Value:")
print(results_Min)

print("Mean Value:")
print(results_Mean)