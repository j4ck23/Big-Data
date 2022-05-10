import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml import feature
import seaborn as sns
import pyspark.sql.functions as F

#Load data to dataframe
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when, regexp_replace, col, count
from pyspark.sql.types import *
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)

#replaces the string vals of status with bianry digits for the logic tree
dfNum = df.withColumn('Status', 
    when(df.Status.endswith('Normal'),regexp_replace(df.Status,'Normal','1')) \
   .when(df.Status.endswith('Abnormal'),regexp_replace(df.Status,'Abnormal','0')))

#changes the status column to a interger from a string
dfV = dfNum.withColumn("Status",col("Status").cast(IntegerType()))

#assembles every column except Status into a new feaures column.
assembler = VectorAssembler(inputCols=['Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3 ',
                            'Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3',
                            'Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'],
                            outputCol="features")

output = assembler.transform(dfV)

#loads the features form vector assemble and the status column
modelDF = output.select("features", "Status")

#splits the data to train set and test set by a 70% to 30% ratio
train,test = modelDF.randomSplit([0.7, 0.3])

#the neural network, fits the training data. layers = 12 because of the input columns, and out to two because there is only 2 output options.
Network = MultilayerPerceptronClassifier(featuresCol="features", labelCol='Status', layers=[12,24,2], 
                                        maxIter=100,blockSize=8,seed=7,solver='gd').fit(train)

#applys the test data to the trained network
pre = Network.transform(test)

print("Artificail Neural Network:")
#what they values mean
print("Reactor Status Normal = 1, Reactor Status Abnormal = 0")

pre.show()

#accuracy of the model.
acc = pre.filter(pre.Status == pre.prediction).count() / float (test.count())
print("Accuracy of model: ", acc)

#Calculates the Error rate
Error = pre.filter(pre.Status != pre.prediction).count() / float (test.count())
print("Error Rate: ", Error)

print("Anaylsis:")
#count how many rows have correct Normal predictions
Normalcorrect = pre.where((col("prediction")=="1") & (col("Status")==1)).count()
print("Normal Correct: ", Normalcorrect)

#count how many rows have correct Abnormal predictions
Abnormalcorrect = pre.where((col("prediction")=="0") & (col("Status")==0)).count()
print("Abnormal Correct: ", Abnormalcorrect)

#count how many rows have incorrect Normal predictions
Normalincorrect = pre.where((col("prediction")=="1") & (col("Status")==0)).count()
print("Normal Incorrect: ", Normalincorrect)

#count how many rows have incorrect Abnormal predictions
Abnormalincorrect = pre.where((col("prediction")=="0") & (col("Status")==1)).count()
print("Abnormal Incorrect: ", Abnormalincorrect)

#counts and prints the total of correct and incorrect predictions.
totalcorrect = Normalcorrect + Abnormalcorrect
totalincorrect = Normalincorrect + Abnormalincorrect
print("Total correct: ",totalcorrect)
print("Total Incorrect: ",totalincorrect)

#lables the confusion matrix
labels = ['True Abnormal','False Normal','False Abnormal','True Normal']

#gathers the data calculated above
values = [Abnormalcorrect,Normalincorrect,Abnormalincorrect,Normalcorrect]

#shapes the matrix with the labels and data
labels = np.asarray(labels).reshape(2,2)
values = np.asarray(values).reshape(2,2)
confusion = ([[Abnormalcorrect,Normalincorrect],[Abnormalincorrect,Normalcorrect]])

#prints the matrix
print("Confusion Matrix:")
print(labels)
print(values)

#plots a heat map for the matrix
map =sns.heatmap(confusion,annot=True,fmt='g')
plt.title('Artifical Neural Network, Normal = 1, Abnormal = 0', fontsize = 10) # title with fontsize 10
plt.show()