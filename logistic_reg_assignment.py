import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('logisticX.csv')
dataset2 = pd.read_csv('logisticY.csv')

X1 = dataset.iloc[:,0].values
X2 = dataset.iloc[:,1].values
Y =  dataset2.iloc[:,0].values
for i in range(99):
    if Y[i]==1:
        plt.scatter(X1[i],X2[i],marker='*',color='Blue')
    else:
        plt.scatter(X1[i],X2[i],marker='^',color='Orange')
plt.show()

