import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('C:\\Users\\mahapgo\\machinelearning4beginner\\linearRegression\\homeprices.csv')

new_df = df.drop('price',axis='columns')
print(new_df)

price = df.price
print(price)

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)

area_df = pd.read_csv('C:\\Users\\mahapgo\\machinelearning4beginner\\linearRegression\\areas.csv')
p = reg.predict(area_df)
print(p)

area_df['prices']=p
print(area_df)
