#Income Prediction
import pandas as pd
import matplotlib.pyplot as plt
#read csv file
df= pd.read_csv('C:\\Users\\marce\\Desktop\\adult.csv')

print(df)
print(df["education"].value_counts())
print(df["occupation"].value_counts())
print(df["relationship"].value_counts())

#prefix of ? for one hot encoding

print(df.gender.value_counts())
