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

df = pd.concat([df.drop("occupation", axis=1),pd.get_dummies(df.occupation).add_prefix("occupation_")],axis=1)
df = pd.concat([df.drop("workclass", axis=1),pd.get_dummies(df.workclass).add_prefix("workclass_")],axis=1)
df = df.drop("education", axis=1)
df = pd.concat([df.drop("marital-status", axis=1),pd.get_dummies(df["marital-status"]).add_prefix("marital-status_")],axis=1)
df = pd.concat([df.drop("relationship", axis=1),pd.get_dummies(df.relationship).add_prefix("relationship_")],axis=1)
df = pd.concat([df.drop("race",axis=1), pd.get_dummies(df.race).add_prefix("race_")],axis=1)
df = pd.concat([df.drop("native-country",axis=1), pd.get_dummies(df["native-country"]).add_prefix("native-country_")],axis=1)

df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")

sorted_correlations = df.corr()["income"].abs().sort_values()
num_cols_to_drop = int(0.8*len(df.columns))
cols_to_drop = sorted_correlations[:num_cols_to_drop].index
df_dropped = df.drop(cols_to_drop, axis=1)

plt.figure(figsize=(15, 10))
sns.heatmap(df_dropped.corr(), annot=True, cmap="coolwarm")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df= df.drop("fnlwgt", axis=1)

train_df, test_df = train_test_split(df, test_size=0.2)

train_X = train_df.drop("income", axis=1)
train_y = train_df["income"]

test_X = test_df.drop("income", axis=1)
test_y = test_df["income"]

forest = RandomForestClassifier()
forest.fit(train_X, train_y)

print(forest.score(test_X, test_y))
print(forest.feature_importances_)

importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importances = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}

print(importances)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 250],
    "max_depth": [5, 10, 30, None],
    "min_samples_split": [2, 4],
    "max_features": ["sqrt", "log2"]
}
grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                           param_grid=param_grid, verbose=10)
grid_search.fit(train_X, train_y)

forest = grid_search.best_estimator_
print(forest.score(test_X, test_y))
