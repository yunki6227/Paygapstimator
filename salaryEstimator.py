import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import pickle

np.random.seed(42)

def loadSalaryData():
    salary = pd.read_csv('Salary_Data.csv')
    return salary

salary = loadSalaryData()

#drops rows with missing values
salary = salary.dropna()
salary = salary[salary['Gender'] != 'Other']

# Reset the index
salary = salary.reset_index(drop=True)

# Find the top 10 most common job titles
top_10_job_titles = salary['Job Title'].value_counts().head(10).index

# Create a boolean mask for rows with job titles not in the top 10
mask = salary['Job Title'].isin(top_10_job_titles)

# Filter and drop rows that don't meet the condition
salary = salary[mask]

X = salary.drop(columns=['Salary'])
y = salary['Salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

salary_num = X_train.select_dtypes(include=[np.number])
salary_cat = X_train.select_dtypes(exclude=[np.number])

num_attribs = list(salary_num)
cat_attribs = list(salary_cat)

num_pipeline = Pipeline([
    ('std_scalar',StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(handle_unknown='ignore'),cat_attribs)
])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators = 100, random_state=42)
forest_reg.fit(X_train_prepared,y_train)

# Perform 5-fold cross-validation
scores = cross_val_score(forest_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)

# Calculate the root mean squared error (RMSE) for the cross-validation scores
rmse_scores = np.sqrt(-scores)

# Print the RMSE scores for each fold
# for fold, rmse in enumerate(rmse_scores, 1):
#     print(f"Fold {fold}: RMSE = {rmse}")

# Calculate the mean and standard deviation of RMSE scores
mean_rmse = rmse_scores.mean()
std_rmse = rmse_scores.std()

# print(f"Mean RMSE: {mean_rmse}")
# print(f"Standard Deviation of RMSE: {std_rmse}")
#print(X_train.iloc[0])

pickle.dump(forest_reg, open('model.pkl','wb'))