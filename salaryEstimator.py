import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns

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

X_all = pd.concat([X_train, X_test])
X_all['Salary'] = pd.concat([y_train, y_test])
# corr_matrix = X_all.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

for cat_column in cat_attribs:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cat_column, y='Salary', data=X_all, ci=None)
    plt.title(f'Correlation between {cat_column} and Salary')
    plt.xlabel(cat_column)
    plt.ylabel('Salary')
    plt.show()

