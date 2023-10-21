import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def loadSalaryData():
    salary = pd.read_csv('Salary_Data.csv')
    return salary

salary = loadSalaryData()

# print(str(len(salary)))
# incomplete_rows = salary[salary.isnull().any(axis=1)].head()
# print(incomplete_rows)