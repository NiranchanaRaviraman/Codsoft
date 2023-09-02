import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


titanic_predict=pd.read_csv("tested.csv")
numerical_columns = titanic_predict.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numerical_columns.corr(), cmap="YlGnBu")
plt.show()

# print(titanic_predict)
# 


