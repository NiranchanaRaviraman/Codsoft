import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


titanic_predict = pd.read_csv("tested.csv")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_predict, titanic_predict["Survived"]):
    strat_train_set = titanic_predict.loc[train_indices]
    strat_test_set = titanic_predict.loc[test_indices]


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
strat_train_set["Survived"].hist()
strat_train_set["Pclass"].hist()


plt.subplot(1, 2, 2)
strat_test_set["Survived"].hist()
strat_test_set["Pclass"].hist()

strat_train_set.info()




plt.tight_layout()
plt.show()


