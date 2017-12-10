import pandas
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

data = pandas.read_csv("../data/admission.csv")

y = admission['admit']
X = admission.drop('admit', axis=1)
