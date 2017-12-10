import pandas
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from skelearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#------------------------------------------------------
def validate(model, X, y):
	shuffle100 = ShuffleSplit(n_splits=100, test_size=0.1)
	scores = cross_validate(model, X, y, cv=shuffle100, scoring=['precision_weighted', 'recall_weighted', 'accuracy'])
	print('Precision:', round(scores['test_precision_weighted'].mean(),2))
	print('Recall:   ', round(scores['test_recall_weighted'].mean(),2))
	print('Accuracy: ', round(scores['test_accuracy'].mean(),2))
#------------------------------------------------------

data = pandas.read_csv("../data/admission.csv")

y = admission['admit']
X = admission[['gre','gpa','rank']]

models = {
  DecisionTreeClassifier(),
  LogisticRegression(),
  svm.SVC(),
  GaussianNB(),
  RandomForestClassifier(n_estimators=50),
  GradientBoostingClassifier(),
}

for m in models:
  validate(m, X, y)
