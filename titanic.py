import pandas
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import lcplot
from matplotlib import pyplot
import draw_tree
from sklearn import metrics, svm

#------------------------------------------------------
def validate(model, X, y):
	print(model)
	shuffle100 = ShuffleSplit(n_splits=100, test_size=0.1)
	scores = cross_validate(model, X, y, cv=shuffle100, scoring=['precision_weighted', 'recall_weighted', 'accuracy'])
	print('Precision:', round(scores['test_precision_weighted'].mean(),2))
	print('Recall:   ', round(scores['test_recall_weighted'].mean(),2))
	print('Accuracy: ', round(scores['test_accuracy'].mean(),2))
	print('-'*60)

#------------------------------------------------------

data_file = '../data/titanic.csv'
data = pandas.read_csv(data_file)
df = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'survived']]
df = df.dropna()
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch']]
X = pandas.get_dummies(X, columns=['sex'])
y = df['survived']

models = [
	DecisionTreeClassifier(max_depth=4),
	LogisticRegression(),
	svm.SVC(),
	GaussianNB(),
	ExtraTreesClassifier(n_estimators=50),
	RandomForestClassifier(n_estimators=50),
	GradientBoostingClassifier(),
]

for m in models:
	validate(m, X, y)
