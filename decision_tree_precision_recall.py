import pandas
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lcplot
from matplotlib import pyplot

#------------------------------------------------------
def validate(model, X, y):
	shuffle100 = ShuffleSplit(n_splits=100, test_size=0.1)
	scores = cross_validate(model, X, y, cv=shuffle100, scoring=['precision_weighted', 'recall_weighted', 'accuracy'])
	print('Precision:', round(scores['test_precision_weighted'].mean(),2))
	print('Recall:   ', round(scores['test_recall_weighted'].mean(),2))
	print('Accuracy: ', round(scores['test_accuracy'].mean(),2))

#------------------------------------------------------

data = pandas.read_csv('../data/iris.csv')

# y : Species
# X : everything else
# model : decision tree
y = data['Species']
X = data.drop('Species', axis=1)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Build a decision model using training data.

# Find out good training size
# lcplot.plot(DecisionTreeClassifier(), X, y)
# we'll use 0.1 for test_size

# ----------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Test your model using the testing data.
scores = dt.score(X_test, y_test)

# Validate
validate(dt, X,  y)

# Make sense out of the decision model
dt = DecisionTreeClassifier()
dt.fit(X,y)
dt.predict([ [5.5,3.6,1.3,0.25] , [5.55,3.33,4.0,1.5], [6.34,3.1,5.6,1.9]  ])

# Visualize tree

draw_tree.visualize_tree(dt, X.columns, 'output')
# Use this command for Windows cmd
# dot -Tpng -o dt.png output.dot 
