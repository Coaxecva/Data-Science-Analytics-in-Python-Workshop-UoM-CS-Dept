import pandas
import seaborn
import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, cross_val_score, learning_curve

#----------------------------------------------------------
def plot(estimator, X, y, title="", ylim=None, cv=None, n_jobs=1, train_sizes=numpy.linspace(.1, 0.95, 10)):
	print(train_sizes)
	pyplot.figure()
	pyplot.title(title)
	if ylim is not None:
		pyplot.ylim(*ylim)
	pyplot.xlabel("Training examples")
	pyplot.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = numpy.mean(train_scores, axis=1)
	train_scores_std = numpy.std(train_scores, axis=1)
	test_scores_mean = numpy.mean(test_scores, axis=1)
	test_scores_std = numpy.std(test_scores, axis=1)
	pyplot.grid()

	pyplot.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
	pyplot.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
	pyplot.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
	pyplot.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
	pyplot.legend(loc="best")
	return pyplot

#-------------------------------------------------
def my_predict(model, x):
	return x * model.coef_ + model.intercept_
#-------------------------------------------------

data_file = '/Users/vphan/Dropbox/Python Workshop/December 2017/data/iris.csv'
data = pandas.read_csv(data_file)

# predict petal width using petal length
y = data['PetalWidth']
X = data[['PetalLength']]

# build and predict
model = LinearRegression()
model.fit(X,y)
print(model.predict([ [1.5], [1.3], [2.0] ]))

# evaluate model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
model.fit(X_train, y_train)
print('Score:', model.score(X_test, y_test))

# cross validate
# scores = cross_val_score(model, X, y, cv=ShuffleSplit(n_splits=20, train_size=0.9))

scores = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True))
print('Scores:', scores)
print('Average score:', scores.mean())


# learning curve
# train_sizes, train_scores, valid_scores = learning_curve(
# 	model,
# 	X,
# 	y,
# 	train_sizes=numpy.linspace(.1, 1.0, 10),
# 	cv=KFold(n_splits=10, shuffle=True)
# )

plot(model, X, y, cv=KFold(n_splits=10, shuffle=True))
pyplot.show()
