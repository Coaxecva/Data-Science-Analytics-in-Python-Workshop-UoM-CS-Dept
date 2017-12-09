import numpy
from matplotlib import pyplot
from sklearn.model_selection import learning_curve

def plot(estimator, X, y, title="", ylim=None, cv=None, n_jobs=1, train_sizes=numpy.linspace(.1, 0.95, 10)):
	# print(train_sizes)
	pyplot.figure()
	pyplot.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
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
