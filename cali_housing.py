import pandas
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

data_file = '../data/cali_housing.csv'
data = pandas.read_csv(data_file)

#------------------------------------------------------------------------
def evaluate(model, X, y):
	print(model)
	scores = cross_val_score(model, X, y, cv=ShuffleSplit(50, train_size=0.9))
	print('R:', scores.mean())

#------------------------------------------------------------------------

data.info()
data.sum()
data.sample(10)

data.median_income.hist()
pyplot.show()

# 1. Deal with missing data
data = data.dropna()

# data.hist(bins=50, figsize=(10,7))
# pyplot.show()

# 2. Discretize some variables (e.g. median income)
a = numpy.ceil( data.median_income / 1.5 )
data['income_cat'] = a.where( a < 5, 5 )

# 3. Deal with categorical variables.
data = pandas.get_dummies(data, columns=['ocean_proximity'])

