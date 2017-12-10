import pandas
from matplotlib import pyplot
data_file = '../data/cali_housing.csv'
data = pandas.read_csv(data_file)

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

