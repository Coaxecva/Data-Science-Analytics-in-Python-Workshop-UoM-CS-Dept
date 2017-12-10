import pandas, numpy
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

#------------------------------------------------------------------------
def evaluate(model, X, y):
	print(model)
	scores = cross_val_score(model, X, y, cv=ShuffleSplit(100, train_size=0.9))
	print('R:', round(scores.mean(), 2))

#------------------------------------------------------------------------

data_file = '../data/cali_housing.csv'
data = pandas.read_csv(data_file)

# 1. Deal with missing data
data = data.dropna()

# 2. Discretize some variables (e.g. median income)
a = numpy.ceil( data.median_income / 1.5 )
data['income_cat'] = a.where( a < 5, 5 )

# 3. Deal with categorical variables.
data = pandas.get_dummies(data, columns=['ocean_proximity'])

df = data.drop('median_income', axis=1)

# 4. Get features and y
y = df.median_house_value
X = df.drop('median_house_value', axis=1)


from sklearn.externals import joblib
# Learn
model = GradientBoostingRegressor()
model.fit(X,y)
# Save
joblib.dump(model, 'CAhouses_gbr.model')
# Load
model = joblib.load('CAhouses_gbr.model')
# Predict
model.predict([subdiv1, subdiv2, subdiv3, subdiv4])
