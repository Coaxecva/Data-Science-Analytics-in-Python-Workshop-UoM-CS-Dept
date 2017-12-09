import pandas
import seaborn
from matplotlib import pyplot

data_file = '/Users/vphan/Dropbox/Python Workshop/December 2017/data/iris.csv'
data = pandas.read_csv(data_file)

# Hypothesis: sepal width and length are good features
# to predict species of the iris flower.
seaborn.lmplot(x='SepalWidth', y='SepalLength', hue='Species', fit_reg=False, data=data)

seaborn.stripplot(x='Species', y='SepalWidth', jitter=True, data=data)
seaborn.boxplot(x='Species', y='SepalWidth', data=data)
seaborn.violinplot(x='Species', y='SepalWidth', data=data)

pyplot.show()


'''
features - X
target variable - y

1. y is unknown.
Group/cluster data based on X.  Clustering.

2. y is known.
Build model based on known data.  Use this model to predict
new data.
(i) y is continuous.  Regression.
(ii) y is discrete.  Classification.

'''
