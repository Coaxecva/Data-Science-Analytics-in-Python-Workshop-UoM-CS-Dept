import pandas
import seaborn
from matplotlib import pyplot
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics

data_file = '/Users/vphan/Dropbox/Python Workshop/December 2017/data/iris.csv'
data = pandas.read_csv(data_file)

# Hypothesis: sepal width and length are good features
# to predict species of the iris flower.
seaborn.lmplot(x='SepalWidth', y='SepalLength', hue='Species', fit_reg=False, data=data)

seaborn.stripplot(x='Species', y='SepalWidth', jitter=True, data=data)
seaborn.boxplot(x='Species', y='SepalWidth', data=data)
seaborn.violinplot(x='Species', y='SepalWidth', data=data)

X = data[['SepalWidth', 'PetalLength']]

# 1. kmeans
km = KMeans(n_clusters=3)
km.fit(X)
# .fit  - model building
# .predict - making prediction
# .score - scoring
# .labels_ - cluster labels of data points

data['kmeans'] = km.labels_

# DBSCAN
dbs = DBSCAN()
dbs.fit(X)
data['dbscan'] = dbs.labels_

# Evaluate prediction *visually*
seaborn.lmplot(x='SepalWidth', y='PetalLength', data=data, fit_reg=False, hue='Species')
seaborn.lmplot(x='SepalWidth', y='PetalLength', data=data, fit_reg=False, hue='kmeans')
seaborn.lmplot(x='SepalWidth', y='PetalLength', data=data, fit_reg=False, hue='dbscan')

# Evaluate prediction analytically
# Use adjusted rand score
print('kmeans:', metrics.adjusted_rand_score( data['Species'], data['kmeans'] ))
print('dbscan:', metrics.adjusted_rand_score( data['Species'], data['dbscan'] ))

X = data[['PetalWidth', 'PetalLength']]
km = KMeans(n_clusters=3)
km.fit(X)
data['kmeans2'] = km.labels_
seaborn.lmplot(x='PetalWidth', y='PetalLength', data=data, fit_reg=False, hue='Species')
seaborn.lmplot(x='PetalWidth', y='PetalLength', data=data, fit_reg=False, hue='kmeans2')


from sklearn.feature_selection import chi2, SelectKBest, SelectFdr
X = data[['SepalWidth', 'SepalLength', 'PetalWidth', 'PetalLength']]
# selector = SelectKBest(chi2, k=2)
selector = SelectFdr(chi2)
selector.fit(X, data['Species'])
X_selected = selector.transform(X)
km.fit(X_selected)
data['km_selected'] = km.labels_

from sklearn.decomposition import PCA
X = data[['SepalWidth', 'SepalLength', 'PetalWidth', 'PetalLength']]
pca = PCA(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)
print(pca.explained_variance)
print(pca.explained_variance_ratio_)

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
