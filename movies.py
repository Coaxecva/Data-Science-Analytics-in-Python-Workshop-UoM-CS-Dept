import pandas
from matplotlib import pyplot

u = pandas.read_csv('../data/movielens/users.csv')
m = pandas.read_csv('../data/movielens/movies.csv')
r = pandas.read_csv('../data/movielens/ratings.csv')

ur = pandas.merge(u,r)
data = pandas.merge(m, ur)

# What are the age groups?
data['age'].value_counts()

# How many F and M entries are there?
data['gender'].value_counts()

# Show histogram of 50 random samples
data['age'].sample(50).plot.hist()
# pyplot.show()
