import pandas
from matplotlib import pyplot

u = pandas.read_csv('../data/movielens/users.csv')
m = pandas.read_csv('../data/movielens/movies.csv')
r = pandas.read_csv('../data/movielens/ratings.csv')

data = pandas.merge(m, pandas.merge(u,r))

# What are the age groups?
data['age'].value_counts()

# How many F and M entries are there?
data['gender'].value_counts()

# Show histogram of 50 random samples
data['age'].sample(50).plot.hist()
# pyplot.show()

# Select entries from the 1-34 age group.
df = data[ data.age <= 18 ]

# Select entries reviewers under 18 and above 50
df = data[ (data.age <= 18) | (data.age >= 50) ]

# What are the most popular movies?  GroupBy, Aggregate
by_titles = data.groupby('title')
by_titles_count = by_titles.count()
sorted = by_titles_count.sort_values(by='rating', ascending=False)

# What are the most popular movies?
# We need to *group* same movies into groups and *count* them
by_titles = data.groupby('title')
by_titles_count = by_titles[['rating']].count().sort_values(by='rating', ascending=False)

by_titles_mean = by_titles.mean()
