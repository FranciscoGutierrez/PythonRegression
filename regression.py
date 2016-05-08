import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn import datasets, linear_model
from pymongo import MongoClient

### MongoDB Calling
client = MongoClient("mongodb://127.0.0.1:3001/meteor")
db = client['meteor']
###

data = db.tweets_boston.find({},{"polarity": 1, "quality_of_life_index":1, "_id":0})
# quality_of_life_index
#### Positives ####
# health_care_index
# safety_index
#### Negatives ####
# traffic_index
# pollution_index
polarity = []
index = []

for post in data:
    index.append(post.values()[0])
    polarity.append(post.values()[1])

# print len(polarity)
# print len(index)
#
# index = np.array(index).reshape((len(index), 1))
# #index = np.array(index).reshape((len(index), 1))
#
# # Use only one feature
# # polarity  = data.polarity[:, np.newaxis, 2]
# # index = data.pollution_index
# # Create linear modelession object
# model = linear_model.LinearRegression()
# model.fit(index, polarity)
#
# # The coefficients
# print('Coefficients: \n', model.coef_)
# print('Intercept: \n', model.intercept_)
#
# plt.scatter(index, polarity, color='black')
# plt.plot(index, model.predict(index), color='blue',linewidth=3)
# plt.show()
prob = 0.95
x = np.array(index)
y = np.array(polarity)
n = len(x)
xy = x * y
xx = x * x
# estimates
b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
b0 = y.mean() - b1 * x.mean()
s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in xrange(n)])
print 'b0 = ',b0
print 'b1 = ',b1
print 's2 = ',s2
#confidence intervals
alpha = 1 - prob
c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
print 'the confidence interval of s2 is: ',[n*s2/c2,n*s2/c1]
c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
