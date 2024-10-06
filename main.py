import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.dates as mdates

X_data = np.random.random(50) * 100
Y_data = np.random.random(50) * 100

# plt.scatter(X_data, Y_data)
# plt.plot([1,2,3,4,5], [2,3,4,5,6])
# plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
print(t)


# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs' , t, t**3, 'g^')
# plt.show()

# Plotting with keyword strings
# data = {'a' : np.arange(50),
#         'b' : np.random.randint(0, 50, 50),
#         'c' : np.random.randn(50)}

# data['b'] = data['a'] + 10 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100

# plt.scatter('a', 'b',c='c', s='d', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')
# plt.show()


# names = ['Berkcan', 'Mariia', 'Kuzja']
# values = [100, 200, 300]

# plt.subplot(131)
# plt.bar(names, values)
# plt.subplot(132)
# plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(names, values)
# plt.suptitle('Categorical Plotting')
# plt.show()
# plt.show()

# x = [1,2,3,4,5] * 10
# y = [2,3,4,5,7] * 10
# # print(x)
# # print(y)
# # plt.plot(x, y, linewidth=2)
# # plt.show()

# line, = plt.plot(x, y, '-')
# line.set_antialiased(False) # turn off antialiasing
# plt.show()

# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)

# plt.figure()
# plt.subplot(211)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()


data = pd.read_csv('btc-market-price.csv')
data.columns = ['TimeStamp', 'Price']
data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
# print(data)

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# data = np.array(ct.fit_transform(data))
# print(data)


plt.plot(data['TimeStamp'], data['Price'], c='r', linewidth=1.0)
# Format the x-axis as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Optionally, rotate the date labels for better visibility
plt.gcf().autofmt_xdate()
plt.show()
