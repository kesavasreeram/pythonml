#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kesava Sreeram, Talabattula
"""

import numpy as numpy
from TFANN import MLPR
import matplotlib.pyplot as mPlotLib
from sklearn.preprocessing import scale

path = 'yahooData.csv'

# use loadtxt from numpy to extract data from the csv file
# first row contains the column names so skip that row
# skipping first columdn of the data as it contains date and the classification
# does not depend of date
stockData = numpy.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5))
stockData2 = numpy.genfromtxt(path, delimiter=",", dtype=None, skip_header=1, usecols=(1, 2, 3, 4, 5))

# data is in descending order of dates with latest at the top and oldest at the bottom
# so reverse the list to get the data in a time series format
stockDataASC = list(reversed(stockData))

# since the stock data is in huge numbers lets scale it down using "scale" funcationality
print(stockData2)

print(scale(stockDataASC))

# to get only specific columns of data from a array use :   
# data[:, [1, 9]] where data is array and you want columns 1, 9 (index start from 0)

# scale the stock data, volume to ease the calculations and fit within the data range

# Number of neurons in the input layer
# 4 neurons to indicate the candle stick doji patterns
# 4 neurons to indicate the previous most tested highs which are greater than previous day closing prices
# 4 neurons to indicate the previous most tested lows which are less than previous day closing prices
# 5 neurons to indicate the previous day open, close, high and low prices, and volume
i = 17
# Number of neurons in the output layer
# 5 neurons to indicate the current day open, close, high and low prices and volume
o = 5
#Number of neurons in the hidden layers
h = 17
#The list of layer sizes
layers = [i, h, h, h, h, h, h, h, h, h, o]
mlpr = MLPR(layers, maxIter = 1000, tol = 0.40, reg = 0.001, verbose = True)
mlpr.fit()

#Begin prediction
yHat = mlpr.predict(A)
#Plot the results
mpl.plot(A, Y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()