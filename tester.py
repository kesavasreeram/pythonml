#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kesava Sreeram, Talabattula
"""

import numpy as numpy
from TFANN import MLPR
import matplotlib.pyplot as mPlotLib
from sklearn.preprocessing import scale

path = 'yhoo_d.csv'

# use loadtxt from numpy to extract data from the csv file
# first row contains the column names so skip that row
# skipping first columdn of the data as it contains date and the classification
# does not depend of date
stockData = numpy.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
# stockData2 = numpy.genfromtxt(path, delimiter=",", dtype=None, skip_header=1, usecols=(1, 2, 3, 4, 5))

# scale down the data and reverse the array
A = scale(stockData)[::-1] # A is input data
Y = A # Y is expected output

#Number of neurons in the input layer
i = 5
#Number of neurons in the output layer
o = 5
#Number of neurons in the hidden layers
h = 32
#The list of layer sizes
layers = [i, h, h, h, h, h, h, h, h, h, o]
mlpr = MLPR(layers, maxIter = 10000, tol = 0.010, reg = 0.001, verbose = True)

#Length of the hold-out period
n = len(A)
nDays = int(round(n*.3))
#Learn the data
mlpr.fit(A[0: (n - nDays)], Y[1:(n - nDays + 1)])

#Begin prediction
yHat = mlpr.predict(A[(n-nDays): (n-1)])
#Plot the results

mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (0)].reshape(-1, 1), c='#b04fff')
mPlotLib.plot(list(range(nDays - 1)), yHat[:, (0)].reshape(-1, 1), c='#000000')
# # mPlotLib.plot(A[(n-nDays): (n-1)], yHat, c='#5aa9ab')
mPlotLib.show()