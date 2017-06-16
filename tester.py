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
stockData = numpy.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,2,3,4))
# stockData2 = numpy.genfromtxt(path, delimiter=",", dtype=None, skip_header=1, usecols=(1, 2, 3, 4, 5))

# scale down the data and reverse the array
A = scale(stockData)[::-1] # A is input data
Y = A # Y is expected output

#Number of neurons in the input layer
i = 4
#Number of neurons in the output layer
o = 4
#Number of neurons in the hidden layers
h = 4
#The list of layer sizes
layers = [i, h, o]
mlpr = MLPR(layers, maxIter = 10000, tol = 0.0010, reg = 0.001, verbose = True)

#Length of the hold-out period
n = len(A)
nDays = int(round(n*.3))
#Learn the data
mlpr.fit(A[0: (n - nDays)], Y[1:(n - nDays + 1)])

#Begin prediction
yHat = mlpr.predict(A[0: (n - 1)])
#Plot the results

# mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (0)].reshape(-1, 1), c='#b04fff')
# mPlotLib.plot(list(range(nDays - 1)), yHat[:, (0)].reshape(-1, 1), c='#000000')
# # # mPlotLib.plot(A[(n-nDays): (n-1)], yHat, c='#5aa9ab')
# mPlotLib.show()


# plot with various axes scales
mPlotLib.figure(1)
nDays = n
# linear
mPlotLib.subplot(221)
mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (0)].reshape(-1, 1), c='#b04fff')
mPlotLib.plot(list(range(nDays - 1)), yHat[:, (0)].reshape(-1, 1), c='#000000')
mPlotLib.title('Open price prediction')
mPlotLib.grid(True)


# log
mPlotLib.subplot(222)
mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (1)].reshape(-1, 1), c='#b04fff')
mPlotLib.plot(list(range(nDays - 1)), yHat[:, (1)].reshape(-1, 1), c='#000000')
mPlotLib.title('High price prediction')
mPlotLib.grid(True)


# symmetric log
mPlotLib.subplot(223)
mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (2)].reshape(-1, 1), c='#b04fff')
mPlotLib.plot(list(range(nDays - 1)), yHat[:, (2)].reshape(-1, 1), c='#000000')
mPlotLib.title('Low price prediction')
mPlotLib.grid(True)

# logit
mPlotLib.subplot(224)
mPlotLib.plot(list(range(nDays - 1)), Y[(n-nDays + 1):, (3)].reshape(-1, 1), c='#b04fff')
mPlotLib.plot(list(range(nDays - 1)), yHat[:, (3)].reshape(-1, 1), c='#000000')
mPlotLib.title('Close price prediction')
mPlotLib.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
# mPlotLib.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
mPlotLib.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

mPlotLib.show()