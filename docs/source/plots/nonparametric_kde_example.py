# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 2014

Author: Barbier de Reuille, Pierre
"""

from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm

dist = stats.norm(0, 1)
data = dist.rvs(3000)
model = sm.nonparametric.KDE(data)
est = model.fit()

fig = plt.figure()
grid, freqs = est.grid()
plt.hist(data, bins=data.ptp() / est.bandwidth, normed=True, label='Histogram')
plt.plot(grid.full(), freqs, label='KDE estimate')
plt.plot(grid.full(), dist.pdf(grid.full()), '--', label='Target distribution')
plt.legend(loc='best')
