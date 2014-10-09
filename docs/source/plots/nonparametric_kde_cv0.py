from scipy import stats
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

d1 = stats.norm(-2, 1)
d2 = stats.norm(1, 0.2)
data = np.r_[d1.rvs(3000), d2.rvs(3000)]

f = plt.figure()
model = sm.nonparametric.KDE(data)
est = model.fit()
grid, freqs = est.grid()
xs = grid.full()
plt.plot(xs, freqs, label='Estimate')
real = (d1.pdf(xs) + d2.pdf(xs)) / 2
plt.plot(xs, real, '--', label='Real distribution')
plt.legend(loc='best')
