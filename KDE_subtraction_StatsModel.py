from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats.distributions import norm
from matplotlib.legend_handler import HandlerLine2D

def kde_statsmodels_u(x, x_grid, bandwidth=10, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_m(x, x_grid, bandwidth=10, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)

#plotting grid
x_grid = np.linspace(0, 4000, 4000)

#Import data from csv file
def getdata ():
    with open('BF11Ages1s.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        ages = [map(float, row) for row in reader]

        return ages

def getmoredata ():
    with open('BF12Ages1s.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        ages = [map(float, row) for row in reader]

        return ages

#acquire KDE and subtract
pdf = kde_statsmodels_m(getdata(), x_grid, bandwidth=20)
pdf2 = kde_statsmodels_m(getmoredata(), x_grid, bandwidth=20)
difference = [a - b for a,b in zip(pdf,pdf2)]

bf11, = plt.plot(x_grid, pdf, label="BF11",color='blue', alpha=0.5, lw=1)
bf12, = plt.plot(x_grid, pdf2, label="BF12",color='red', alpha=0.5, lw=1)
diff, = plt.plot(x_grid, difference, label="Difference", color='purple', alpha=0.5, lw=1)

plt.legend()
plt.title("BF11-BF12")
plt.xlim(0, 4000)
plt.savefig('univariate.pdf')
