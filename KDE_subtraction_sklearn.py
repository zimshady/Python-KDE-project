from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats.distributions import norm
from matplotlib.legend_handler import HandlerLine2D

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

#plotting grid
x_grid = np.linspace(0, 4000, 4000)

#Import data from csv file
def getdata ():
    with open('C:\Google Drive\Python KDE project\BF11Ages1s.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        ages = [map(float, row) for row in reader]
        return ages

def getmoredata ():
    with open('C:\Google Drive\Python KDE project\BF12Ages1s.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        ages = [map(float, row) for row in reader]
        return ages

#acquire KDE and subtract
pdf = kde_sklearn(getdata(), x_grid, bandwidth=20)
pdf2 = kde_sklearn(getmoredata(), x_grid, bandwidth=20)
difference = [a - b for a,b in zip(pdf,pdf2)]
print pdf.get_params()
bf11, = plt.plot(x_grid, pdf, label="BF11",color='blue', alpha=0.5, lw=1)
bf12, = plt.plot(x_grid, pdf2, label="BF12",color='red', alpha=0.5, lw=1)
diff, = plt.plot(x_grid, difference, label="Difference", color='purple', alpha=0.5, lw=1)

plt.legend()
plt.title("BF11-BF12")
plt.xlim(0, 4000)
plt.savefig('sklearn.pdf')
