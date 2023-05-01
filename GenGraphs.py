from numpy.fft import fft, ifft
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import sys

with open(sys.argv[1]) as datafile:
    fields = datafile.readline().split(', ')
    line = datafile.readline()  # reads first real entry
    df = pd.DataFrame(columns=fields)
    while line != '':
        line = line.split(', ')
        datline = np.asarray(line, dtype=float)
        df.loc[len(df)] = datline

        line = datafile.readline()

# print(df)
# we now have all our data in a pandas database. Now we can begin to plot their entries one by one


# collection of independent variables
kvals = df['k'].unique()
Nvals = df['N'].unique()


# plot ln Lp vs ln k
plt.figure(1)
for n in Nvals:
    # all the data for N = n
    ndata = df.loc[df['N'] == n]
    # print(ndata)
    x = np.log(np.array(ndata['k'].tolist()))
    y = np.log(np.array(ndata['lp'].tolist()))
    plt.plot(x, y, 'o-')

plt.title("Log-Log plot of k vs lp for various N")
plt.xlabel("Log(k)")
plt.ylabel("Log(lp)")
plt.legend(Nvals,title='Number of monomers')

plt.figure(2)

for k in kvals:
    # all the data for N = n
    ndata = df.loc[df['k'] == k]
    # print(ndata)
    x = np.log(np.array(ndata['N'].tolist()))
    y = np.log(np.array(ndata['R_g'].tolist()))
    plt.plot(x, y, 'o-')

plt.title("Log-Log plot of N vs Rg^2 for various N")
plt.xlabel("Log(N)")
plt.ylabel("Log(Rg^2)")
plt.legend(kvals,title='Bending Cost')

plt.show()


