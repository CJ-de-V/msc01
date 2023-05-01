import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys


# Pandas dataframe, x dependent & y independent variable, variable on legend and number of figure for plotting
# figure numbers have to be distinct
def loglogplot(datfram, dependent, independent, legend, number):
    plt.figure(number)
    legset = datfram[legend].unique()

    for leg in legset:
        ndata = df.loc[df[legend] == leg]
        x = np.log(np.array(ndata[dependent].tolist()))
        y = np.log(np.array(ndata[independent].tolist()))
        plt.plot(x, y, 'o-')

    plt.title("Log-Log plot of " + dependent + " vs " + independent + " for various " + legend)
    plt.xlabel("Log(" + dependent + ")")
    plt.ylabel("Log(" + independent + ")")
    plt.legend(legset, title=legend)


with open(sys.argv[1]) as datafile:
    fields = datafile.readline().split(', ')
    line = datafile.readline()  # reads first real entry
    df = pd.DataFrame(columns=fields)
    while line != '':
        line = line.split(', ')
        datline = np.asarray(line, dtype=float)
        df.loc[len(df)] = datline

        line = datafile.readline()

# plot ln Lp vs ln k

loglogplot(df, 'k', 'lp', 'N', 1)

# plot ln N vs ln Rg^2

loglogplot(df, 'N', 'R_g', 'k', 2)

# plot ln R_g vs ln lp

loglogplot(df, 'R_g', 'lp', 'k', 3)

plt.show()
