import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sys

# plots persistence length of the given dump file, graphical version of dataextraction.
# other version does not weight the entries by any means, this leads to long distance
# correlation function values being given more weight in the fit than they should
# trying this to see if it brings down this variance at all, though the computational cost increase is immediately
# noticable

avgbondlength = 0


def normalize(v):
    norm = np.linalg.norm(v)
    global avgbondlength
    avgbondlength += norm
    # this is only ever called to normalize bond vectors, but before that we simply
    # record the length of said vector, should possibly also average it sometime...
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def skiplines(f, n):
    for i in range(n):
        f.readline()


def test(x, b):
    return np.exp(-x / b)


np.set_printoptions(threshold=np.inf)
p1 = np.linspace(0., 0., 3)
p2 = np.linspace(0., 0., 3)
N = 0  # Number of particles
Nb = 0  # number of bonds
Nc = 0  # number of correlation functions
bondvectors = 0

with open(sys.argv[1]) as datafile:
    skiplines(datafile, 3)
    N = int(datafile.readline())
    Nb = N - 1
    print(N)
    skiplines(datafile, 5)
    bondvectors = np.zeros((Nb, 3))
    x = []  # values we'll be working on

    minnavg = 1  # minimum number of different pairs we want to be able to average over, allow correlating with itself
    Nc = Nb - minnavg
    correlationfunctions = []
    # Index-1 of the above indicates how many atoms separate the bondvectors, max number should have at least 30

    line = 'liney'
    numavg = 0  # Number of timesteps we're averaging over, used for the normalization
    while line != '':
        # print(line)
        line = (datafile.readline()).split(" ")
        p1 = np.array([float(line[2]), float(line[3]), float(line[4])])
        # position 1 1 is now real
        p2 = np.array([0.0, 0.0, 0.0])
        # reads particle 2-N in and finds the bondvectors
        for i in range(1, N):
            line = (datafile.readline()).split(" ")
            p2 = np.array([float(line[2]), float(line[3]), float(line[4])])
            bondvectors[i - 1] = normalize(np.subtract(p2, p1))
            p1 = p2

        # i is the separation between these vectors
        for i in range(Nc):  # iterates over different spacings particles can have
            runninavg = 0.0
            for j in range(0, Nb - i):  # iterates over all legal bonds with i bonds between them
                correlationfunctions = np.append(correlationfunctions, np.dot(bondvectors[j], bondvectors[j + i]))
                x = np.append(x, i)

        skiplines(datafile, 8)
        line = datafile.readline()
        numavg += 1
        print('finished timestep no.'+str(numavg))

print(len(correlationfunctions))
print(len(x))
avgbondlength = avgbondlength / (Nb * numavg)
print(numavg)

x = x * avgbondlength
y = correlationfunctions

print('starting fitting procedure')

#print(x)
#print(y)

param, param_cov = curve_fit(test, x, y, maxfev=5000)
print("persistence length:")
print(param)
print("Covariance of coefficients:")
print(param_cov)
print("Average Bond Length: ")
print(avgbondlength)
plt.text(5, 0.9,
         'PL: ' + str(param[0]) + 'with covar:' + str(param_cov[0, 0]) + '\n avg bond length: ' + str(avgbondlength),
         fontsize=14)
plt.xlim(left=0, right=max(x) + 1)
plt.plot(x, correlationfunctions, '.')
# plt.savefig('persistenceplot.pdf')
plt.plot(test(np.linspace(0,max(x)+1,100), param[0]))
plt.show()
