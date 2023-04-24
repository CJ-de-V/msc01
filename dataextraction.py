from numpy.fft import fft, ifft
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sys

avgbondlength=0

def normalize(v):
    norm = np.linalg.norm(v)
    global avgbondlength
    avgbondlength+=norm
    #this is only ever called to normalize bond vectors, but before that we simply
    #record the length of said vector, should possibly also average it sometime...
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm

def skiplines(f,n):
    for i in range(n):
        f.readline()

np.set_printoptions(threshold=np.inf)
p1 = np.linspace(0., 0., 3)
p2 = np.linspace(0., 0., 3)
N=0 #Number of particles
Nb=0 #number of bonds
Nc=0 #number of correlation functions
bondvectors = 0


with open('dump.dynamics') as datafile:
    skiplines(datafile, 3)
    N=int(datafile.readline())
    Nb=N-1
    print(N)
    skiplines(datafile,5)
    bondvectors = np.zeros((Nb, 3))

    minnavg = 30 # minimum number of different pairs we want to be able to average over, allow correlating with itself
    Nc=Nb-minnavg
    correlationfunctions = np.zeros(Nc)
    # Index-1 of the above indicates how many atoms separate the bondvectors, max number should have at least 30

    line='liney'
    numavg=0 #Number of timesteps we're averaging over, used for the normalization
    while line != '':
        #print(line)
        line = (datafile.readline()).split(" ")
        p1 = np.array([float(line[2]), float(line[3]), float(line[4])])
        # position 1 1 is now real
        p2 = np.array([0.0, 0.0, 0.0])
        #reads particle 2-N in and finds the bondvectors
        for i in range(1,N):
            line=(datafile.readline()).split(" ")
            p2 = np.array([float(line[2]), float(line[3]), float(line[4])])
            bondvectors[i - 1] = normalize(np.subtract(p2, p1))
            p1 = p2

        # i is the separation between these vectors
        for i in range(Nc): #iterates over different spacings particles can have
            runninavg = 0.0
            for j in range(0, Nb-i): #iterates over all legal bonds with i bonds between them
                runninavg += np.dot(bondvectors[j], bondvectors[j + i]) #Here be where we absed
            correlationfunctions[i] += runninavg / (Nb-i)
        #print(bondvectors)
        #print(correlationfunctions)
        skiplines(datafile,8)
        line=datafile.readline()
        numavg+=1

print(numavg)
correlationfunctions=correlationfunctions/numavg

#plt.title("Correlation Function vs bond separation")
#plt.xlabel("number of separating bonds")
#plt.ylabel("correlation function")
#plt.ylim([-1, 1])
#plt.xlim([0, 100])

#start fitting procedure
# p=np.polyfit(np.arange(len(correlationfunctions)),np.log(correlationfunctions),1)
# a = np.exp(p[1])
# b = p[0]
# print(a,"and",b)
# x_fitted = np.arange(len(correlationfunctions))
# y_fitted = a * np.exp(b * x_fitted)
#
#ax = plt.axes()
# ax.scatter(x_fitted, correlationfunctions, label='Raw data')
# ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
# ax.set_title('Using polyfit() to fit an exponential function')
# ax.set_ylabel('Pair Correlation Function x#dumps')
# ax.set_xlabel('Number of separating bonds/monomers? check +-1')
#plt.axes.legend()
#End of first malarkey filled fitting procedure

#start of second curve fitting procedure
x = np.arange(len(correlationfunctions)) #x range because for SOME REASON we need that
y=correlationfunctions

def test(x, b):
    return np.exp(-x/b)

param, param_cov = curve_fit(test, x, y,maxfev=1000)
print("persistence length:")
print(param)
print("Covariance of coefficients:")
print(param_cov)
print("Average Bond Length: ")
print(avgbondlength/(Nb*numavg))
plt.text(5, 0.9, 'PL: '+str(param[0]) + 'with covar:'+str(param_cov[0,0])+'\n avg bond length: '+str(avgbondlength/(Nb*numavg)), fontsize = 14)

plt.plot(test(x,param[0]))
plt.plot(correlationfunctions,'.')
#plt.plot(fft(correlationfunctions),'r')
plt.savefig('persistenceplot.pdf')
plt.show()
