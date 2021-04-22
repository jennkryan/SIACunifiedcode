# import necessary stuff for code                                                                                               
import numpy as np
import sympy as sym


from numpy import *
from scipy import *
from scipy import integrate
from scipy.special import binom
import matplotlib.pyplot as plt
import math

#import np.linalg
import scipy.linalg   # SciPy Linear Algebra Library
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve

from math import *

#  Post-processing functions
import getkernelcoeff
import evalkernel

from getkernelcoeff import getkernelcoeff
from evalkernel import evalkernel

########################################################################                                                        
#  MAIN PART OF PROGRAM                                                                                                         
# Set number of evaluation points per element                                                                                   
evalPoints = 1

# poly degree
pin = input('Input polynomial degree (NOT ORDER):  ');   #polynomial degree  
p=int(pin)
# Define kernel smoothness                                                                              
ellp2 = input('Input smoothness required (>=0).  0 = continuous:  ');
ellp2 = int(ellp2)
# ell is the order of the B-spline
ell = ellp2 + 2

RS = p
Nx = 2*RS+ell+1

# Get quadrature points and weights for the evaluation points                                                              
zEval = np.zeros((evalPoints))
wEval = np.zeros((evalPoints))
zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)

# ASSUMING UNIFORM INTERVALS.  DOMAIN IS [-(2*RS+ell)/2,(2*RS+ell)/2].                                     
xright = np.float_(-0.5*(2*RS+ell))
xleft = np.float_(0.5*(2*RS+ell))
xlength = xright - xleft
delta_x = 1
x_grid = np.zeros((Nx+1))
for k in arange(Nx+1):
    x_grid[k] = xright+float(k)*delta_x
print('xgrid=',x_grid,'\n')
# get kernel coefficients
cgam = getkernelcoeff(ell,RS)
xEval =[]
fExact = []
for nel in arange(Nx-1):
    h = 1
    zmap = 0.5*h*(zEval+1.0) + x_grid[nel]
    f = evalkernel(ell,RS,cgam,evalPoints,zmap)

    xEval.extend(zmap)
    fExact.extend(f)

print('fExact=',fExact,'\n')
if (p==2) and (evalPoints==1):
    Exactval = np.zeros((Nx-1))
    Exactval[0] = 37/15360
    Exactval[1] = -77/640*4+2*559/960-1777/2560
    Exactval[2] = 229/256-1891/768+8137/5120
    Exactval[3] = 3739/3840
    Exactval[4] = 229/256-1891/768+8137/5120
    Exactval[5] = -77/640*4+2*559/960-1777/2560
    Exactval[6] = 37/15360

    KerErr = np.subtract(Exactval, fExact)
    print('Error in Kernel values:',KerErr,'\n')

plt.plot(xEval,fExact)
plt.show()
