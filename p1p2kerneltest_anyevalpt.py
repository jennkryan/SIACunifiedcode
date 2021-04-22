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
import symmetricpp
import p2postpcoeff
import exactp2convcoeff
import exactp1convcoeff

from symmetricpp    import symmetricpp
from p2postpcoeff   import p2postpcoeff
from exactp2convcoeff import exactp2convcoeff
from exactp1convcoeff import exactp1convcoeff

########################################################################


########################################################################  
#  MAIN PART OF PROGRAM
# Set number of evaluation points per element and the polynoial order
evalPoints = 2
p = 2


# Get quadrature points and weights
gpts = p+3
z = np.zeros((gpts))
w = np.zeros((gpts))
# Gauss-Legendre (default interval is [-1, 1])
z, w = np.polynomial.legendre.leggauss(gpts)

# Now  get quadrature points and weights for the evaluation points
zEval = np.zeros((evalPoints))
wEval = np.zeros((evalPoints))
zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)

# B-spline order
ell = p+1

# Define the number of splines (2*RS+1)
RS = int(max(ceil(0.5*(p+ell-1)),ceil(0.5*p)));
kwide = int(ceil(RS+0.5*ell))

#symcc is the symmetric post-processing matrix
symcc = symmetricpp(p,ell,RS,zEval)

# Exact post-processing coefficients
if p == 2:
    exactcc = exactp2convcoeff(zEval)
elif p==1:
    exactcc = exactp1convcoeff(zEval)

#hard coded matrix using monomials over interval scaled to (-0.5,0.5)
#symcc0 = p2postpcoeff(p,evalPoints)

#Translate to Legendre
#ccmod = np.zeros((2*kwide+1,p+1,evalPoints))
#for kk in arange(2*kwide+1):        
#    ccmod[kk][0][:] = symcc0[kk][0][:]
#    ccmod[kk][1][:] = 2*symcc0[kk][1][:]
#    ccmod[kk][2][:] = 6*symcc0[kk][2][:]-0.5*symcc0[kk][0][:]


# [0] looks like precision error -- 10^{-77} is replaced by zero
print('\n')
print('RS=',RS,'kwide=',kwide)
print('\n')

print('symcc:  size',symcc.shape)
print('exact cc:  size',exactcc.shape)
#print('ccmod: size',ccmod.shape)


print('Evaluation points=',zEval)
print('Difference in convolution coefficients:')
for kk in arange(2*kwide+1):
    calc = symcc[kk][0][0]
    exactc = exactcc[kk][0][0]
#    hardc = ccmod[kk][0][0]
    differenceEx = exactc - calc
#    differenceH = hardc-calc
#    differenceExH = exactc - hardc
#    difference = hardc-exactc
#    print('kk=',kk,'exactc[kk][2][0]=',exactc,'symcc[kk][2][0]=',calc)
    print('   kk=',kk,'    diffEx=',differenceEx)
#    print('    diffH=',differenceH,'    differenceExH=',differenceExH)
print('\n')

