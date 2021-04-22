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

from symmetricpp    import symmetricpp
from p2postpcoeff   import p2postpcoeff

########################################################################


########################################################################  
#  MAIN PART OF PROGRAM
# Set number of evaluation points per element and the polynoial order
evalPoints = 6
p = 2


# Get quadrature points and weights

z = np.zeros((p+1))
w = np.zeros((p+1))
# Gauss-Legendre (default interval is [-1, 1])
z, w = np.polynomial.legendre.leggauss(p+1)

# Now  get quadrature points and weights for the evaluation points
zEval = np.zeros((evalPoints))
wEval = np.zeros((evalPoints))
zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)

# B-spline order
ell = 3

# Define the number of splines (2*RS+1)
RS = int(max(ceil(0.5*(p+ell-1)),ceil(0.5*p)));
kwide = int(ceil(RS+0.5*ell))

#symcc is the symmetric post-processing matrix
symcc = symmetricpp(p,ell,RS,zEval)

#hard coded matrix using monomials over interval scaled to (-0.5,0.5)
symcc0 = p2postpcoeff(p,evalPoints)

#Translate to Legendre
ccmod = np.zeros((2*kwide+1,p+1,evalPoints))
for kk in arange(2*kwide+1):        
    ccmod[kk][0][:] = symcc0[kk][0][:]
    ccmod[kk][1][:] = 2*symcc0[kk][1][:]
    ccmod[kk][2][:] = 6*symcc0[kk][2][:]-0.5*symcc0[kk][0][:]


# [0] looks like precision error -- 10^{-77} is replaced by zero
print('\n')
print('RS=',RS,'kwide=',kwide)
print('\n')

print('symcc:  size',symcc.shape)
print('ccmod: size',ccmod.shape)

diff = np.zeros((2*kwide+1,p+1,evalPoints))
diff = symcc - ccmod
print('diff=',diff)
#for kk in arange(2*kwide+1):
#    print('kk=',kk,'\n')
#    for j in arange(evalPoints):
#    j=0
#    print('j=',j,diff[kk][:][j],'\n')
    #calc = symcc[kk][0][0]
    #hardc = ccmod[kk][0][0]
#    difference = hardc-calc
#    print('kk=',kk,'symcc[kk][0][0]=',calc,'ccmod[kk][0][0]=',hardc)
#    print('   diff=',difference)
#print('\n')

