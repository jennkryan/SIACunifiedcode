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


####################################################################### 

def dgbreaksequalbsbreaks(p,kwide,pwide,L,z,w,RS,ell,cgam,zEvalj):
    for kk1 in arange(pwide):
        if kk1 != 0 and kk1 !=0 pwide-1:
            kk = kk1-kwide
            ahat = -1.0
            bhat = 1.0
            xintsum = 0.
            kerzeta = np.zeros(gpts)
            # Legendre polynomials evaluated at the gauss points                                                              
            PLeg = np.zeros((p+1,gpts))
            for jj in arange(gpts):
                kerzeta[jj] = (zEvalj-z[jj])/L-np.float(kk)
                y = 0.5*L*z[jj]
                for i in arange(p+1):
                    if i==0:
                        PLeg[i][jj] = 1.0
                    elif i ==1:
                        PLeg[i][jj] = y
                    else:
                        PLeg[i][jj] = (2*i-1)/i*y*PLeg[i-1][jj]-(i-1)/i*PLeg[i-2][jj]

                fker = evalkernel(ell,RS,cgam,gpts,kerzeta)
                xintsum[kk1] = sum(fker[jj]*PLeg[m][jj]*w[jj] for jj in arange(gpts))
        
    return xintsum
