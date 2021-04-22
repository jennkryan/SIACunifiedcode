# import necessary stuff for code
# change
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

########################################################################

class PPA:
    # Pass in order of dg approximation and smoothness of reconstruction.
    # order>0 and smoothness>=0
    # The reconstruction is computed at the points given by zEval
    # in the reference element [-1,1]
    def __init__(self,order,smoothness,zEval,L):
        self.ell_ = smoothness+2
        self.order_ = order
        self.zEval_ = zEval
        self.RS_ = int(max(ceil(0.5*(self.order_+self.ell_-1)),ceil(0.5*self.order_)));
        self.kwide_ = int(ceil(self.RS_+0.5*self.ell_))
        self.evalPoints_ = len(zEval)
        self.symcc_ = symmetricpp(self.order_,self.ell_,smoothness,self.RS_,self.evalPoints_,self.zEval_,self.zEval_)
        self.L_ = L

    # Evaluate the reconstruction at all points given in the constructor
    # for the grid cell 'nel'. uhat has to contain data for the cells
    # 'nel-kwide' to 'nel+kwide' otherwise an exception is raised.
    def evaluate(self,uhat,nel):
        if nel-self.kwide_<0 or nel+self.kwide_>=len(uhat):
            raise ValueError("element stencil not large enough")
        upost = np.zeros(self.evalPoints_)
        for j in arange(self.evalPoints_):
            for kk in arange(2*self.kwide_+1):
                ukk = uhat[nel+kk-self.kwide_][:]
                for m in arange(self.order_+1):
                    upost[j] = upost[j] + self.symcc_[kk][m][j]*ukk[m]
        return upost

########################################################################
