# J.K. Ryan                                                                                                    # 15 June 2018 

# import necessary stuff for code
import numpy as np
import sympy as sym



from numpy import *
from scipy import *
from scipy import integrate
from scipy.special import binom 

import matplotlib.pyplot as plt
import math
import scipy.linalg   # SciPy Linear Algebra Library

from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from math import *

#  Post-processing functions
import symmetricpp

from symmetricpp    import symmetricpp

########################################################################
# This forms a DG-type approximation using collocation points and then
# post-process that approximaiton using the SIAC filter.
########################################################################  
#  MAIN PART OF PROGRAM
# Set number of evaluation points per element
evalPoints = input('Input number of evaluation points per element:  ');
evalPoints = int(evalPoints)

# Define number of elements, the polynomial degree
Nx = input('Input number of elements:  ');  #number of elements
p = input('Input polynomial degree (NOT ORDER):  ');   #polynomial degree
Nx = int(Nx)
p = int(p)

# Get quadrature points and weights
z = np.zeros((p+1))
w = np.zeros((p+1))
# Gauss-Legendre (default interval is [-1, 1])
z, w = np.polynomial.legendre.leggauss(p+1)

# Now  get quadrature points and weights for the evaluation points
zEval = np.zeros((evalPoints))
wEval = np.zeros((evalPoints))
zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)

# Evaluate the Legendre polynomials at p+1 points per element
LegPolyAtz = np.zeros((p+1,p+1))
LegMass = np.zeros((p+1))
for i in arange(p+1):
    if i==0:
        LegPolyAtz[i][:] = 1.0
    elif i ==1:
        LegPolyAtz[i][:] = z
    else:
        LegPolyAtz[i][:] = (2*i-1)/i*z[:]*LegPolyAtz[i-1][:]-(i-1)/i*LegPolyAtz[i-2][:]
        
    LegMass[i] = np.dot(w,LegPolyAtz[i][:]*LegPolyAtz[i][:])

#Evaluate the Legendre polynomials at the evaluation points
LegPolyAtzEval = np.zeros((p+1,evalPoints))
for i in arange(p+1):
    if i==0:
        LegPolyAtzEval[i][:] = 1.0
    elif i ==1:
        LegPolyAtzEval[i][:] = zEval
    else:
        LegPolyAtzEval[i][:] = (2*i-1)/i*zEval[:]*LegPolyAtzEval[i-1][:]-(i-1)/i*LegPolyAtzEval[i-2][:]

# ASSUMING UNIFORM INTERVALS.  DOMAIN IS [0,1].
xright = np.float_(1.0)
xleft = np.float_(0.0)
xlength = xright - xleft
delta_x = xlength/float(Nx)
x_grid = np.zeros((Nx+1))
for k in arange(Nx+1):
    x_grid[k] = float(k)*delta_x

# Define modes of approximation
uhat = np.zeros((Nx,p+1))
for nel in arange(Nx):
    h = x_grid[nel+1] - x_grid[nel]
    zmap = 0.5*h*(z+1.0) + x_grid[nel]  #affine mapping of [-1,1] quadrature points on to element

    f = np.zeros((evalPoints))
    f = np.sin(2.0*math.pi*zmap)
    
    g = np.zeros((p+1))
    for i in arange(p+1):
        g = f[:]*LegPolyAtz[i][:]
        
        uhat[nel][i] =  np.dot(w,g)
        uhat[nel][i] =  uhat[nel][i]/LegMass[i]

# Gather values in order to plot the exact solution and the projection at the evaluation points
xEval=[]
fExact=[]
fApprox=[]

for nel in arange(Nx):
    h = x_grid[nel+1] - x_grid[nel]
    zmap = 0.5*h*(zEval+1.0) + x_grid[nel]
    f = np.zeros(evalPoints)
    f = np.sin(2.0*math.pi*zmap)
    
    xEval.extend(zmap)
    fExact.extend(f)

    #Form L2-projection 
    uval = np.zeros((evalPoints))
    for j in arange(evalPoints):
        for i in arange(p+1):
            uval[j] = uval[j] + uhat[nel][i]*LegPolyAtzEval[i][j]
                
    fApprox.extend(uval)

# Calculate errors of L2-projection
ApproxErr = np.subtract(fExact, fApprox)
PtwiseAErr = np.abs(ApproxErr)
LinfAErr = np.max(PtwiseAErr)
print('\n')
print('L-inf Error for the Projection =',LinfAErr)
print('\n')
# Plot L2-projection against the exact solution
plt.figure(1)
plt.plot(xEval,fExact,xEval,fApprox,':')
plt.legend(['Exact', 'L2-projection'])


# Plot the error in log scale
plt.figure(2)
plt.semilogy(xEval,PtwiseAErr)
plt.legend(['L2-Projection'])


# Define kernel smoothness for post-processing
ellp2 = input('Input smoothness required (>=0).  0 = continuous:  ');
ellp2 = int(ellp2)
# ell is the order of the B-spline
ell = ellp2 + 2


# problemtype = input('Inpute 1=elliptic/parabolic, 2=hyperbolic:  ')
# problemtype = int(problemtype)
# print(problemtype)
# If Elliptic/Parabolic -- Define the number of splines (2*RS+2)
# if problemtype == 1:
#     if p+ell >= 2*p+2:
#         RS = p+1
#     elif p+ell <=p+1:
#         RS = int(ceil(p/2))+1
#     else:
#         RS = int(ceil((p+ell-1)/2))+1
# elif problemtype == 2:

    # If Hyperbolic -- Define the number of splines (2*RS+1)
    # Define number of splines
if p+ell >= 2*p+1:
    RS = p
elif p+ell <=p+1:
    RS = int(ceil(p/2))
else:
    RS = int(ceil((p+ell-1)/2))
print('RS = ',RS,'    Order of Spline = ',ell,'    p+ell=',p+ell,'\n')

# Half width of kernel support
kwide = int(ceil(RS+0.5*ell))
ksup = 2*kwide+1
print('kwide=',kwide,'    ksup=',ksup,'\n')


#symcc is the symmetric post-processing matrix
symcc = symmetricpp(p,ell,RS,zEval)

PPxEval=[]
PPfExact=[]
PPfApprox=[]

# Calculate post-processed soluton over whole domain
for nel in arange(Nx):
    h = x_grid[nel+1] - x_grid[nel]
    zmap = 0.5*h*(zEval+1.0) + x_grid[nel]
    f = np.zeros(evalPoints)
    f = np.sin(2.0*math.pi*zmap)
    
    PPxEval.extend(zmap)
    PPfExact.extend(f)

    upost = np.zeros(evalPoints)
    for j in arange(evalPoints):
    #Form post-processed solution
        if kwide <= nel <= Nx-2-kwide:
        # Post-process interior elements
            for kk in arange(2*kwide+1):
                kk2 = kk-kwide
                for m in arange(p+1):
                    upost[j] = upost[j] + symcc[kk][m][j]*uhat[nel+kk2][m]
        elif nel < kwide:
        # Left boundary elements
            for kk in arange(2*kwide+1):
                kk2 = kk-kwide
                for m in arange(p+1):
                    # periodic  
                    if nel+kk2 < 0:
                         upost[j] = upost[j] + symcc[kk][m][j]*uhat[Nx+nel+kk2][m]
                    else:
                        upost[j] = upost[j] + symcc[kk][m][j]*uhat[nel+kk2][m]
        elif nel > Nx-2-kwide:
        # Right boundary elements
            for kk in arange(2*kwide+1):
                kk2 = kk-kwide
                for m in arange(p+1):
                # periodic
                    if kk2 <=0:
                        upost[j] = upost[j]+symcc[kk][m][j]*uhat[nel+kk2][m]
                    else:
                        upost[j] = upost[j]+symcc[kk][m][j]*uhat[nel-Nx+kk2][m]


    PPfApprox.extend(upost)

PPApproxErr = np.subtract(PPfExact, PPfApprox)
PPPtwiseAErr = np.abs(PPApproxErr)
PPLinfAErr = np.max(PPPtwiseAErr)


print('\n')
print('L-inf Error for the Post-processed Projection =',PPLinfAErr)
print('\n')
plt.figure(1)
plt.plot(PPxEval,PPfApprox,'g')
plt.legend(['Exact','L2-Projection','Post-processed L2-projection'])
plt.savefig('L2proj.png')


plt.figure(2)
plt.semilogy(PPxEval,PPPtwiseAErr)
plt.legend(['L2-Projection Error','Post-Processed Error'])
plt.savefig('L2Err.png')

plt.show()
