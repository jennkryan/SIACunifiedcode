# import necessary stuff for code                                                                                                                             
import numpy as np

from numpy import *

########################################################################                                                                                      
#symmetric post-processing coefficients for p=1 to confirm whether code works   

def exactp1convcoeff(zEval):

    Neval = np.size(zEval)
    exsym0 = np.zeros((5,2,Neval))

    for j in arange(Neval):
        Zeta = zEval[j]
  
        exsym0[0][0][j] = -(-1 + Zeta) ** 2 / 96
        exsym0[0][1][j] = -0.1e1 / 0.144e3 - Zeta ** 3 / 288 + Zeta / 96

        exsym0[1][0][j] = 0.1e1 / 0.12e2 - 0.7e1 / 0.24e2 * Zeta + Zeta ** 2 / 6
        exsym0[1][1][j] = 0.7e1 / 0.72e2 - Zeta / 6 + Zeta ** 3 / 18

        exsym0[2][0][j] = 0.41e2 / 0.48e2 - 0.5e1 / 0.16e2 * Zeta ** 2
        exsym0[2][1][j] = .5e1 / 0.16e2 * Zeta - 0.5e1 / 0.48e2 * Zeta ** 3

        exsym0[3][0][j] = 0.7e1 / 0.24e2 * Zeta + 0.1e1 / 0.12e2 + Zeta ** 2 / 6
        exsym0[3][1][j] =  -0.7e1 / 0.72e2 - Zeta / 6 + Zeta ** 3 / 18

        exsym0[4][0][j] = -(Zeta + 1) ** 2 / 96
        exsym0[4][1][j] = Zeta / 96 + 0.1e1 / 0.144e3 - Zeta ** 3 / 288

    return exsym0

###################################################################################
