# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:18:08 2012

@author: sebastien
"""

from casadi import *
from numpy import *
from casadi.tools import *
import matplotlib.pyplot as plt
from pylab import matshow
import random as rand
import numpy as np
from matplotlib import interactive
interactive(True)


import DistWTG
reload(DistWTG)
from DistWTG import *

#import copy as copy

Nturbine = 4
Nshooting = 50

ScaleT = 1e-4

N = 97. 
rho = 1.23

#taug = 0.5
R = 61.5
A = pi*R**2
etag = 0.944

Ig = 534.116
Ir = 3*11.776047e6
Itot = Ig + Ir/N/N

lambdaOpt = 7.816734020931053  
BetaOpt = -1.926792728895357
Cpmax =  0.488804857667546

Ogmax = 2*pi*1173.7/60
Ogmin = 2*pi*670/60
Tgmax = 43093.55
Tgmin = 0
betamax = 15
betamin = -3
dbetamax = 7
Powmax = 5

PowerSmoothingWeight = 1e-2

W3 = R*Ogmax/lambdaOpt/N

W0 = 8.





def Norm2Weibul(lambda_, kwind, Norm):
                       
                       Phi = 0.5*(1 + sc_spec.erf(Norm/sqrt(2)))#-np.mean(Norm)
                       Wind = list(lambda_*(-np.log(1-Phi))**(1/kwind))

                       return Wind
                    
def GenWind(lambda_, kwind, TauWind,Nprofile):
                       
                       aWind = (TauWind-1)/TauWind
                       Std   = np.sqrt(1-aWind**2)

                       dNorm = [rand.normalvariate(0.,Std) for k in range(Nprofile)]
                       Norm = np.zeros(Nprofile)
                       Norm[0] = rand.normalvariate(0.,.1)
                       for k in range(Nprofile-1):
                           Norm[k+1] = aWind*Norm[k] + dNorm[k+1] 
                       
                       Wind = Norm2Weibul(lambda_, kwind, Norm)

                       return Wind, Norm

lambda_       = 9.
kwind         = 3.
TauWind       = 600.

#plt.figure()
#plt.hold('on')
Wact = []
for k in range(Nturbine):
    #a,_ = GenWind(lambda_, kwind, TauWind,Nshooting)
    Wk = [W0]
    for k in range(Nshooting):
        Wk.append(Wk[-1] + rand.normalvariate(0,5e-2))
    #plt.plot(Wk)
    
    Wact.append(Wk)
##plt.show()

T = Turbine(Inputs = ['dbeta', 'Tg'], States = ['Og', 'beta'])


#Cp interpolation
p00 =       -1.22  
p10 =      0.6322  
p01 =      0.1424  
p20 =    -0.08696  
p11 =    -0.05145  
p02 =   -0.002668  
p30 =    0.005305  
p21 =     0.00536  
p12 =   0.0009366  
p40 =  -0.0001252  
p31 =  -0.0001725 
p22 =  -0.0001384  
 

Or = T.States['Og']/N
lambda_ = R*Or/T.Wind

x = lambda_
y = T.States['beta']
Cp = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*(x**2)*y + p12*x*y**2 + p40*x**4 + p31*(x**3)*y + p22*(x**2)*(y**2)    

Tr = 0.5*rho*A*Cp*T.Wind**3/Or
dOg = (Tr/N-T.Inputs['Tg']/ScaleT)/(Ig + Ir/N/N)

RHS = [dOg,T.Inputs['dbeta']]

T.setDynamics(RHS)

#Cost function
Cost  = (1e-2*T.PowerVar/(0.5*rho*A)/ScaleT)**2                 # Power variation
Cost += T.Inputs['dbeta']**2                                    # Pitch rate
Cost += -Cp*T.Wind**3                                           # Power capture

CostTerminal = -Cp*T.Wind**3
    
#Define Electrical Power

T.ElecPower(T.Inputs['Tg']*T.States['Og'])

T.setCost(Cost)
T.setCost(Cost, Terminal = True)

T.setTurbine(Nshooting = Nshooting)

F = WindFarm(T, Nturbine = Nturbine, Nshooting = Nshooting, PowerSmoothingWeight = 1.)

# Set bounds

# -----------------------------------------------------------------------------
Or0 = lambdaOpt*W0/R
Og0 = min(Or0*N,Ogmax*0.999)
Tg0 = min(0.5*rho*A*Cpmax*W0**3/Or0/N,Tgmax*0.999)
Power0 = Tg0*Og0*ScaleT


F.init['Turbine',:,'States',:,'Og']   =  Og0
F.init['Turbine',:,'States',:,'beta'] =  BetaOpt


F.lbV['Turbine',:,'States',:,'Og']    =  Ogmin
F.ubV['Turbine',:,'States',:,'Og']    =  Ogmax

F.lbV['Turbine',:,'States',:,'beta']  =  betamin
F.ubV['Turbine',:,'States',:,'beta']  =  betamax

F.lbV['Turbine',:,'Inputs',:,'dbeta'] = -dbetamax
F.ubV['Turbine',:,'Inputs',:,'dbeta'] =  dbetamax

F.lbV['Turbine',:,'Inputs',:,'Tg']    =  Tgmin*ScaleT
F.ubV['Turbine',:,'Inputs',:,'Tg']    =  Tgmax*ScaleT

#Distribute Initial conditions

Og0k = [rand.normalvariate(Og0,0.0*Og0) for k in range(Nturbine)]

for i in range(Nturbine):
    F.lbV['Turbine',i,'States',0,'Og'] = max(Ogmin,min(Ogmax,Og0k[i]))
    F.ubV['Turbine',i,'States',0,'Og'] = max(Ogmin,min(Ogmax,Og0k[i]))
    
    beta0 = max(0.99*BetaOpt,min(0.99*betamax,rand.normalvariate(BetaOpt,0)))

    F.lbV['Turbine',i,'States',0,'beta'] = beta0
    F.ubV['Turbine',i,'States',0,'beta'] = beta0

    F.EP['Turbine',i,'States0','beta']  = beta0
    F.EP['Turbine',i,'States0','Og']    = max(Ogmin,min(Ogmax,Og0k[i]))
    F.EP['Turbine',i,'Inputs0','dbeta'] = 0.
    F.EP['Turbine',i,'Inputs0','Tg']    = Tg0*ScaleT
    
#Embbed wind profiles
for i in range(Nturbine):
    F.lbV['Turbine', i,'Wind']  = Wact[i]
    F.ubV['Turbine', i,'Wind']  = Wact[i]
    F.init['Turbine',i,'Wind']  = Wact[i]

F.EP['PowerVarRef']      = 0.

Primal, Lambdas = F.Solve()

#Initial guess for the adjoints
Adjoint = []
for i in range(Nturbine):
    Adjoint.append(np.array(Lambdas['Turbine'+str(i)]))

#Initial guess for the dual variables
Dual = np.array(Lambdas['PowerConst']).reshape(Nshooting,1)

#Perturb the solution (wind)
Per = 5e-2
for i in range(Nturbine):
    for k in range(Nshooting+1):
        Primal['Turbine',i,'Wind',k] += rand.normalvariate(0,Per)


#Embbed perturbed wind profiles
for i in range(Nturbine):
    F.lbV['Turbine', i,'Wind']  = Primal['Turbine',i,'Wind']
    F.ubV['Turbine', i,'Wind']  = Primal['Turbine',i,'Wind']
    F.init  = Primal

#Compute perturbed solution
Primal_Per, Lambdas_per = F.Solve()

Norm = []

Primal0    = dict(Primal)
Primal_Per = dict(Primal_Per)

#F.Plot(T, Primal, dt = 0.2)


##### NMPC LOOP #####

#for k in range(Nsimulation):

# SQP Step
Primal, Adjoint, Residual = F.DistributedSQP(Primal, Adjoint, Dual, iter_Dual = 5, iter_SQP = 5)
                       


Primal = dict(Primal)
F.PlotStep(Primal_Per,Primal0,'k')
F.PlotStep(Primal,Primal0,'r')


plt.show()
raw_input()
plt.close()

