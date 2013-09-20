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
import scipy.io

import DistWTG
reload(DistWTG)
from DistWTG import *


Nturbine = 4
Nshooting = 50
Nsimulation = 250

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

dt = 0.2



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
#WProfiles = []
#for k in range(Nturbine):
#    #a,_ = GenWind(lambda_, kwind, TauWind,Nshooting)
#    Wk = [W0]
#    for k in range(Nsimulation+Nshooting):
#        Wk.append(Wk[-1] + rand.normalvariate(5e-3,5e-2) + 0*(8. - Wk[-1]))    
#    WProfiles.append(Wk)
#    plt.plot([k*dt for k in range(Nsimulation+Nshooting+1)],Wk)
#
#plt.show()
#raw_input()
#plt.close()
#
#Dic = {}
#for i in range(Nturbine):
#    Dic['Wind'+str(i)] = WProfiles[i]
#scipy.io.savemat('WindData2', Dic)

Dic = scipy.io.loadmat('WindData')

plt.figure()
plt.hold('on')
WProfiles = []
for i in range(Nturbine):
    Wk = Dic['Wind'+str(i)].ravel()
    WProfiles.append(list(Wk))
    
    plt.plot([k*dt for k in range(Nsimulation+Nshooting+1)],Wk)

plt.show()
raw_input()
plt.close()

T = Turbine(Inputs = ['dbeta', 'Tg'], States = ['Og', 'beta'], Slacks = ['sOg'])


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

T.setDynamics(RHS, dt = dt)

ScaleLocalCost = 1/W0**3

#Cost function
Cost  = (T.Inputs['Tg'] - T.InputsPrev['Tg'])**2#(1e-2*T.PowerVar/(0.5*rho*A)/ScaleT)**2                 # Power variation
Cost += T.Inputs['dbeta']**2                                    # Pitch rate
Cost += -Cp*T.Wind**3                                           # Power capture

Cost += T.Slacks['sOg']

CostTerminal = -Cp*T.Wind**3
CostTerminal += T.Slacks['sOg']

Cost         *= ScaleLocalCost
CostTerminal *= ScaleLocalCost

IneqConst = [
                        T.States['Og']
            ]
    
#IneqConst = [
#                        Ogmin  - T.States['Og'], # <= 0 - T.Slacks['sOg']
#                       -Ogmax  + T.States['Og']  # <= 0 - T.Slacks['sOg']
#            ]

#T.setIneqConst(IneqConst)
#T.setIneqConst(IneqConst, Terminal = True)

#Define Electrical Power
T.ElecPower(T.Inputs['Tg']*T.States['Og'])

T.setCost(Cost)
T.setCost(CostTerminal, Terminal = True)

T.setTurbine(Nshooting = Nshooting, Nsimulation = Nsimulation)



F = WindFarm(T, Nturbine = Nturbine, PowerSmoothingWeight = PowerSmoothingWeight)



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
#Og0k = [0.95*Og0 + 0.1*k*Og0/float(Nturbine-1) for k in range(Nturbine)]
beta0 = [max(0.99*BetaOpt,min(0.99*betamax,rand.normalvariate(BetaOpt,0))) for i in range(Nturbine)]

for i in range(Nturbine):    
    F.EP['Turbine',i,'States0','beta']  = beta0[i]
    F.EP['Turbine',i,'States0','Og']    = max(Ogmin,min(Ogmax,Og0k[i]))
    F.EP['Turbine',i,'Inputs0','dbeta'] = 0.
    F.EP['Turbine',i,'Inputs0','Tg']    = Tg0*ScaleT
F.EP['PowerVarRef']      = 0.



Primal, Adjoints = F.Solve(WProfiles)

time = {'States': [dt*k for k in range(Nshooting+1)],
        'Inputs': [dt*k for k in range(Nshooting)]}

timeNMPC = {'States': [dt*k for k in range(Nsimulation+1)],
            'Inputs': [dt*k for k in range(Nsimulation)]}

F.PlotBasic(T, Primal, time, 'r')

#Initial guess for the dual variables
Dual = np.array(Adjoints['PowerConst']).reshape(Nshooting,1)

##### NMPC LOOP #####
#Note: the initial conditions (and inputs) are communicated via F.EP, the wind profiles are sent independently



DistributedError = []

#Create independent copies for the distributed problem
PrimalDistributed   = F.V(Primal.cat)
AdjointsDistributed = F.g(Adjoints.cat)

ResidualLog  = []
StepSizeLog  = []
StatusLog    = []
for k in range(Nsimulation):

                       #Central Solution
                       #PrimalCentral, AdjointsCentral = F.Solve(WProfiles, time = k)
                       #F.init = PrimalCentral
                       
                       # SQP Step
                       PrimalDistributed, AdjointsDistributed, Dual, Residual, StepSize, Status = F.DistributedSQP(PrimalDistributed, AdjointsDistributed, Dual, WProfiles, time = k, iter_Dual = 1, iter_SQP = 1, FullDualStep = False, ReUpdate = True)
                       ResidualLog.append(float(np.sqrt(np.dot(Residual.T,Residual))))
                       StepSizeLog.append(StepSize)
                       StatusLog.append(Status)
                       
                       ##Plot
                       #plt.close('all')
                       #F.PlotBasic(T, PrimalDistributed,     time, 'r')
                       #raw_input()
                       #if (ResidualLog[-1] > 10):
                       #                       F.PlotBasic(T, PrimalDistributed, time, 'r')
                       #                       assert(0==1)
                       #
                       #
                       #raw_input()
                       
                       #
                       
                       #Check
                       #errorDist = PrimalCentral.cat - PrimalDistributed.cat
                       #errorDist = np.sqrt(np.dot(errorDist.T,errorDist))
                       #DistributedError.append(float(errorDist))
                       
                       #Store
                       for i in range(Nturbine):
                           F.StorageDistributed['Turbine',i,...,k] = PrimalDistributed['Turbine',i,...,0]
                           #F.StorageCentral['Turbine',i,...,k]     = PrimalCentral['Turbine',i,...,0]
                       F.StorageDistributed['PowerVar',k] = float(PrimalDistributed['PowerVar',0])     
                       #F.StorageCentral['PowerVar',k]     = float(PrimalCentral['PowerVar',0])
                       
                       #Catch the first input                   
                       F.EP['Turbine',:,'Inputs0'] = PrimalDistributed['Turbine',:,'Inputs',0]
                       
                       
                       #Actual wind profile at current time
                       Wact = [WProfiles[i][k] for i in range(Nturbine)]
                       
                       #Simulate
                       F.EP['Turbine',:,'States0'] = F.Simulate(Wact)
                       
                       error = veccat(F.EP['Turbine',:,'States0'])-veccat(PrimalDistributed['Turbine',:,'States',1])
                       print "Check simulation error = ", np.sqrt(np.dot(error.T,error))
                       
                       #Shift: Dual shifting fucks up, I dunno why !!!
                       #PrimalDistributed, AdjointsDistributed, Dual = F.Shift(PrimalDistributed, AdjointsDistributed, Dual)  
                       PrimalDistributed, AdjointsDistributed, _ = F.Shift(PrimalDistributed, AdjointsDistributed, Dual)  

                       #raw_input()
                       
plt.figure(1000)
plt.subplot(2,1,1)
plt.plot(timeNMPC['Inputs'], ResidualLog,linestyle = 'none',marker = 'o', color = 'k')
plt.title('Dual Residual')
plt.subplot(2,1,2)
plt.plot(timeNMPC['Inputs'], StepSizeLog,linestyle = 'none',marker = 'o', color = 'k')
plt.title('Dual Step Size')

#F.PlotBasic(T, F.StorageCentral,     timeNMPC, 'k')
F.PlotBasic(T, F.StorageDistributed, timeNMPC, 'r')

plt.show()