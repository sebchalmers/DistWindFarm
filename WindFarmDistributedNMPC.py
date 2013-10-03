# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:18:08 2012

@author: Sebastien Gros

Assistant Professor
 
Department of Signals and Systems
Chalmers University of Technology
SE-412 96 Göteborg, SWEDEN
grosse@chalmers.se

Python/casADi Module:
A Fast Algorithm for Power Smoothing of Wind Farms based on Distributed Optimal Control

Requires the installation of the open-source Python module casADi together with the NLP solver ipopt

Required version of CasADi: v1.7.x
 
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

W0 = 9.

dt = 0.2



NewWind = False

if NewWind:
                       Nsimulation = 600/float(dt)    #10' simulation
                       print "Draw New Wind Profile"
                       plt.figure()
                       plt.hold('on')
                       
                       #Average wind speed
                       Wmean = [W0]
                       for k in range(Nsimulation+Nshooting):
                           Wmean.append(Wmean[-1] + rand.normalvariate(0,4e-2) + 2e-4*(W0 - Wmean[-1]))
                           
                       #Filter the mean speed
                       WmeanFilt = [W0]
                       for k in range(Nsimulation+Nshooting):
                           WmeanFilt.append(WmeanFilt[-1] + 1e-1*(Wmean[k] - WmeanFilt[-1])) 
                       plt.plot([k*dt for k in range(Nsimulation+Nshooting+1)],WmeanFilt,color = 'k',linewidth = 2)
                       
                       WProfiles = []
                       for k in range(Nturbine):
                           Wk = [rand.normalvariate(WmeanFilt[0],0.)]
                           for k in range(Nsimulation+Nshooting):
                               Wk.append(Wk[-1] + 4e-2*(WmeanFilt[k] - Wk[-1]) + rand.normalvariate(0,2e-1) )    
                           WkFilt = [Wk[0]]
                           for k in range(Nsimulation+Nshooting):
                               WkFilt.append(WkFilt[-1] + 1e-2*(Wk[k] - WkFilt[-1]))    
                           
                           WProfiles.append(WkFilt)
                           plt.plot([k*dt for k in range(Nsimulation+Nshooting+1)],WkFilt,color = 'r')
                       
                       plt.show()
                       raw_input()
                       plt.close()
                       #
                       Dic = {}
                       for i in range(Nturbine):
                           Dic['Wind'+str(i)] = WProfiles[i]
                       scipy.io.savemat('WindData5', Dic)

else:
                       print "Load Wind Profile"
                       Dic = scipy.io.loadmat('WindData4')
                       Nsimulation = Dic['Wind0'].shape[0]-Nshooting-1
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

 

p00 =     -0.1835 
p10 =   -0.006315  
p01 =     -0.0103  
p20 =     0.05653 
p11 =     0.02148  
p02 =    0.003457  
p30 =   -0.009622  
p21 =   -0.006306  
p12 =   -0.001859 
p03 =  -0.0003552  
p40 =   0.0005994  
p31 =   0.0005777 
p22 =   0.0002237  
p13 =   0.0001367  
p04 =   9.706e-06  
p50 =  -1.319e-05  
p41 =  -1.658e-05 
p32 =  -1.291e-05  
p23 =  -1.088e-05  
p14 =  -2.112e-06  
p05 =  -7.427e-08  


Or = T.States['Og']/N
lambda_ = R*Or/T.Wind

x = lambda_
y = T.States['beta']

Cp = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*(x**2)*y  \
     + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*(x**3)*y + p22*(x**2)*(y**2)          \
     + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*(x**4)*y + p32*(x**3)*(y**2)          \
     + p23*(x**2)*(y**3) + p14*x*y**4 + p05*y**5

Tr = 0.5*rho*A*Cp*T.Wind**3/Or
dOg = (Tr/N-T.Inputs['Tg']/ScaleT)/(Ig + Ir/N/N)

RHS = [dOg,T.Inputs['dbeta']]

T.setDynamics(RHS, dt = dt)

ScaleLocalCost = 1/W0**3

#Cost function
Cost  = (T.Inputs['Tg'] - T.InputsPrev['Tg'])**2                # Torque variation
Cost += T.Inputs['dbeta']**2                                    # Pitch  rate
Cost += -Cp*T.Wind**3                                           # Power  capture

Cost += T.Slacks['sOg']**2

CostTerminal = -Cp*T.Wind**3
CostTerminal += T.Slacks['sOg']**2

Cost         *= ScaleLocalCost
CostTerminal *= ScaleLocalCost

StageConst = [
                        Ogmin  - T.States['Og'] - T.Slacks['sOg'], # <= 0 
                       -Ogmax  + T.States['Og'] - T.Slacks['sOg'], # <= 0
             ]

TermConst = [
                        Ogmin  - T.States['Og'] - T.Slacks['sOg'], # <= 0 
                       -Ogmax  + T.States['Og'] - T.Slacks['sOg']  # <= 0 
             ]

T.setIneqConst(StageConst)
T.setIneqConst(TermConst, Terminal = True)

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

#F.lbV['Turbine',:,'States',:,'Og']    =  Ogmin
#F.ubV['Turbine',:,'States',:,'Og']    =  Ogmax

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



#Initial guess for the dual variables
Dual = np.array(Adjoints['PowerConst']).reshape(Nshooting,1)

##### NMPC LOOP #####
#Note: the initial conditions (and inputs) are communicated via F.EP, the wind profiles are sent independently



DistributedError = []

#Create independent copies for the distributed problem
PrimalDistributed   = F.V(Primal.cat)
AdjointsDistributed = F.g(Adjoints.cat)

ResidualLog     = []
StepSizeLog     = []
StatusLog       = []
CondLog         = []
ErrorLog        = []
DualLog         = []
ASLog           = []
AdjointLog      = []
MuLog           = []
ActivationLog   = []
DeActivationLog = []
ALog            = []
GapLog          = []
MuCheckLog      = []

for k in range(Nsimulation):

                       #Central Solution
                       #if k > 0:
                       #    F.EP['Turbine',:,'States0'] = StatePlusCentral
                       #PrimalCentral, AdjointsCentral = F.Solve(WProfiles, time = k)
                       #F.init = PrimalCentral

                       ## SQP Step
                       if k > 0:
                            F.EP['Turbine',:,'States0'] = StatePlusDistributed
                       PrimalDistributed, AdjointsDistributed, Dual, Residual, StepSize, Status, CondHess, Error, A, Mu, Gap, MuCheck, QPs = F.DistributedSQP(PrimalDistributed, AdjointsDistributed, Dual, WProfiles, time = k, iter_Dual = 1, iter_SQP = 1, FullDualStep = True, ReUpdate = True)
                       
                       AdjointLog.append(np.array(AdjointsDistributed.cat))
                       MuLog.append(Mu)
                       GapLog.append(Gap)
                       MuCheckLog.append(MuCheck)
                       
                       #Check Activation/Deactivation
                       if (k == 0):
                                              ActivationLog.append([])
                                              DeActivationLog.append([])
                       else:
                                              Activation   = []
                                              DeActivation = []
                                              for i in range(Nturbine):
                                                  Activation_i   = []
                                                  DeActivation_i = []
                                                  for index in A['AB'][i]:
                                                      if not(index in ALog[-1]['AB'][i]):
                                                          Activation_i.append(F._TurbineV.getLabel(index))
                                                  for index in ALog[-1]['AB'][i]:
                                                      if not(index in A['AB'][i]):
                                                          DeActivation_i.append(F._TurbineV.getLabel(index))
                                                      
                                                  Activation.append(    Activation_i)
                                                  DeActivation.append(DeActivation_i)   
                                              
                                              ActivationLog.append(    Activation)
                                              DeActivationLog.append(DeActivation)
                       ALog.append(A)
                       

                       ## Log info
                       ResidualLog.append(float(np.sqrt(np.dot(Residual.T,Residual))))
                       StepSizeLog.append(StepSize)
                       StatusLog.append(Status)
                       CondLog.append(CondHess)
                       ErrorLog.append(Error)
                       DualLog.append(np.array(Dual.T))
                                              
                       #Store
                       for i in range(Nturbine):
                           F.StorageDistributed['Turbine',i,...,k] = PrimalDistributed['Turbine',i,...,0]
                           #F.StorageCentral['Turbine',i,...,k]     = PrimalCentral['Turbine',i,...,0]
                       F.StorageDistributed['PowerVar',k] = float(PrimalDistributed['PowerVar',0])     
                       #F.StorageCentral['PowerVar',k]     = float(PrimalCentral['PowerVar',0])
                       
                       #Actual wind profile at current time
                       Wact = [WProfiles[i][k] for i in range(Nturbine)]
                       
                       #Simulate Distributed
                       F.EP['Turbine',:,'Inputs0'] = PrimalDistributed['Turbine',:,'Inputs',0]
                       StatePlusDistributed = F.Simulate(Wact)
                       
                       ##Simulate Central
                       #F.EP['Turbine',:,'Inputs0'] = PrimalCentral['Turbine',:,'Inputs',0]
                       #StatePlusCentral = F.Simulate(Wact)

                       #Shift: Dual shifting fucks up, I dunno why !!!
                       PrimalDistributed, AdjointsDistributed, _ = F.Shift(PrimalDistributed, AdjointsDistributed, Dual)  

                                 

timeDiplay = {'States': [dt*j for j in range(k+1)],
              'Inputs': [dt*j for j in range(k)]}
                       
#plt.figure(1000)
#plt.subplot(2,1,1)
#plt.plot(timeDiplay['States'], ResidualLog,linestyle = 'none',marker = 'o', color = 'k')
#plt.title('Dual Residual')
#plt.subplot(2,1,2)
#plt.plot(timeDiplay['States'], np.array(StepSizeLog),linestyle = 'none',marker = 'o', color = 'k')
#plt.title('Dual Step Size')
#
#CondLog = np.array(CondLog)
#plt.figure(1001)
#for fig in [0,1]:
#    plt.subplot(2,1,fig)
#    plt.plot(timeDiplay['States'], CondLog[:,fig])
#
#plt.figure(1002)
#for i in range(Nturbine):
#    plt.subplot(4,1,i)
#    plt.plot(timeNMPC['Inputs'],F.StorageDistributed['Turbine',i,'Slacks',:-1,'sOg'],color='k')
#
#
##F.PlotBasic(T, F.StorageCentral,     timeNMPC, col = 'k', style = 'x', LW = 2)
#F.PlotBasic(T, F.StorageDistributed, timeNMPC, col = 'k', style = '-')
#
#plt.show()
#
#plt.figure(1000)
#plt.subplot(2,1,1)
#plt.plot(timeDiplay['Inputs'], ResidualLog,linestyle = 'none',marker = 'o', color = 'k')
#plt.title('Dual Residual')
#plt.subplot(2,1,2)
#plt.plot(timeDiplay['Inputs'], np.array(StepSizeLog),linestyle = 'none',marker = 'o', color = 'k')
#plt.title('Dual Step Size')
#
#CondLog = np.array(CondLog)
#plt.figure(1001)
#for fig in [0,1]:
#    plt.subplot(2,1,fig)
#    plt.plot(timeDiplay['Inputs'], CondLog[:,fig])
#
##plt.figure(1002)
##for i in range(Nturbine):
##    plt.subplot(4,1,i)
##    plt.plot(timeNMPC['Inputs'],F.StorageDistributed['Turbine',i,'Slacks',:-1,'sOg'],color='k')
#

#F.PlotBasic(T, F.StorageCentral,     timeNMPC, col = 'k', style = 'x', LW = 2)
F.PlotBasic(T, F.StorageDistributed, timeNMPC, col = 'k', style = '-')

plt.show()