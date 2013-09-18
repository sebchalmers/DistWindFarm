# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:18:08 2012

@author:

Sebastien Gros
Assistant Professor
 
Department of Signals and Systems
Chalmers University of Technology
SE-412 96 Gšteborg, SWEDEN
grosse@chalmers.se

Python/casADi Module:
NMPC for Dynamic Optimal Power Flow and Power Dispatch

Requires the installation of the open-source Python module casADi together with the NLP solver ipopt

Required version of CasADi: v1.7.x

OBS: the code assumes only simple bounds !! Can be extended by modifying method "QPStep"...

"""

from casadi import *
from casadi.tools import *
#import math as math

import numpy as np
import scipy.special as sc_spec
from scipy import linalg

import matplotlib.pyplot as plt
from pylab import matshow

def assertList(var):
    if not(isinstance(var,list)):
        var = [var]
    return var

#Solver Constructor (used in both Turbine and WindFarm class)
def _setSolver(self, V,Cost,g,P):

    lbg = g()
    ubg = g()
    
    nl = MXFunction(nlpIn(x=V,p=P),nlpOut(f=Cost,g=g))
    nl.init()
    
    # set-up solver
    solver = IpoptSolver(nl)  
    solver.setOption("expand",True)
    #solver.setOption("print_level",1)  
    solver.setOption("hessian_approximation","exact")
    solver.setOption("max_iter",2000)
    solver.setOption("tol",1e-12)
    solver.setOption("linear_solver","ma27")
    
    solver.init()
    
    H = solver.hessLag()
    H.init()
    dg = solver.jacG()
    dg.init()
    
    gfunc = MXFunction([V,P],[g])
    gfunc.init()
    
    f = solver.gradF() 
    f.init()
    
    Solver = {'solver': solver, 'H': H, 'f': f, 'dg': dg, 'g': gfunc, 'lbg': lbg, 'ubg': ubg, 'V': V}
    
    return Solver

class Turbine:
    def __init__(self, Inputs = [], States = []):        

        self._frozen = False
            
        InputList = []
        for key in assertList(Inputs):
            InputList.append(entry(key))
            
        # States declared by the user
        StateList = []
        for key in assertList(States):
            StateList.append(entry(key)) 
    
        P = ssym('P')
        W = ssym('W')

        # lists of names (strings)
        self.States            = struct_ssym(StateList)
        self.Inputs            = struct_ssym(InputList)
        self.StatesPrev        = struct_ssym(StateList)
        self.InputsPrev        = struct_ssym(InputList)
        self.Wind              = W
        self.PowerVar          = P
            
    
    def ElecPower(self, expr):
        Power = SXFunction([self.States,self.Inputs],[expr])
        Power.init()
        
        self.ElecPower = Power
        
    def setDynamics(self, RHS = [], dt = .2, nstep = 10):
        if (self._frozen == True):
            print "Plant already added to the grid, call ignored"
            return

        
        print "Right-Hand side: ", RHS
        
        if isinstance(RHS,list):
            RHS = veccat(RHS)
        
        X = self.States
        U = self.Inputs
        W = self.Wind
        
        dtRK4 = dt/float(nstep)

        #CONSTRUCT SHOOTING
        # -----------------------------------------------------------------------------
        f = SXFunction([X,U,W],[RHS])
        f.init()
        
        [k1]  = f.eval([X               ,U, W])
        [k2]  = f.eval([X+0.5*dtRK4*k1  ,U, W])
        [k3]  = f.eval([X+0.5*dtRK4*k2  ,U, W])
        [k4]  = f.eval([X+dtRK4*k3      ,U, W])
        
        rk4_step = SXFunction([X,U,W],[X + (1./6)*dtRK4*(k1 + 2*k2 + 2*k3 + k4)])
        rk4_step.init()
        
        out = X
        for i in range(0,nstep):
            [out] = rk4_step.eval([out,U,W])
            
        Shoot = SXFunction([X,U,W],[out])        
        Shoot.init()
        
        self.Shoot = Shoot
        
        
    def _BuildFunc(self, Expr, Terminal):

        X           = self.States
        U           = self.Inputs
        Uprev       = self.InputsPrev
        Xprev       = self.StatesPrev
        PowerVar    = self.PowerVar
        W           = self.Wind  
            
        if Terminal == False:
            listFuncInput = [U, X, W, Uprev, Xprev, PowerVar]            
        else:
            listFuncInput = [X, W, Xprev]
            
        Func = SXFunction(listFuncInput,[Expr])
        Func.init()
        
        return Func

    def setConstraints(self, Const, Terminal = False):
        
        if (self._frozen == True):
            print "Wind farm already created, call ignored"
            return
        
        if not(isinstance(Const,list)):
            Const = [Const]
        
        #ConstFunc = self._BuildFunc(veccat(Const), Terminal)
                
        if    (Terminal == False):
            self._StageConst    = self._BuildFunc(veccat(Const), Terminal)
        elif  (Terminal == True): 
            self._TerminalConst = self._BuildFunc(veccat(Const), Terminal)        
                
        
    def setCost(self, Expr, Terminal = False):
        if (self._frozen == True):
            print "Wind farm already created, call ignored"
            return
        
        if    (Terminal == False):
            self._StageCost    = self._BuildFunc(Expr, Terminal)
        elif  (Terminal == True): 
            self._TerminalCost = self._BuildFunc(Expr, Terminal)
            
            


    def _CreateQP(self, solver,V):
        
        solver['H'].evaluate()
        H = solver['H'].output()
        
        solver['dg'].evaluate()
        dg = solver['dg'].output()
        
        QPsolver = NLPQPSolver(qpStruct(h=H.sparsity(),a=dg.sparsity()))
        QPsolver.setOption({"nlp_solver":IpoptSolver, "nlp_solver_options": {"tol": 1e-12,"verbose": 0} })
       
        QPsolver.init()
    
        return QPsolver

    def _setTurbineVariables(self,Nelements):
        Variables  = struct_msym([
                                    entry('States',   struct = self.States,   repeat = Nelements+1),
                                    entry('Inputs',   struct = self.Inputs,   repeat = Nelements),
                                    entry('Wind',     struct = self.Wind,     repeat = Nelements+1),
                                    entry('PowerVar', struct = self.PowerVar, repeat = Nelements)
                                ])
        
        return Variables
    
    def setTurbine(self, Nshooting = 50, Nsimulation = 50 ):
        
        self.Nshooting   = Nshooting
        self.Nsimulation = Nsimulation
        
        self.V       = self._setTurbineVariables(Nshooting)
        self.Storage = self._setTurbineVariables(Nsimulation)
        
        EP         = struct_msym([
                                    entry('States0', struct = self.States),
                                    entry('Inputs0', struct = self.Inputs)
                                 ])
        
        self.EP = EP
        

        #### Setup Local Cost and Constraints ####
        DynConst = []
        PowerConst = []
        Cost = 0
        
        #N elements
        for k in range(Nshooting):
            #Dynamics 
            [Xp] = self.Shoot.call([self.V['States',k],self.V['Inputs',k],self.V['Wind',k]])    
            DynConst.append(Xp-self.V['States',k+1])
        
            #Power Variation
            if (k == 0):
                [Power_minus] = self.ElecPower.call([self.EP['States0']  ,self.EP['Inputs0']])
            else:
                [Power_minus] = self.ElecPower.call([self.V['States',k-1],self.V['Inputs',k-1]])
                
            [Power]       =  self.ElecPower.call([self.V['States',k],self.V['Inputs',k]])    
            PowerVar = Power - Power_minus
            PowerConst.append(self.V['PowerVar',k]-PowerVar)
        
            #Power variation        
            if (k == 0):
                [StageCost] = self._StageCost.call([
                                                    self.V['Inputs',k],
                                                    self.V['States',k],
                                                    self.V['Wind',k],
                                                    self.EP['Inputs0'],
                                                    self.EP['States0'],
                                                    self.V['PowerVar',k]
                                                    ])
                Cost += StageCost
            else:
                [StageCost] = self._StageCost.call([
                                                    self.V['Inputs',k],
                                                    self.V['States',k],
                                                    self.V['Wind',k],
                                                    self.V['Inputs',k-1],
                                                    self.V['States',k-1],
                                                    self.V['PowerVar',k]
                                                    ])
            Cost += StageCost
            
        k = Nshooting
        [TerminalCost] = self._TerminalCost.call([
                                                    self.V['States',k],
                                                    self.V['Wind',k],
                                                    self.V['States',k-1]
                                                ])
        Cost += TerminalCost
                             
         
        g =      struct_MX([
                              entry('DynConst',      expr = DynConst),
                              entry('PowerConst',    expr = PowerConst)
                           ])     
        
        
        
        ConstFun = MXFunction([self.V,self.EP],[g])
        ConstFun.init()
        self._Const = ConstFun
        
        CostFun  = MXFunction([self.V,self.EP],[Cost])
        CostFun.init()
        self._Cost = CostFun
        
        #Create a local solver to generate the underlying QP
        Solver = _setSolver(self,self.V,Cost,g,self.EP)
        self._Solver = Solver
        
        #Prepare the QP call
        QPsolver = self._CreateQP(Solver,self.V)
        self._QPsolver = QPsolver

     
class WindFarm:
    
    def __init__(self, Turbine, Nturbine = 0, PowerSmoothingWeight = 0.):     
        
        
        self.Nturbine   = Nturbine
        Nshooting       = Turbine.Nshooting
        Nsimulation     = Turbine.Nsimulation

        self.Nshooting            = Nshooting
        self.PowerSmoothingWeight = PowerSmoothingWeight
        
        

        
        #Carry over local stuff
        self.Nsimulation    = Nsimulation
        self._TurbineSolver = Turbine._Solver
        self._QPsolver      = Turbine._QPsolver
        self._VTurbine      = Turbine.V
        self._Shoot         = Turbine.Shoot
        
        # Container for the global decision variables
        V          =       struct_msym([
                                            entry('Turbine',   struct = Turbine.V,              repeat = Nturbine),
                                            entry('PowerVar',  struct = Turbine.PowerVar,       repeat = Nshooting)
                                        ])

    
        self.V = V
        
        EP          =       struct_msym([
                                            entry('Turbine',     struct = Turbine.EP,             repeat = Nturbine  ), 
                                            entry('PowerVarRef', struct = Turbine.PowerVar,       repeat = Nshooting )
                                        ])
        
        self.EP = EP()
                                   
        
        Storage     = struct_msym([
                                    entry('Turbine',  struct = Turbine.Storage,   repeat = Nturbine),
                                    entry('PowerVar', struct = Turbine.PowerVar,  repeat = Nsimulation)
                                  ])
        
        self.StorageCentral     = Storage()
        self.StorageDistributed = Storage()
        
        #### Centralized Problem ####

        #Power variation, cost and constraint
        TotPowerVar = []
        Cost = 0
        Const = []
        
        #Wind farm total power variation & total power constraints
        for k in range(Nshooting):
            TotPowerVar_k = V['PowerVar', k]
            for i in range(Nturbine):            
                TotPowerVar_k -= V['Turbine',i,'PowerVar',k]
                
            TotPowerVar.append(TotPowerVar_k)
            Cost += 0.5*PowerSmoothingWeight*(EP['PowerVarRef',k]-V['PowerVar', k])**2
            
        Const.append(entry('PowerConst',    expr = TotPowerVar))

        #Assemble local constraints & costs        
        for i in range(Nturbine):
            [Const_i] = Turbine._Const.call([ V['Turbine',i], EP['Turbine',i]])    
            [Cost_i]  = Turbine._Cost.call( [ V['Turbine',i], EP['Turbine',i]])
            Const.append(entry('Turbine'+str(i), expr = Const_i))
        
            Cost += Cost_i
            
        g = struct_MX(Const)
        self.g      = g
        
        #Setup Central Solver
        Solver = _setSolver(self,V,Cost,g,EP)
        self.Solver = Solver
        
        self.lbV    = V(-inf)
        self.ubV    = V( inf)
        self.init   = V()
        
    def Embedding(self,Wact, time):

        #Embbed I.C.    
        for i in range(self.Nturbine):
            self.lbV['Turbine',i,'States',0] = self.EP['Turbine',i,'States0']
            self.ubV['Turbine',i,'States',0] = self.EP['Turbine',i,'States0']

        #Embbed wind profiles
        for i in range(self.Nturbine):
            self.lbV['Turbine', i,'Wind']  = Wact[i][time:]
            self.ubV['Turbine', i,'Wind']  = Wact[i][time:]
            self.init['Turbine',i,'Wind']  = Wact[i][time:]
  
    def Solve(self,Wact, time = 0):
        
        self.Embedding(Wact, time)
        
        self.Solver['solver'].setInput(self.init,                   'x0')
        self.Solver['solver'].setInput(self.Solver['lbg'],         "lbg")
        self.Solver['solver'].setInput(self.Solver['lbg'],         "ubg")
        self.Solver['solver'].setInput(self.lbV,                   "lbx")
        self.Solver['solver'].setInput(self.ubV,                   "ubx")
        self.Solver['solver'].setInput(self.EP,                      "p")
            
        self.Solver['solver'].solve()
        #print np.array(self.Solver['solver'].output('x'))
        
        Adjoints = self.g(np.array(self.Solver['solver'].output('lam_g')))
        Primal  = self.V(np.array(self.Solver['solver'].output('x')))
        
        return Primal, Adjoints
    
    
    
    def PrepareQPs(self, Primal,Adjoint):
            
        QPs = []
        
        solver = self._TurbineSolver
        for i in range(self.Nturbine):
            EP = self.EP['Turbine',i]
            V = Primal['Turbine',i]
            mu = Adjoint['Turbine'+str(i)]
            
            solver['H'].setInput(V, 0)
            solver['H'].setInput(1.,1)
            solver['H'].setInput(1.,2)
            solver['H'].setInput(mu,3)
            solver['H'].evaluate()
        
            solver['f'].setInput(V, 0)
            solver['f'].setInput(EP,1)
            solver['f'].evaluate()
            
            solver['dg'].setInput(V, 0)
            solver['dg'].setInput(EP,1)
            solver['dg'].evaluate()
            
            solver['g'].setInput(V, 0)
            solver['g'].setInput(EP,1)
            solver['g'].evaluate()
        
            QPs.append({
                        'H'  : DMatrix(solver[ 'H'].output()),
                        'f'  : DMatrix(solver[ 'f'].output()),
                        'dg' : DMatrix(solver['dg'].output()),
                        'g'  : DMatrix(solver[ 'g'].output()),
                        'lbX': DMatrix(self.lbV['Turbine',i] - V),
                        'ubX': DMatrix(self.ubV['Turbine',i] - V)
                        })
        return QPs
    
    
    def QPStep(self, QPs, Dual):
        ### Distributed solver ###
        ### All the operation performed within this method are meant to be local ###
        
        V = self._VTurbine
            
        Xall = []
        AdjointUpdate  = self.g()
        MuBoundall     = []
        ABextra_all    = []
        dPrimal        = []
        dAdjoint       = []
        dMuBound       = []
        
        Homotopy = {'Primal':{'Matrices': [], 'RHS':[]}, 'Dual':{'Matrices': [], 'RHS':[]}}
        
        Status = {'QPsolver' : [], 'Factorization': []}
        DualHess = -np.eye(self.Nshooting)/self.PowerSmoothingWeight
        for i in range(self.Nturbine):
            Hi   = DMatrix(QPs[i]['H'])
            fi   = DMatrix(QPs[i]['f'])
            dgi  = DMatrix(QPs[i]['dg'])
            gi   = DMatrix(QPs[i]['g'])
            lbXi = DMatrix(QPs[i]['lbX'])
            ubXi = DMatrix(QPs[i]['ubX'])
            
            #Index of the power variations
            IndexDual = list(V.i['PowerVar',veccat])
            
            #Dualize the coupling constraint
            f_dual = DMatrix(fi)
            for m, index in enumerate(IndexDual):
                f_dual[index] -= Dual[m]
            
            #Set solver

            self._QPsolver.setInput( Hi,   'h')
            self._QPsolver.setInput( f_dual,   'g')
            self._QPsolver.setInput( dgi,   'a')
            self._QPsolver.setInput( lbXi, 'lbx')
            self._QPsolver.setInput( ubXi, 'ubx')
            self._QPsolver.setInput( -gi, 'lba')
            self._QPsolver.setInput( -gi, 'uba')
    
            self._QPsolver.solve()
    
            #Status['QPsolver'].append(self._QPsolver.getStat('return_status'))
            
            X        = np.array(self._QPsolver.output('x'))
            Mudg     = np.array(self._QPsolver.output('lam_a'))
            MuBound  = self._QPsolver.output('lam_x')
            
            MuBoundall.append(MuBound)
            
            #Detect active/inactive bounds
            #(AB reports the variables that are EITHER on the upper OR on the lower bound)
            AB  = []
            IAB = []
            
            eps = 1e-8 #active constraint threshold
            
            #Variable to monitor collpased bounds
            BoundsGap = np.abs(np.array(ubXi) - np.array(lbXi))
            
            lb_gap = np.array(lbXi) - X
            ub_gap = X - np.array(ubXi)
            for ivar in range(X.shape[0]):
                if (lb_gap[ivar] >= -abs(MuBound[ivar])) or (ub_gap[ivar] >= -abs(MuBound[ivar])) or (BoundsGap[ivar] < eps):
                    AB.append(ivar)
                else:
                    IAB.append(ivar)

            #ABall.append(AB)
            

            
            #Construct dual homotopy (work on the AB)
            # sign(MuBound)*(MuBound + dMu) >= 0 hence  -sign(MuBound)*dMu =<  |MuBound|
            
            HomotopyDualMAT = []
            HomotopyDualRHS = []
            ABextra = []
            for m, ivar in enumerate(AB):
                HomotopyDualMAT.append(np.sign(MuBound[ivar]))
                if (BoundsGap[ivar] > eps):
                    ABextra.append(AB[m])
                    HomotopyDualRHS.append(abs(MuBound[ivar]))
                else:
                    HomotopyDualRHS.append(inf) #Do not check a multiplier associated to a collpased constraint
            ABextra_all.append(ABextra)
                       
            HomotopyDualMAT = np.diag(HomotopyDualMAT)
            HomotopyDualRHS = np.array(HomotopyDualRHS).reshape(len(HomotopyDualRHS),1)
                
            #Construct primal homotopy 
            #       lbXi <= X + dX <= ubXi
            # i.e.  lbXi - X <= dX <= ubXi - X
            # or
            # 1. dX <= -ub_gap and 2. -dX <= -lb_gap  constructed seperately 
            #
            # i.e. check
            #
            # [ I] * dX <= -[ub_gap]
            # [-I]          [lb_gap]
        
            HomotopyPrimalMAT = np.concatenate([np.eye(X.shape[0]),-np.eye(X.shape[0])],axis=0) 
            RHSub = []
            RHSlb = []
            for ivar in range(X.shape[0]): 
                if (BoundsGap[ivar] > eps) and (ub_gap[ivar] < -abs(MuBound[ivar])):
                    RHSub.append(-ub_gap[ivar]) #Check only inactive bounds
                else:
                    RHSub.append(inf) #Do not check primal variables blocked by collapsed constraints

                if (BoundsGap[ivar] > eps) and (lb_gap[ivar] < -abs(MuBound[ivar])):
                    RHSlb.append(-lb_gap[ivar]) #Check only inactive bounds
                else:
                    RHSlb.append(inf) #Do not check primal variables blocked by collapsed constraints
            
            HomotopyPrimalRHS = np.concatenate([RHSub,RHSlb],axis = 0)
            HomotopyPrimalRHS = HomotopyPrimalRHS.reshape(HomotopyPrimalRHS.shape[0],1)
            
            #The homotopy must check:
            # HomotopyPrimalMAT * dX       <= HomotopyPrimalRHS
            # HomotopyDualMAT   * dMuBound <= HomotopyDualRHS
            
            #Block the active constraints
            # gActive = [Active bounds
            #                dg       ], i.e. (1st simple bounds, 2nd A*X = b)
            
            block = np.zeros([len(AB),X.shape[0]])
            for line, col in enumerate(AB):
                block[line,col] = 1.
            gActive = np.concatenate([block,QPs[i]['dg']],axis = 0)
            
            # KKT Matrix
            KKTMat = np.concatenate([QPs[i]['H'],gActive],axis = 0)
            Nconst = gActive.shape[0]
            Addon = np.concatenate([gActive.T,np.zeros([Nconst,Nconst])])
            KKT = np.concatenate([KKTMat,Addon],axis = 1)
    
            #Right-hand side
            b = np.zeros([KKT.shape[0],len(IndexDual)])
            for col, line in enumerate(IndexDual):
                b[line,col] = 1.
            
            #dPrimalAdjoint = KKT\b
            try:
                dPrimalAdjoint = linalg.solve(KKT,b)
            except: #In case of a badly identified Active Set
                Status['Factorization'].append('RankDefficienty, Turbine'+str(i))
                dPrimalAdjoint, _, _, _ = linalg.lstsq(KKT,b)
                
            #The "dPrimalAdjoint" provides:
            #[d X       ]
            #[d MuBound ] = dPrimalAdjoint * d Dual
            #[d Mudg    ]
            
            # Homotopy must check:
            #
            #   HomotopyPrimalMAT*dPrimalAdjoint <= HomotopyPrimalRHS
            #   HomotopyDualMAT  *dPrimalAdjoint <= HomotopyDualRHS
            #
            # Hence
            #
            #   Homotopy['Primal']['Matrices']*DualStep <= Homotopy['Primal']['RHS']
            #   Homotopy[  'Dual']['Matrices']*DualStep <= Homotopy[  'Dual']['RHS']
            #
            
            dPrimal.append(dPrimalAdjoint[:X.shape[0],:])
            dMuBound.append(dPrimalAdjoint[X.shape[0]:X.shape[0]+len(AB),:])
            dAdjoint.append(dPrimalAdjoint[X.shape[0]+len(AB):,:])
            
            Homotopy['Primal']['Matrices'].append(np.dot(HomotopyPrimalMAT,dPrimal[-1]))
            Homotopy['Primal'][     'RHS'].append(HomotopyPrimalRHS)
            
            Homotopy[  'Dual']['Matrices'].append(np.dot(HomotopyDualMAT,dMuBound[-1]))
            Homotopy[  'Dual'][     'RHS'].append(HomotopyDualRHS)
            
            DualHess -= dPrimalAdjoint[IndexDual]
            
            Xall.append(V(X))
    
            AdjointUpdate['Turbine'+str(i)] = np.array(self._QPsolver.output('lam_a'))
        return Xall, AdjointUpdate, DualHess, dPrimal, dAdjoint, Homotopy, Status

    def DistributedSQP(self, Primal, Adjoint,Dual, Wact, time = 0, iter_SQP = 1,iter_Dual = 1, FullDualStep = True, ReUpdate = False):
        
        self.Embedding(Wact, time)
        
        #LocalResidual = 10
        #while (LocalResidual > 1e-6):
        for iterate in range(iter_SQP):
            
            #Construct local QPs (without dualization)
            QPs = self.PrepareQPs(Primal,Adjoint)
            
            LocalResidual = 0
            for i in range(self.Nturbine):
                LocalResidual += np.sqrt(np.dot(QPs[i]['g'].T,QPs[i]['g']))
                
            #Dual decomposition iteration
            Norm_Dual = []
            CondHess = []
            StatusLog = []
            for iter_dual in range(iter_Dual):
            
                
                #Local Primal/Adjoint Step
                StepLocal, Adjoint, DualHess, dPrimal, dAdjoint, Homotopy, Status = self.QPStep(QPs, Dual)
                StatusLog.append(Status)
                
                #Check Dual Hessian conditioning
                #u, s, vh = linalg.svd(DualHess)
                #CondHess.append(s[0]/s[-1])
                
                #z step
                Stepz = self.EP['PowerVarRef',veccat]-Primal['PowerVar',veccat] - Dual/float(self.PowerSmoothingWeight)
                
                #Constraints residual
                Residual = Primal['PowerVar',veccat] + Stepz
                for i in range(self.Nturbine):
                    Residual -= Primal['Turbine',i,'PowerVar',veccat] + StepLocal[i]['PowerVar',veccat]
                    
                #Dual full step
                StepDual = linalg.solve(DualHess,Residual)
                                        
                NormStepDual = np.sqrt(mul(StepDual.T,StepDual))
                NormResidual = sqrt(mul(Residual.T,Residual))
                
                ################### DUAL STEP SIZE ####################
                
                if (FullDualStep == False):
                    tmax = {'Primal':1.,'Dual':1.}
                    eps = 1e-10
                    for key in tmax.keys():
                        for i in range(self.Nturbine):
                            a = np.dot(Homotopy[key]['Matrices'][i],StepDual)
                            b = Homotopy[key]['RHS'][i]
                            for index in range(a.shape[0]):
                                if (a[index] > eps) and (a[index]-b[index] > eps) and (b[index] > 1e-1):
                                    tmaxnew = float(b[index]/a[index])
                                    if (tmaxnew < tmax[key]):
                                        tmax[key] = tmaxnew
                    
                    tstep = np.min([1.,tmax['Primal'],tmax['Dual']])
                else:
                    tstep = 1.
                
                #######################################################
        
                headerstr = "Iter \t Dual Residual \t Dual step-size \t  Dual full step norm \t Local Residuals"
                print headerstr
                print "%3d \t %.5E \t %.5E \t %.5E \t %.5E" %  (iter_dual, NormResidual,  tstep,  NormStepDual, LocalResidual)
                
                Dual -= tstep*StepDual
                          
                Norm_Dual.append(NormResidual)
                
            ################### SECOND UPDATE ####################
                
            if (ReUpdate == True):
                for i in range(self.Nturbine):
                    StepLocal[i] = self._VTurbine(StepLocal[i].cat + np.dot(dPrimal[i],-tstep*StepDual))
                    Adjoint['Turbine'+str(i)] += np.dot(dAdjoint[i],-tstep*StepDual)
                
                Stepz = self.EP['PowerVarRef',veccat]-Primal['PowerVar',veccat] - Dual/float(self.PowerSmoothingWeight)
                
            #######################################################
                
                
        
            #plt.figure(200)
            #plt.semilogy(Norm_Dual[1:])
            #plt.hold('on')
            #plt.grid()
            #plt.title('2 norm of the (dualized) constraint residual')
            #plt.show()
            ##raw_input()
            #plt.close()
            
      
            #Update primal variables
            for index in range(len(Primal['PowerVar'])):
                Primal['PowerVar',index] += Stepz[index]      
            
            for i in range(self.Nturbine):
                for key in StepLocal[i].keys():
                    Primal['Turbine',i,key,veccat] += StepLocal[i][key,veccat]
                    
                    #Project in the bounds 
                    Primal['Turbine',i,key,veccat] = min(max(Primal['Turbine',i,key,veccat],self.lbV['Turbine',i,key,veccat]),self.ubV['Turbine',i,key,veccat])
            
            #Pass on Dual variables (not needed, but for consistency)
            Adjoint['PowerConst',veccat] = Dual
            
            #Update the residual (needed if second update is used)
            ResidualOut = Primal['PowerVar',veccat]
            for i in range(self.Nturbine):
                ResidualOut -= Primal['Turbine',i,'PowerVar',veccat]
            
            
            
        return Primal, Adjoint, Dual, ResidualOut, tstep, StatusLog
    
    ############ Simulation ########
    def Simulate(self,W):

        StatesPlus = []
        for i in range(self.Nturbine):
            
            self._Shoot.setInput(self.EP['Turbine',i,'States0'],0)
            self._Shoot.setInput(self.EP['Turbine',i,'Inputs0'],1)
            self._Shoot.setInput(W[i],2)
            self._Shoot.evaluate()
            Xplus = np.array(self._Shoot.output())
            
            #self.EP['Turbine',i,'States0'] = Xplus
            StatesPlus.append(Xplus)
            
        return StatesPlus

    ############ Shifting ##########
    def Shift(self, Primal, Adjoint, Dual):
                
        PrimalShifted = self.V()
        for i in range(self.Nturbine):
            PrimalShifted['Turbine',i,...,:-1] = Primal['Turbine',i,...,1:]
            PrimalShifted['Turbine',i,..., -1] = Primal['Turbine',i,...,-1]
        
        PrimalShifted['PowerVar',:-1] = Primal['PowerVar',1:]
        PrimalShifted['PowerVar',-1]  = Primal['PowerVar',-1]
        
        AdjointShifted = self.g()
        for key in Adjoint.keys():
            AdjointShifted[key,:-1] = Adjoint[key,1:]
            AdjointShifted[key, -1] = Adjoint[key,-1]

        
        DualShifted = np.zeros(Dual.shape)  
        DualShifted[:-1,:] = Dual[1:,:]
        DualShifted[ -1,:] = Dual[-1,:]
        
        return PrimalShifted, AdjointShifted, DualShifted

         
    ############ Plotting ##########
    def PlotStep(self, Primal,Primal0,col,dt = 0.2):
        
        Nshooting = self.Nshooting
        Nturbine = self.Nturbine
        
        time = {'States': [dt*k for k in range(Nshooting+1)],
                'Inputs': [dt*k for k in range(Nshooting)]}
    
        Nsubp = np.ceil(np.sqrt(Nturbine))
        key_dic = {'States': ['Og','beta'], 'Inputs': ['Tg']}
        counter = 0
        for type_ in key_dic.keys():
            for key in key_dic[type_]:
                for k in range(Nturbine):
                    Primal_struc  = self._VTurbine(Primal['Turbine'][k])
                    Primal0_struc = self._VTurbine(Primal0['Turbine'][k])
                    diff   = veccat(Primal_struc[type_,:,key])-veccat(Primal0_struc[type_,:,key])
               
                    plt.figure(10+counter)
                    plt.subplot(Nsubp,Nsubp,k)
                    plt.hold('on')
                    if (type_ == 'States'):
                        plt.plot(time[type_],diff,color=col)
                    else:
                        plt.step(time[type_],diff,color=col)    
            
                plt.title('Step'+key)   
                counter += 1
    
    def PlotBasic(self, T, Primal, time, col):
        for fig, key in enumerate(T.States.keys()):
            for i in range(self.Nturbine):
                plt.figure(fig)
                plt.subplot(self.Nturbine,1,i)
                plt.hold('on')
                plt.plot(time['Inputs'],    Primal['Turbine',i,'States',:-1,key],color=col)
            plt.title('State '+key)
            
        for fig, key in enumerate(T.Inputs.keys()):
            for i in range(self.Nturbine):
                plt.figure(100+fig)
                plt.subplot(self.Nturbine,1,i)
                plt.hold('on')
                plt.step(time['Inputs'],    Primal['Turbine',i,'Inputs',:,key],color=col)
            plt.title('Input '+key)
    
        plt.figure(200)

        PowerVar = 0
        for i in range(self.Nturbine):
            PowerVar += np.array(Primal['Turbine',i,'PowerVar'])
            plt.hold('on')
            plt.plot(time['Inputs'],Primal['Turbine',i,'PowerVar'],color=col,linestyle = ':')

        plt.plot(time['Inputs'],    PowerVar,color=col,linewidth = 2)

    def Plot(self, Turbine, Primal0, dt = 0.2):
        
        Nturbine = self.Nturbine
        Nsubp = np.ceil(np.sqrt(Nturbine))
        Nshooting = self.Nshooting
        
        time = {'States': [dt*k for k in range(Nshooting+1)],
                'Inputs': [dt*k for k in range(Nshooting)]}
            
        
        Dict = {'States':Turbine.States, 'Inputs':Turbine.Inputs}
        fig = 0
        for typekey in Dict.keys():
            for key in Dict[typekey].keys():
                for k in range(Nturbine):
                    plt.figure(fig)
                    plt.subplot(Nsubp,Nsubp,k)
                    plt.hold('on')
                    plt.plot(time[typekey],veccat(Primal0['Turbine',k,typekey,:,key]),color='k')
                    plt.title(key)
                fig += 1
                
        PgTot = 0
        for k in range(Nturbine):  
            Pg = np.array(Primal0['Turbine',k,'PowerVar'])
            PgTot += Pg
            
            plt.figure(100)
            plt.subplot(Nsubp,Nsubp,k)
            plt.plot(time['Inputs'],Pg)
            plt.title('Power Variation')
        
            plt.figure(101)
            plt.subplot(Nsubp,Nsubp,k)
            plt.step(time['States'],Primal0['Turbine',k,'Wind'],color = 'k')
            plt.title('Wind')
                    
        plt.figure(6)
        plt.hold('on')
        plt.plot(time['Inputs'],PgTot)
        plt.plot(time['Inputs'],self.EP['PowerVarRef'])
        plt.title('Total Power (av. ov)')
        plt.ylim([20,-20])
        plt.show() 
