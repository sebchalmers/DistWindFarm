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
A Fast Algorithm for Power Smoothing of Wind Farms based on Distributed Optimal Control

Requires the installation of the open-source Python module casADi together with the NLP solver ipopt

Required version of CasADi: v1.7.x
 
"""

from casadi import *
from casadi.tools import *
#import math as math

import numpy as np
import scipy.special as sc_spec
from scipy import linalg
import scipy.io

import cplexinterface
import scipy as sc

import matplotlib.pyplot as plt
from pylab import matshow

def assertList(var):
    if not(isinstance(var,list)):
        var = [var]
    return var

def RegularizeMatrix(A):
    minEigval = 0
    n = np.size(A,axis=1)
    for i in range(n):
        tempCentre = A[i,i]
        tempRadius = 0            
        for j in range(n):
            if i != j:
                tempRadius += abs(A[i,j])
            
        if tempCentre - tempRadius < minEigval:
            minEigval = tempCentre - tempRadius
    print 'Min eigenvalue:', minEigval
    if minEigval < 0:
        RegularizedA = A + np.eye(n)*np.abs(minEigval)
    else:
        RegularizedA = A
    return RegularizedA

#Solver Constructor (used in both Turbine and WindFarm class)
def _setSolver(self, V,Cost,g,P):

    lbg = g()
    ubg = g()
    if ('IneqConst' in g.keys()):
        print "Turbine Solver"
        lbg['IneqConst'] = -1e20
        ubg['IneqConst'] =  0
        EquConst         = list(g.i['EquConst'])
        IneqConst        = list(g.i['IneqConst'])
        
    if hasattr(self,'Nturbine'):
        print "Wind Farm Solver"
        EquConst  = []
        IneqConst = []
        key_list = ['Turbine'+str(i)+'_IneqConst' for i in range(self.Nturbine)]
        for key in g.keys():
            if (key in key_list):
                lbg[key] = -1e20
                ubg[key] =  0
                IneqConst.append(list(g.i[key_list]))
            else:
                EquConst.append(list(g.i[key_list]))
                
    
    nl = MXFunction(nlpIn(x=V,p=P),nlpOut(f=Cost,g=g))
    nl.init()
    
    # set-up solver
    solver = IpoptSolver(nl)  
    solver.setOption("expand",True)
    #solver.setOption("print_level",1)  
    solver.setOption("hessian_approximation","exact")
    solver.setOption("max_iter",2000)
    solver.setOption("tol",1e-8)
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
    
    Solver = {'solver': solver, 'H': H, 'f': f, 'dg': dg, 'g': gfunc, 'lbg': lbg, 'ubg': ubg, 'V': V, 'EquConst':EquConst,  'IneqConst':IneqConst}
    
    return Solver

def _CreateQP(solver,V):
    
    solver['H'].evaluate()
    H = solver['H'].output() + np.eye(solver['H'].output().shape[0])
    
    solver['dg'].evaluate()
    dg = solver['dg'].output()
    
    QPsolver = NLPQPSolver(qpStruct(h=H.sparsity(),a=dg.sparsity()))
    QPsolver.setOption({"nlp_solver":IpoptSolver, "nlp_solver_options": {"tol": 1e-8,"verbose": 0} })
   
    QPsolver.init()

    return QPsolver

class Turbine:
    def __init__(self, Inputs = [], States = [], Slacks = []):        

        self._frozen = False
        
        #Inputs declared by the user
        InputList = []
        for key in assertList(Inputs):
            InputList.append(entry(key))
            
        # States declared by the user
        StateList = []
        for key in assertList(States):
            StateList.append(entry(key))
            
        # Slacks declared by the user (for constraints relaxation)
        SlackList = []
        for key in assertList(Slacks):
            SlackList.append(entry(key)) 
        
        P = ssym('P')
        W = ssym('W')

        # lists of names (strings)
        self.States            = struct_ssym(StateList)
        self.Inputs            = struct_ssym(InputList)
        self.StatesPrev        = struct_ssym(StateList)
        self.InputsPrev        = struct_ssym(InputList)
        self.Wind              = W
        self.PowerVar          = P
            
        if (len(SlackList) > 0):
            self.Slacks        = struct_ssym(SlackList)
    
    def ElecPower(self, expr):
        Power = SXFunction([self.States,self.Inputs],[expr])
        Power.init()
        
        self.ElecPower = Power
        
    def setDynamics(self, RHS = [], dt = .2, nstep = 10):
        if (self._frozen == True):
            print "Plant already added to the grid, call ignored"
            return

        
        #print "Right-Hand side: ", RHS
        
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
            
        if hasattr(self,'Slacks'):
            Slacks      = self.Slacks
            
        #CAREFUL, THE INPUT SCHEME MUST MATCH THE FUNCTION CALLS !!
        if Terminal == False:
            listFuncInput = [U, X, W, PowerVar, Uprev, Xprev]            
        else:
            listFuncInput = [X, W, Xprev]
        
        if hasattr(self,'Slacks'):    
            listFuncInput.append(Slacks)
            
        Func = SXFunction(listFuncInput,[Expr])
        Func.init()
        
        return Func

    def setIneqConst(self, Const, Terminal = False, Start = 1):
        
        if (self._frozen == True):
            print "Wind farm already created, call ignored"
            return
        
        if not(isinstance(Const,list)):
            Const = [Const]
        
        #ConstFunc = self._BuildFunc(veccat(Const), Terminal)
                
        if    (Terminal == False):
            self._StageConst    = self._BuildFunc(veccat(Const), Terminal)
            self._StartStageConst = Start
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
        
        Variables = [
                        entry('States',   struct = self.States,   repeat = Nelements+1),
                        entry('Inputs',   struct = self.Inputs,   repeat = Nelements),
                        entry('Wind',     struct = self.Wind,     repeat = Nelements+1),
                        entry('PowerVar', struct = self.PowerVar, repeat = Nelements)
                    ]
        
        if hasattr(self,'Slacks'):
            Variables.append(entry('Slacks',   struct = self.Slacks,   repeat = Nelements+1))
            
        Variables  = struct_msym(Variables)
        
        return Variables
    
    def setTurbine(self, Nshooting = 50, Nsimulation = 50 ):
        
        if hasattr(self,'_TerminalConst') or hasattr(self,'_StageConst'):
            self._hasIneqConst = True
        else:
            self._hasIneqConst = False    
            
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
        PowerConst = []
        DynConst   = []
        StageConst = []
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
            
            InputList = [ 
                                self.V['Inputs',k],
                                self.V['States',k],                                
                                self.V['Wind',k],
                                self.V['PowerVar',k]
                        ]
            
            if (k == 0):
                InputList.append(self.EP['Inputs0'])
                InputList.append(self.EP['States0'])
                
            else:
                InputList.append(self.V['Inputs',k-1])
                InputList.append(self.V['States',k-1])
                              
         
            
            if hasattr(self,'Slacks'):
                InputList.append(self.V['Slacks',k])
                
            #Stage cost
            [StageCost]  = self._StageCost.call(InputList)
            Cost += StageCost
        
            #Stage Inequality constraints
            if hasattr(self,'_StageConst') and (k > self._StartStageConst):
                [StageConst_k] = self._StageConst.call(InputList)
                StageConst.append(StageConst_k)

        #Terminal stuff
        k = Nshooting
        InputList = [
                        self.V['States',k],
                        self.V['Wind',k],
                        self.V['States',k-1]
                    ]
        
        if hasattr(self,'Slacks'):
            InputList.append(self.V['Slacks',k])
        
        if hasattr(self,'_TerminalCost'):
            [TerminalCost] = self._TerminalCost.call(InputList)
            Cost += TerminalCost
          
        #Stage Inequality constraints
        if hasattr(self,'_TerminalConst'):
            [TermConst] = self._TerminalConst.call(InputList)

        
        self._Functions = {}
            
        #Construct Constraints Structure
        EquConst = [
                    entry('DynConst',    expr = DynConst),
                    entry('PowerConst',  expr = PowerConst)
                   ]
        
        
        if self._hasIneqConst:           
            IneqConstEntry = [
                              entry('StageConst', expr = StageConst),
                              entry( 'TermConst', expr =  TermConst)
                             ]
        
        
        
        ##Create EquConst function 
        EquConstFun = MXFunction([self.V,self.EP],[struct_MX(EquConst)])
        EquConstFun.init()
        self._Functions['EquConst'] = EquConstFun
          
        ##Create Cost function
        CostFun  = MXFunction([self.V,self.EP],[Cost])
        CostFun.init()
        self._Functions['Cost'] = CostFun
        
        #Create IneqConst function (if applicable)
        if self._hasIneqConst:
            IneqConstFun = MXFunction([self.V,self.EP],[struct_MX(IneqConstEntry)])
            IneqConstFun.init()
            self._Functions['IneqConst'] = IneqConstFun
        
        
        #Create QP solver having the same (lcoal) structure as the central problem          
        [     Cost] = self._Functions[     'Cost'].call([ self.V, self.EP ])
        [ EquConst] = self._Functions[ 'EquConst'].call([ self.V, self.EP ])
        
        Const = [
                    entry('EquConst',   expr = EquConst)
                ]
        
        if self._hasIneqConst:
            [IneqConst] = self._Functions['IneqConst'].call([self.V, self.EP ])
            Const.append(entry('IneqConst', expr = IneqConst))

        gLocal = struct_MX(Const)
        
        self._g = gLocal
                
        #Create a local solver to generate the underlying QP
        Solver = _setSolver(self,self.V,Cost,gLocal,self.EP)
        self._Solver = Solver
        
        #Prepare the QP call
        self._QPsolver = _CreateQP(Solver,self.V)
        

     
class WindFarm:
    
    def __init__(self, Turbine, Nturbine = 0):     
        
        
        self.Nturbine   = Nturbine
        Nshooting       = Turbine.Nshooting
        Nsimulation     = Turbine.Nsimulation

        self.Nshooting            = Nshooting
         
        #Carry over local stuff
        self.Nsimulation      = Nsimulation
        self._TurbineSolver   = Turbine._Solver
        self._TurbineQPsolver = Turbine._QPsolver
        self._TurbineV        = Turbine.V
        self._Shoot           = Turbine.Shoot
        self._gLocal          = Turbine._g
        
        #Do the wind turbines have inequality constraints ?
        if hasattr(Turbine,'_TerminalConst') or hasattr(Turbine,'_StageConst'):
            self._hasIneqConst = True
        else:
            self._hasIneqConst = False
            
        # Container for the global decision variables
        V          =       struct_msym([
                                            entry('Turbine',   struct = Turbine.V,              repeat = Nturbine),
                                            entry('PowerVar',  struct = Turbine.PowerVar,       repeat = Nshooting)
                                        ])

    
        self.V = V
        
        PowerSmoothingWeight = ssym('PowerSmoothingWeight')
        
        EP          =       struct_msym([
                                            entry('Turbine',              struct = Turbine.EP,             repeat = Nturbine  ), 
                                            entry('PowerVarRef',          struct = Turbine.PowerVar,       repeat = Nshooting ),
                                            entry('PowerSmoothingWeight', struct = PowerSmoothingWeight)
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
        IneqConst = []
        
        #Wind farm total power variation & total power constraints
        for k in range(Nshooting):
            TotPowerVar_k = V['PowerVar', k]
            for i in range(Nturbine):            
                TotPowerVar_k -= V['Turbine',i,'PowerVar',k]
                
            TotPowerVar.append(TotPowerVar_k)
            Cost += 0.5*EP['PowerSmoothingWeight']*(EP['PowerVarRef',k]-V['PowerVar', k])**2
            
        Const.append(entry('PowerConst',    expr = TotPowerVar))

        
        #Assemble local constraints & costs        
        for i in range(Nturbine):
            [     Cost_i] = Turbine._Functions[     'Cost'].call([ V['Turbine',i], EP['Turbine',i] ])
            [ EquConst_i] = Turbine._Functions[ 'EquConst'].call([ V['Turbine',i], EP['Turbine',i] ])
            
            Cost += Cost_i
            Const.append(entry('Turbine'+str(i)+'_EquConst', expr = EquConst_i))
            
            if self._hasIneqConst:
                [IneqConst_i] = Turbine._Functions['IneqConst'].call([ V['Turbine',i], EP['Turbine',i] ])
                Const.append(entry('Turbine'+str(i)+'_IneqConst', expr = IneqConst_i))
            
        g = struct_MX(Const)
        self.g      = g
        
        #Setup Central Solver
        Solver = _setSolver(self,V,Cost,g,EP)
        self.Solver = Solver
        
        ##Prepare the QP call
        #self._QPsolver = _CreateQP(Solver,self.V)
        
        self.lbV    = V(-1e20)
        self.ubV    = V( 1e20)

        if hasattr(Turbine,'Slacks'):
            self.lbV['Turbine',:,'Slacks'] = 0
        
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
        self.Solver['solver'].setInput(self.Solver['ubg'],         "ubg")
        self.Solver['solver'].setInput(self.lbV,                   "lbx")
        self.Solver['solver'].setInput(self.ubV,                   "ubx")
        self.Solver['solver'].setInput(self.EP,                      "p")
            
        self.Solver['solver'].solve()
        
        self._lbg = self.Solver['lbg']
        self._ubg = self.Solver['ubg']
        
        Adjoints = self.g(np.array(self.Solver['solver'].output('lam_g')))
        Primal   = self.V(np.array(self.Solver['solver'].output('x')))
        
        return Primal, Adjoints
    
    
    
    def PrepareQPs(self, Primal,Adjoint):
            
        QPs  = []
        Rlog = [] #Log regularization
        
        solver = self._TurbineSolver
        for i in range(self.Nturbine):
            EP = self.EP['Turbine',i]
            V = Primal['Turbine',i]
            
            Mu = self._gLocal()
            Mu['EquConst']      = Adjoint['Turbine'+str(i)+'_EquConst']
            if self._hasIneqConst:
                Mu['IneqConst'] = Adjoint['Turbine'+str(i)+'_IneqConst']
                        
            solver['H'].setInput(V, 0)
            solver['H'].setInput(1.,1)
            solver['H'].setInput(1.,2)
            solver['H'].setInput(Mu,3)
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
        
            H = DMatrix(solver[ 'H'].output())
            
            ######  Regularization Brute Force ######
            ##R = 0
            #Eig, D  = linalg.eig(H)                
            #R = np.min(np.real(Eig))-1e-4
            #Rlog.append(R)
            #
            #IndexReg = range(H.shape[0])
            #Reg = [0 for index in range(H.shape[0])]
            #for index in IndexReg:
            #    Reg[index] = 1.
            #
            #HReg     = H - R*np.diag(Reg)
            ########################################
            
            ######  Regularization Gerschgorin ######
            self.H = H
            HReg     = RegularizeMatrix(np.array(H))
            
                    
            QPs.append({
                        'H'  : HReg,
                        'f'  : DMatrix(solver[ 'f'].output()),
                        'dg' : DMatrix(solver['dg'].output()),
                        'g'  : DMatrix(solver[ 'g'].output()),
                        'lbX': DMatrix(self.lbV['Turbine',i] - V),
                        'ubX': DMatrix(self.ubV['Turbine',i] - V),
                        'lbg': DMatrix(solver['lbg'] - solver['g'].output()),
                        'ubg': DMatrix(solver['ubg'] - solver['g'].output()),
                   'EquConst': solver[ 'EquConst'],
                  'IneqConst': solver[ 'IneqConst']
                        })
            
        return QPs, Rlog
    
    
    def PrepareCentralQP(self, solver, Primal, Adjoint):
            
        EP = self.EP
        
        solver['H'].setInput(Primal, 0)
        solver['H'].setInput(1.,1)
        solver['H'].setInput(1.,2)
        solver['H'].setInput(Adjoint,3)
        solver['H'].evaluate()
    
        solver['f'].setInput(Primal, 0)
        solver['f'].setInput(EP,1)
        solver['f'].evaluate()
        
        solver['dg'].setInput(Primal, 0)
        solver['dg'].setInput(EP,1)
        solver['dg'].evaluate()
        
        solver['g'].setInput(Primal, 0)
        solver['g'].setInput(EP,1)
        solver['g'].evaluate()
    
        QP =       {
                    'H'  : DMatrix(solver[ 'H'].output()),
                    'f'  : DMatrix(solver[ 'f'].output()),
                    'dg' : DMatrix(solver['dg'].output()),
                    'g'  : DMatrix(solver[ 'g'].output()),
                    'lbX': DMatrix(self.lbV.cat - Primal.cat),
                    'ubX': DMatrix(self.ubV.cat - Primal.cat),
                    'lbg': DMatrix(solver['lbg'].cat - solver['g'].output()),
                    'ubg': DMatrix(solver['ubg'].cat - solver['g'].output())
                    }
        
        return QP
    
    def solveQP(self, Solver, QP, fadd = 0):
        
        H   = DMatrix(QP['H'])
        f   = DMatrix(QP['f'])
        dg  = DMatrix(QP['dg'])
        g   = DMatrix(QP['g'])
        lbX = DMatrix(QP['lbX'])
        ubX = DMatrix(QP['ubX'])
        
        lbg = DMatrix(QP['lbg'])
        ubg = DMatrix(QP['ubg'])
        
        HCPLEX   = sc.sparse.csc_matrix(H)
        fCPLEX   = np.array( f + fadd )
        lbXCPLEX = list( lbX )
        ubXCPLEX = list( ubX )
        
        Aeq      = np.array(  dg[QP['EquConst'] ,:] )
        beq      = np.array( lbg[QP['EquConst'] ,:] )
        
        Aineq    = np.array(  dg[QP['IneqConst'],:] )
        lbineq   = np.array( lbg[QP['IneqConst'],:] )
        ubineq   = np.array( ubg[QP['IneqConst'],:] )
                
        AeqCPLEX   = sc.sparse.csc_matrix(   Aeq )
        AineqCPLEX = sc.sparse.csc_matrix( Aineq )
                

        #Set solver
        CPS = cplexinterface.CplexQPSolver()
        [X, fopt, MuIneq, MuEqu,  MuBound, sol_status, sol_string] = CPS.cplex_solve(HCPLEX, fCPLEX, Aineq, ubineq, AeqCPLEX, beq, lbXCPLEX, ubXCPLEX)
        
        
        #Set solver

        #Solver.setInput( H,      'h')
        #Solver.setInput( f+fadd,  'g')
        #Solver.setInput( dg,     'a')
        #Solver.setInput( lbX,    'lbx')
        #Solver.setInput( ubX,    'ubx')
        #Solver.setInput( lbg,    'lba')
        #Solver.setInput( ubg,    'uba')
        #
        #Solver.solve()
        #
        #X   = np.array(Solver.output('x'))[:f.shape[0]]
        
        ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVE WHEN ACTIVATING CPLEX !!!!!!!!!!!!!!!!!!!!!!!!!!!
        #MuIneq  = list(np.array(Solver.output('lam_a'))[QP['IneqConst']])
        #MuEqu   = list(np.array(Solver.output('lam_a'))[QP['EquConst'] ])
        #MuBound = list(Solver.output('lam_x'))
        #Xcplex  = X
        ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        #Error = X-np.array(Xcplex).reshape(X.shape[0],1)
        #self.X = X
        #self.Xcplex = Xcplex
        #
        #
        #Error = np.sqrt(np.dot(Error.T,Error))
        #
        #if (Error > 1e0):
        #
        #    plt.figure(99)
        #    plt.hold('on')
        #    plt.plot(X,     color = 'k', linewidth = 2,label='ip sol')
        #
        #    
        #    plt.plot(Xcplex,color = 'r',label='A-S sol')
        #    plt.legend()
        #
        #    raw_input()
    
        return np.array(X).reshape(len(X),1), MuEqu, MuIneq, MuBound
    
    
    def QPStep(self, QPs, Dual):
        ### Distributed solver ###
        ### All the operation performed within this method are meant to be local ###
        
        V = self._TurbineV
            
        Xall = []
        AdjointUpdate  = self.g()
        
        dPrimal        = []
        dAdjoint       = []
         
        dMu            = []
        dMuBound       = []
        
        MuLog          = []
        ALog           = {'AS': [], 'AB': [], 'Ag': []}
        GapLog         = []
        
        Homotopy = {'Primal':{'Matrices': [], 'RHS':[]}, 'Dual':{'Matrices': [], 'RHS':[]}}
        
        Status = {'QPsolver' : [], 'Factorization': []}
        if (self.EP['PowerSmoothingWeight'] > 0):
            DualHess = -np.eye(self.Nshooting)/self.EP['PowerSmoothingWeight']
        else:
            #To allow simulations with no smoothing
            DualHess = -np.eye(self.Nshooting)
            
        for i in range(self.Nturbine):

            #Index of the power variations
            IndexDual = list(V.i['PowerVar',veccat])
            
            #Dualize the coupling constraint
            fadd = np.zeros(QPs[i]['f'].shape)
            for m, index in enumerate(IndexDual):
                fadd[index] -= Dual[m]
               

            Xcplex, MuEqucplex, MuIneqcplex, MuBoundcplex = self.solveQP(self._TurbineQPsolver, QPs[i], fadd)
            
            #X2        = np.array(self._TurbineQPsolver.output('x'))
            #Mug2      = list(self._TurbineQPsolver.output('lam_a'))
            #MuBound2  = list(self._TurbineQPsolver.output('lam_x'))
            
            #self.Xcplex    = Xcplex
            #self.X2        = X2
            #self.MuEqu    = MuEqucplex
            #self.MuIneq   = MuIneqcplex
            #self.MuBound2 = MuBoundcplex
            #self.QP       = QPs[i]
            
            X                        = Xcplex
            Mug                      = np.zeros([len(QPs[i]['EquConst'] + QPs[i]['IneqConst'])])
            Mug[QPs[i]['EquConst']]  = -np.array(MuEqucplex)
            Mug[QPs[i]['IneqConst']] = -np.array(MuIneqcplex)
            Mug                      = list(Mug)
            MuBound                  = [-mubound for mubound in MuBoundcplex]
            
            #self.Mug = Mug
            #self.MuBound = MuBound
            #self.Mug2 = Mug2
            #self.MuBound2 = MuBound2
            #assert(0==1)
            #plt.figure(333)
            #plt.hold('on')
            #plt.subplot(4,1,1)
            #plt.plot(X,linestyle = 'none',marker='.',color = 'k')
            #plt.plot(X2,linestyle = 'none',marker='.',color = 'r')
            #plt.subplot(4,1,2)
            #plt.plot(-np.array(MuBound),linestyle = 'none',marker='.',color = 'k')
            #plt.plot(MuBound2,linestyle = 'none',marker='.',color = 'r')
            #plt.subplot(4,1,3)
            #plt.plot(-np.array(Mug)[QPs[i]['EquConst']],linestyle = 'none',marker='.',color = 'k')
            #plt.plot(MuEqu,linestyle = 'none',marker='.',color = 'r')
            #plt.subplot(4,1,4)
            #plt.plot(-np.array(Mug)[QPs[i]['IneqConst']],linestyle = 'none',marker='.',color = 'k')
            #plt.plot(MuIneq,linestyle = 'none',marker='.',color = 'r')
            #

            
            #Detect active/inactive bounds            
            eps = 1e-10 #active constraint threshold
            
            ###############################################
            # Constraints, generic form:
            #
            # [lbV] <= [ I] * X <= [ubV]
            # [lbg]    [dg]        [ubg]
            #
            #  lb   <= A*X  <= ub
            #
            # Mu are the associated multipliers
            # A has as many lines as there are Mu:s
            
            lbXi = np.array(QPs[i]['lbX'])
            ubXi = np.array(QPs[i]['ubX'])
            dgi  = np.array(QPs[i]['dg'])
            lbgi = np.array(QPs[i]['lbg'])
            ubgi = np.array(QPs[i]['ubg'])
            
            lb = np.concatenate([lbXi,lbgi])
            ub = np.concatenate([ubXi,ubgi])
            A  = np.concatenate([np.eye(X.shape[0]),dgi])
            Mu = MuBound+Mug
            
            MuLog.append(Mu)
                        
            #Mu2 = MuBound2 + MuEqu + MuIneq
                        
            #plt.figure(334)
            #plt.hold('on')
            #plt.plot(Mu,linestyle = 'none',marker='.',color = 'k')
            #plt.plot(-np.array(Mu2),linestyle = 'none',marker='.',color = 'r')
            #raw_input()
            #plt.close()
            
            #Construct Active Set, Active bounds and Active Constraints
            TgIndices = list(veccat(V.i['Inputs',:,'Tg']))
            
            AS        = []
            AB        = []
            Ag        = []
            ConstRHS  = []
            BoundsGap = np.abs(ub - lb)
            AX        = np.dot(A,X)
            lb_gap = lb - AX
            ub_gap = AX - ub
            for line in range(A.shape[0]):
                if ((lb_gap[line] >= -abs(Mu[line])) or (ub_gap[line] >= -abs(Mu[line])) or (BoundsGap[line] <= eps)):# and not(line in TgIndices):
                    AS.append(line)
                    ConstRHS.append(AX[line])
                    if (line < X.shape[0]):
                        AB.append(line)
                    else:
                        Ag.append(line-X.shape[0])    
                        
            ALog['AS'].append(AS)
            ALog['AB'].append(AB)
            ALog['Ag'].append(Ag)
            GapLog.append(ub_gap)
            
            ###############################################
                       
            #Active constraints & bounds          
            dgActive = A[AS,:]
            
            # KKT Matrix
            KKTMat = np.concatenate([QPs[i]['H'],dgActive],axis = 0)
            Nconst = dgActive.shape[0]
            Addon = np.concatenate([dgActive.T,np.zeros([Nconst,Nconst])])
            KKT = np.concatenate([KKTMat,Addon],axis = 1)
    
            #Right-hand side
            b = np.zeros([KKT.shape[0],len(IndexDual)])
            for col, line in enumerate(IndexDual):
                b[line,col] = 1.
            
            #Right-hand giving the primal-dual solution
            bsol = np.concatenate([-QPs[i]['f']-fadd,np.array(ConstRHS)])
            
            #dPrimalAdjoint = KKT\b
            try:
                #dPrimalAdjoint = KKT\b
                dPrimalAdjoint = linalg.solve(KKT,b)
                
                #Recompute the QP primal-dual solution
                Sol            = linalg.solve(KKT,bsol)
                
                Status['Factorization'].append(True) 
            except: #In case of a badly identified Active Set
                Status['Factorization'].append('RankDefficienty, Turbine'+str(i))
                dPrimalAdjoint, _, _, _ = linalg.lstsq(KKT,b)
                print "Singular KKT matrix, tubine", i
                Status['Factorization'].append(False)
                assert(0==1)
                raw_input()
                
                
            #The "dPrimalAdjoint" provides:
            #[d X       ]
            #[d MuBound ] = dPrimalAdjoint * d Dual
            #[d Mug    ]
            
            
            #### CHECK KKT
            
            X2           = Sol[:X.shape[0]]
            MuBound2     = np.zeros([len(MuBound),1])
            MuBound2[AB] = Sol[X.shape[0]:X.shape[0]+len(AB)]
            Mug2         = np.zeros([len(Mug),1])
            Mug2[Ag]     = Sol[X.shape[0]+len(AB):]
                        
            MuBoundError = (np.array(MuBound) - MuBound2.T).T
            MugError     = (np.array(Mug) - Mug2.T).T
            XError       = X2 - X
            ErrorDual = np.sqrt(np.dot(MuBoundError.T,MuBoundError)) + np.sqrt(np.dot(MugError.T,MugError))
            ErrorPrimal = np.sqrt(np.dot(XError.T,XError))
            
            if (ErrorDual > 1) or (ErrorPrimal > 1):
                
                #plt.figure(997)
                #plt.plot(bsol)
                plt.close('all')
                plt.figure(999)
                plt.hold('on')
                plt.subplot(3,1,1)
                plt.plot(X,color = 'k', linewidth = 2,label='QP sol')
                plt.plot(X2,color = 'r',label='Reconstructed sol')
                #plt.legend()
                
                plt.subplot(3,1,2)
                plt.plot(MuBound,color = 'k', linewidth = 2)
                plt.plot(MuBound2,color = 'r')
                        
                plt.subplot(3,1,3)
                plt.plot(Mug,color = 'k', linewidth = 2)
                plt.plot(Mug2,color = 'r')
                self.MuBound = MuBound
                self.MuBound2 = MuBound2
                raw_input()
            ######

            
            dPrimal.append(dPrimalAdjoint[:X.shape[0],:])
            dAdjoint_i = np.zeros([self._gLocal.shape[0],self.Nshooting])
            for index, adjindex in enumerate(Ag):
                dAdjoint_i[adjindex,:] = dPrimalAdjoint[X.shape[0]+len(AB)+index,:]
            dAdjoint.append(dAdjoint_i)
            
            #dMu only the active ones have sensitivities
            dMu       = np.zeros([len(Mu),self.Nshooting])
            dMu[AS,:] = dPrimalAdjoint[X.shape[0]:,:]
            
            DualHess -= dPrimalAdjoint[IndexDual]
            
            Xall.append(V(X))
    
            AdjointQP = self._gLocal(self._TurbineQPsolver.output('lam_a'))
    
            for key in AdjointQP.keys():
                AdjointUpdate['Turbine'+str(i)+'_'+key] = AdjointQP[key]
                
        return Xall, AdjointUpdate, DualHess, dPrimal, dAdjoint, Homotopy, Status, ALog, MuLog, GapLog

    def DistributedSQP(self, Primal, Adjoint,Dual, Wact, time = 0, iter_SQP = 1,iter_Dual = 1, FullDualStep = True, ReUpdate = True):
        
        self.Embedding(Wact, time)
        
        #LocalResidual = 10
        #while (LocalResidual > 1e-6):
        for iterate in range(iter_SQP):
            
            #Solve central QP
            #QPCentral = self.PrepareCentralQP(self.Solver, Primal, Adjoint)
            #self.solveQP(self._QPsolver, QPCentral)
            #StepCentral = self.V(self._QPsolver.output('x'))
            #muCentral = self.g(DMatrix(self._QPsolver.output('lam_a')))
            #DualCentral = np.array(muCentral['PowerConst',veccat])
            
            #AdjointCentral = []
            #for i in range(self.Nturbine):
            #    AdjointCentral.append(DMatrix(muCentral['Turbine',i,]))
            
            #Construct local QPs (without dualization)
            QPs, R = self.PrepareQPs(Primal,Adjoint)
            
            LocalResidual = 0
            for i in range(self.Nturbine):
                LocalResidual += np.sqrt(np.dot(QPs[i]['g'].T,QPs[i]['g']))
                
            #Dual decomposition iteration
            Norm_Dual = []
            CondHess  = []
            tstepLog  = []
            StatusLog = []
            
            for iter_dual in range(iter_Dual):
            
                
                #Local Primal/Adjoint Step
                StepLocal, Adjoint, DualHess, dPrimal, dAdjoint, Homotopy, Status, AS, MuLog, GapLog = self.QPStep(QPs, Dual)
                StatusLog.append(Status)
                
                
                #Check Dual Hessian conditioning
                eig, _ = linalg.eig(DualHess)
                CondHess = [np.min(np.real(eig)), np.max(np.real(eig))]
                
                if (self.EP['PowerSmoothingWeight'] > 0):
                    #z step
                    Stepz = self.EP['PowerVarRef',veccat]-Primal['PowerVar',veccat] - Dual/float(self.EP['PowerSmoothingWeight'])
                
                    #Constraints residual
                    Residual = Primal['PowerVar',veccat] + Stepz
                    for i in range(self.Nturbine):
                        Residual -= Primal['Turbine',i,'PowerVar',veccat] + StepLocal[i]['PowerVar',veccat]
                    
                    #Dual full step
                
                    StepDual = linalg.solve(DualHess,Residual)
                else:
                    #Allows for simulating a standard wind farm
                    StepDual = 0*Dual
                    Stepz    = 0*Primal['PowerVar',veccat]
                    
                NormStepDual = np.sqrt(mul(StepDual.T,StepDual))

                
                ################### DUAL STEP SIZE ####################
                        
                tstep = 1.
                
                tstepLog.append(tstep)
                #######################################################
                

                Dual -= tstep*StepDual
                          

                
            ################### SECOND UPDATE ####################
                
            if (ReUpdate == True):
                for i in range(self.Nturbine):
                    StepLocal[i] = self._TurbineV(StepLocal[i].cat + np.dot(dPrimal[i],-tstep*StepDual))
                    
                    AdjointUpdate = self._gLocal(np.dot(dAdjoint[i],-tstep*StepDual))
                    for key in AdjointUpdate.keys():
                        Adjoint['Turbine'+str(i)+'_'+key] = AdjointUpdate[key]
                
                if (self.EP['PowerSmoothingWeight'] > 0):
                    Stepz = self.EP['PowerVarRef',veccat]-Primal['PowerVar',veccat] - Dual/float(self.EP['PowerSmoothingWeight'])
                
            #######################################################
                
                
        
            #plt.figure(200)
            #plt.semilogy(Norm_Dual[1:])
            #plt.hold('on')
            #plt.grid()
            #plt.title('2 norm of the (dualized) constraint residual')
            #plt.show()
            ##raw_input()
            #plt.close()
            
            Error = {}
            #Error['PrimalStep'] = 0
            #for i in range(self.Nturbine):
            #    Error['PrimalStep'] += linalg.norm(StepCentral['Turbine',i] - StepLocal[i].cat)
            #    
            #Error['Adjoint'] = linalg.norm(Adjoint.cat - muCentral.cat)
            #Error['Dual']    = linalg.norm(DualCentral - Dual)
      
            #Update primal variables
            for index in range(len(Primal['PowerVar'])):
                Primal['PowerVar',index] += Stepz[index]      
            
            for i in range(self.Nturbine):
                for key in StepLocal[i].keys():
                    Primal['Turbine',i,key,veccat] += StepLocal[i][key,veccat]
                    
                    #Project in the bounds 
                    #Primal['Turbine',i,key,veccat] = min(max(Primal['Turbine',i,key,veccat],self.lbV['Turbine',i,key,veccat]),self.ubV['Turbine',i,key,veccat])
            
            #Pass on Dual variables (not needed, but for consistency)
            Adjoint['PowerConst',veccat] = Dual
            
            #Update the residual (needed if second update is used)
            ResidualOut = Primal['PowerVar',veccat]
            for i in range(self.Nturbine):
                ResidualOut -= Primal['Turbine',i,'PowerVar',veccat]
            
            NormResidual = sqrt(mul(ResidualOut.T,ResidualOut))
            Norm_Dual.append(NormResidual)
            
            #Verbose
            
            print "Time \t Iter \t Dual Residual \t Dual step-size \t Dual full step norm \t Local Residuals"
            print "%3d  \t %3d  \t %.5E     \t %.5E           \t %.5E                \t %.5E" %  (time, iter_dual, NormResidual,  tstep,  NormStepDual, LocalResidual)
                                    	 


            
        return Primal, Adjoint, Dual, ResidualOut, tstep, StatusLog, CondHess, Error, AS, MuLog, GapLog, QPs, R
    
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
                    Primal_struc  = self._TurbineV(Primal['Turbine'][k])
                    Primal0_struc = self._TurbineV(Primal0['Turbine'][k])
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
    
    def PlotBasic(self, T, Primal, time, LW = 1, style = '-', col = 'k', savePath = [], k = 0, DataName = []):
        
        Dic = {}
        Ogmax = 2*pi*1173.7/60
        ScaleT = 1e-4
        Tgmax = 43093.55
        
        plt.figure(300)
        PowerVar = 0
        for i in range(self.Nturbine):
            PowerVar += np.array(Primal['Turbine',i,'PowerVar'])
            plt.hold('on')
            #plt.plot(time['Inputs'],Primal['Turbine',i,'PowerVar'],color=col, linestyle = style,linewidth = LW)
        plt.plot(time['Inputs'],    PowerVar,color=col, linewidth = 2, linestyle = style)
        Dic['Power'] = PowerVar
        
        if isinstance(savePath,str):
            plt.savefig(savePath+'/Power'+str(k)+'.eps',format='eps')
            
        
        plt.figure(400)
        for i in range(self.Nturbine):
            plt.subplot(self.Nturbine,1,i+1)
            plt.step(time['States'],    Primal['Turbine',i,'Wind'],color=col,linewidth = LW, linestyle = style)
            Dic['Wind'+str(i)] = Primal['Turbine',i,'Wind']
        if isinstance(savePath,str):
            plt.savefig(savePath+'/Wind'+str(k)+'.eps',format='eps')
            
        for fig, key in enumerate(T.Inputs.keys()):
            for i in range(self.Nturbine):
                plt.figure(100+fig)
                plt.subplot(self.Nturbine,1,i+1)
                plt.hold('on')
                plt.step(time['Inputs'],    Primal['Turbine',i,'Inputs',:,key],color=col,linewidth = LW, linestyle = style)
                if (key == 'Tg'):
                    plt.axhline(y=Tgmax*ScaleT, xmin=0, xmax=time['Inputs'][-1])
                plt.title('Input '+key+' Turbine '+str(i))
                Dic['Input_'+key+'_Turbine'+str(i)] = Primal['Turbine',i,'Inputs',:,key]
    
            if isinstance(savePath,str):
                plt.savefig(savePath+'/Input'+key+str(k)+'.eps',format='eps')

        for fig, key in enumerate(['beta','Og']):
            for i in range(self.Nturbine):
                plt.figure(200+fig)
                plt.subplot(self.Nturbine,1,i+1)
                plt.hold('on')
                plt.plot(time['Inputs'],    Primal['Turbine',i,'States',:-1,key],color=col,linewidth = LW, linestyle = style)
                if (key == 'Og'):
                    plt.axhline(y=Ogmax, xmin=0, xmax=time['Inputs'][-1])

                plt.title('State '+key+' Turbine '+str(i))
                Dic['State_'+key+'_Turbine'+str(i)] = Primal['Turbine',i,'States',:-1,key]
                
            if isinstance(savePath,str):
                plt.savefig(savePath+'/States'+key+str(k)+'.eps',format='eps')
                
            if isinstance(DataName,str):
                scipy.io.savemat(savePath+'/'+DataName, Dic)
                
    def PlotPaper(self, T, Primal, time, LW = 1, style = '-', col = 'k', savePath = [], k = 0, DataName = []):
        
        Dic = {'Time': time['Inputs']}
        Ogmax = 2*pi*1173.7/60
        ScaleT = 1e-4
        Tgmax = 43093.55
        
        plt.figure(300)
        plt.hold('on')
        PowerVar = 0
        for i in range(self.Nturbine):
            PowerVar += np.array(Primal['Turbine',i,'PowerVar'])
            plt.hold('on')
            #plt.plot(time['Inputs'],Primal['Turbine',i,'PowerVar'],color=col, linestyle = '--',linewidth = LW)
        plt.plot(time['Inputs'],    PowerVar,color=col, linewidth = 2, linestyle = style)
        Dic['Power'] = PowerVar
        plt.title('TP')
        
        if isinstance(savePath,str):
            plt.savefig(savePath+'/Power'+str(k)+'.eps',format='eps')
            
        
        plt.figure(400)
        plt.hold('on')
        for i in range(self.Nturbine):
            #plt.subplot(self.Nturbine,1,i+1)
            plt.step(time['Inputs'],    Primal['Turbine',i,'Wind',:-1],color=col,linewidth = LW, linestyle = style)
            Dic['Wind'+str(i)] = Primal['Turbine',i,'Wind']
        if isinstance(savePath,str):
            plt.savefig(savePath+'/Wind'+str(k)+'.eps',format='eps')
            
        for fig, key in enumerate(T.Inputs.keys()):
            for i in range(self.Nturbine):
                plt.figure(100+fig)
                #plt.subplot(self.Nturbine,1,i+1)
                plt.hold('on')
                plt.step(time['Inputs'],    Primal['Turbine',i,'Inputs',:,key],color=col,linewidth = LW, linestyle = style)
                if (key == 'Tg'):
                    plt.axhline(y=Tgmax*ScaleT, xmin=0, xmax=time['Inputs'][-1],color = 'k',linewidth = 2)
                
                Dic['Input_'+key+'_Turbine'+str(i)] = Primal['Turbine',i,'Inputs',:,key]
            plt.title('Input '+key)
            
            if isinstance(savePath,str):
                plt.savefig(savePath+'/Input'+key+str(k)+'.eps',format='eps')

        for fig, key in enumerate(['beta','Og']):
            for i in range(self.Nturbine):
                plt.figure(200+fig)
                #plt.subplot(self.Nturbine,1,i+1)
                plt.hold('on')
                plt.plot(time['Inputs'],    Primal['Turbine',i,'States',:-1,key],color=col,linewidth = LW, linestyle = style)
                if (key == 'Og'):
                    plt.axhline(y=Ogmax, xmin=0, xmax=time['Inputs'][-1],color = 'k',linewidth = 2)

                
                Dic['State_'+key+'_Turbine'+str(i)] = Primal['Turbine',i,'States',:-1,key]
            plt.title('State '+key)
            
            if isinstance(savePath,str):
                plt.savefig(savePath+'/States'+key+str(k)+'.eps',format='eps')
                
            if isinstance(DataName,str):
                scipy.io.savemat(savePath+'/'+DataName, Dic)
