import cplex
import numpy as np
from scipy.sparse import *


"""
   Converts a sparse matrix to column compressed format of CPLEX
"""
infinity = cplex.infinity 
def sparseToList(A):
    if type(A) == list:
        return []
    (n, m) = A.shape
    # Going through columns
    Aout = list()
    for k in range(m):
       rowind = A.indices[A.indptr[k] : A.indptr[k+1]]
       first = []                                                                                                                                                                
       for j in rowind:
          first.append(int(j))
       col = list()
       for j in A.data[A.indptr[k] : A.indptr[k+1]]:
           col.append(float(j))
       second = col
       Aout.append([first, second])
    return Aout


class CplexQPSolver:
  def __init__(self, verbose = False):
    # Problem object
    self.p = cplex.Cplex()
    self.is_warm = False
    self.verbose = verbose

    # Setting options
    self.p.parameters.qpmethod.set(0)
    # Enabling pure convex objectives (only for barrier)
    self.p.parameters.solutiontarget.set(2)

    # Setting tolerance
    self.p.parameters.barrier.convergetol.set(1E-12)

    if not verbose:
      self.p.parameters.barrier.display.set(self.p.parameters.barrier.display.values.none)
      self.p.parameters.simplex.display.set(self.p.parameters.simplex.display.values.none)
      self.p.set_results_stream(None)
    else:
      self.p.parameters.barrier.display.set(self.p.parameters.barrier.display.values.normal)
      self.p.parameters.simplex.display.set(self.p.parameters.simplex.display.values.normal)

    self.p.parameters.threads.set(1)
    self.p.objective.set_sense(self.p.objective.sense.minimize)
    self.p.parameters.barrier.crossover.set(1)######################################################### 1 = crossover active, 0 = crossover inactive
  #---------------------------------------------------------------------------------

  """
     Solves the QP with CPLEX
	min  0.5 * x' Q x + c'x
	s.t. A <= b
	     Aeq = b
	     lbx <= x <= ubx
  """
  def cplex_solve(self, Q, c, A = [], b = [], Aeq = [], beq = [], lbx = [], ubx = [], outfile = ''):
     p = self.p
     # Preprocessing vectors a bit
     if type(c) == np.ndarray:
       if c.ndim == 2:
	 c = (c.T)[0].astype(float)
       else:
	 c = c.astype(float)
     if type(b) == np.ndarray:
       if b.ndim == 2:
	 b = (b.T)[0].astype(float)
       else:
	 b = b.astype(float)
     if type(beq) == np.ndarray:
       if beq.ndim == 2:
	 beq = (beq.T)[0].astype(float)
       else:
	 beq = beq.astype(float)
     if type(lbx) == np.ndarray:
       if lbx.ndim == 2:
	 lbx = (lbx.T)[0].astype(float)
       else:
	 lbx = lbx.astype(float)
       for k in range(lbx.size):
	 if lbx[k] == -np.inf:
	   lbx[k] = -cplex.infinity
     if type(ubx) == np.ndarray:
       if ubx.ndim == 2:
	 ubx = (ubx.T)[0].astype(float)
       else:
	 ubx = ubx.astype(float)
       for k in range(ubx.size):
	 if ubx[k] == np.inf:
	   ubx[k] = cplex.infinity



     if c != []:
	c = c.tolist()
     if b != []:
	b = b.tolist()
     if beq != []:
	beq = beq.tolist()


     if lbx == []:
	 lbx = [-cplex.infinity] * len(c)
     else:
	 lbx = list(lbx)
     if ubx == []:
	 ubx = [+cplex.infinity] * len(c)
     else:
	 ubx = list(ubx)
     Qconv = sparseToList(Q)
     if A == []:
	 Ash = [0, 0]
     else:
	 Ash = A.shape
     if Aeq == []:
	 Aeqsh = [0, 0]
     else:
	 Aeqsh = Aeq.shape
	 
     Aunited = []
     if A == []:
	 Aunited = Aeq
     elif Aeq == []:
	 Aunited = A
     else:
       Aunited = lil_matrix((Ash[0] + Aeqsh[0], Ash[1]))
       Aunited[0 : Ash[0], 0 : Ash[1]] = A
       Aunited[Ash[0] : Ash[0] + Aeqsh[0], 0 : Ash[1]] = Aeq
       Aunited = Aunited.tocsc()
       
      
#   Aconv = sparseToList(A)
#   Aeqconv = sparseToList(Aeq)
  
       

     Aconv = sparseToList(Aunited)


     rhs = np.concatenate([b, beq]) 
     sense = len(b) * 'L' + len(beq) * 'E'
     # Setting up everything
     if not self.is_warm:
       p.linear_constraints.add(rhs = rhs, senses = sense)
       p.variables.add(obj = c, lb = lbx, ub = ubx, columns = Aconv )
       p.objective.set_quadratic(Qconv)
     else:
       p.parameters.qpmethod.set(0)
       # Changing only linear term
       linterm = []
       for k in range(len(c)):
	 linterm.append((k, c[k]))
       p.objective.set_linear(linterm) 
       # Changing quadratic term 
       p.objective.set_quadratic(Qconv)
       
       # Injecting initial basis
       p.start.set_start(self.basis_pr, self.basis_slack, self.x_opt, [], self.dual_opt, [])

     if outfile != '':
	 p.write(outfile)

     # Solving QP...
     p.solve()
     
     
     # Obtaining dual variables
     mu = []
     for k in range(Ash[0]):
	 mu.append(p.solution.get_dual_values(k))
	 
     lamb = []
     for k in range(Ash[0], Ash[0] + Aeqsh[0]):
	 lamb.append(p.solution.get_dual_values(k))
     psi = p.solution.get_reduced_costs()

     self.x_opt = p.solution.get_values()
     self.dual_opt = p.solution.get_dual_values()

     f_opt = p.solution.get_objective_value()
     sol_status = p.solution.get_status()
     sol_string = p.solution.get_status_string()

     # Obtaining basis
     (self.basis_pr, self.basis_slack) = p.solution.basis.get_basis()
#   for k in range(len(c)):
#       x_opt.append(p.solution.get_values(k))
     self.is_warm = True
     return (self.x_opt, f_opt, mu, lamb, psi, sol_status, sol_string)
       
