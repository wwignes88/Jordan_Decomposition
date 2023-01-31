import numpy as np
from funcs_.Complex_Gauss import row_reduced
from funcs_.helpful_scripts import find_first_el
Tol_ = 0.0001
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------



def null_basis(M):  
    DIM = M.shape ; L = DIM[0] ; W = DIM[1] ;    

    v = np.matrix(np.zeros(L)).transpose()
    # --------------- [Gaussian] row reduction:

    M,E = row_reduced(M)
    
    # ----------------------
    x_solve = -np.ones(W)
    # ^^ initialize index array of x vals to be solved for.
    # -1 will let us no not to solve for a given variable (it is dependent so we CHOOSE its value)
    
    # find index of first non-zero element in row i:
    i = 0
    rank_ = 0 # also find rank of matrix.
    while i < np.min([L,W]): 
        non_zero   = find_first_el(M[i,:], Tol_) 
        x_solve[i] = non_zero  
        if non_zero >= 0 :
            rank_ += 1 
        i+=1

    # count/ find indices of dependent variables.
    x_dep = [] 
    k = 0 
    while k < len(x_solve):
        if x_solve[k] < 0:
            x_dep.append(k)
        k += 1

    # if no depdendent variables, there is no null space.
    if len(x_dep) == 0:
        return np.matrix(np.eye(W)), W
    
    #print(f'M: \n{np.round(M)}')
    #print(f'\nx_solve : {x_solve}')   
    #print(f'x_dep : {x_dep}')
    #print(f'rank: {rank_}')
    
    k = 0
    while k < len(x_dep):
        x = np.zeros(W).astype('complex128')
        # set one dependent variable at a time to 1. Others remain 0.
        x[x_dep[k]] = 1 + 0j
  
        #print(f'x: {x}')
        
        i = np.min([L,W])-1
        while i >= 0:
            rowi = np.array(M[i,:])[0]
            i_solve = int(x_solve[i]); 
            
            
            if i_solve == W-1: # cannot call x[W:]!!!
                x[i_solve] = v[i]/rowi[i_solve] 
                
            if i_solve >=0 and i_solve < W-1:
                x[i_solve] = (v[i]-np.sum(x[i_solve+1:]*rowi[i_solve+1:]))/rowi[i_solve]
            
            i += -1
            
        # add solution to null space basis set.
        if k == 0: 
            Null_basis = np.matrix(x).transpose()
        if k > 0:
            x_ = np.matrix(x).transpose()
            Null_basis = np.hstack([Null_basis,x_])
        k += 1 
                      # rank of null-space = W-rank_
    return Null_basis, W-rank_
    
    

#===========================================
#======  Find range          basis =========
#===========================================
import funcs_.Complex_solve as SLV


def range_basis(A):
    DIM = A.shape; L = DIM[0] ; W = DIM[1]
    M   = np.matrix(np.copy(A))
    
    zeros  = np.zeros(L).astype('complex128')
    zeros  = np.matrix(zeros).transpose()
    solve_ = SLV.solve(M,zeros, 0)
    x_solve= solve_[3]
    rank   = solve_[2]
    M      = solve_[4]

    #print(F'\n[range]  M: \n{np.round(M,2)}')
    #print(F'x_solve: {x_solve}')
    #print(F'A: {A}')
    #input(F'[range] rank   : {rank}')
    
    
    if len(x_solve) == 0:
        raise ValueError("[range_null_basis.py] range_basis does not accept zero matrix")
    

    i = 0
    while i < len(x_solve):
        indx = x_solve[i]
        if indx >= 0:
            col = A[:,indx] 

            try:
                s = np.hstack([s,col])

            except:
                s = col
                
   
        i   += 1
    #input(F's{i}: \n{s}')
    return s, rank
