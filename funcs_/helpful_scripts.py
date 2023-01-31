import numpy as np
from time import sleep
from sympy import * 
from sympy.abc import x, y, z

 

#----------------------- Gaussian reduction/ solving operations
def row_swap(i1,i2,N): 
    R = np.eye(N)
    R[[i1,i2],:] = R[[i2,i1],:]
    return R

# find first non-zero column in matrix
def find_nonzero_col(M):
    j = 0 ; W = M.shape[1]
    TF = False
    while TF == False:
        colj = M[:,j]
        if np.any(np.real(colj)!=0) and np.any(np.imag(colj)!=0):         
            return colj
        j+=1
    input('[find_nonzero_col] no zero col found')

# find first non-zer element in a vector
def find_first_el(v,Tol_):
    if type(v)==np.matrix:
        dim = v.shape; w = dim[1]; l = dim[0]
        if w > l:
            v = np.array(v)[0]
    i = 0
    while i<len(v):
        if abs(v[i]) > Tol_:
            return i
        i += 1
    return -1



# shuffle zero rows to top
def arrange_reduced_mat(A):
    
    DIM = A.shape; L = DIM[0] ; W = DIM[1] 
    count_dep = 0; 
    J_first = np.zeros(L); R = np.eye(L)
    i = 0
    while i<L:
        first_val_indx = np.where(A[i,:]!=0)[0]
        if len(first_val_indx)>0:
            val_i = first_val_indx[0]
        else:
            val_i = L+count_dep
            count_dep += 1
        J_first[i] = val_i
        i +=1
    new_order = np.sort(J_first)
    i = 0
    while i < len(new_order):
        i_swap = np.where(J_first==new_order[i])[0][0]
        if i != i_swap and J_first[i] != new_order[i]:
            Ri = row_swap(i_swap,i,L)
            R  = np.matmul(Ri,R)
            J_first = np.matmul(Ri,J_first)
        i += 1
    M = np.matmul(R,A)
    return M,R
    

#----------- Evaluate scypy str
I_ = np.matrix(np.eye(2))
def evaluate_str(str_,A_,I_):
    Tol_ = 5
    #print(f'[null_transform] I_= \n{np.round(I_)}')
    #print(f'[null_transform] A_= \n{np.round(A_)}')
    #input(f'[null_transform] T_str= {type(A_)}')
    u_mat   = lambdify([x,I],str_)(A_,I_)
    #print(f'[evaluate_str] u_mat= \n{u_mat}')
    u_mat   = np.matrix(np.round(u_mat,Tol_))
    return u_mat

#----------- Find inverse of matrix
def inverse_(M):
    import numpy as np
    try:
        return np.linalg.inv(M)
    except:
        input("Singular Matrix, Inverse not possible.")




#--------------- Constructing polynomial strings

# construct expanded polynomial str from coeeficient inputs.
def pn_string(xn):
    p_expr = f'{xn[0]}*x**{0}'
    j = 1
    while j < len(xn):
        if xn[j] != 0:
            p_expr += f' + {int(xn[j])}*x**{j}'
        j+=1
    return p_expr


# construct factored polynomial string from roots
def construct_factored_poly(Roots, Degeneracies):
    fact0 = f'x-{Roots[0]}*I'
    i = 1; p_facts = [fact0]; P_str = f'(({fact0})**{Degeneracies[0]})'
    while i < len(Roots):
        facti  = f'x-{Roots[i]}*I'
        P_str += f'*(({facti})**{Degeneracies[i]})'
        p_facts.append(facti)
        i += 1
    return p_facts, P_str


# construct factored polynomial string from factor (str) inputs, i.e. multiply 
# a list of factors.
def construct_poly_facts(facts, degens):
    
    fact0 = f'(({facts[0]})**{degens[0]})'
    i = 1; p_facts = P_str = fact0
    while i < len(facts):
        
        P_str += f'*(({facts[i]})**{degens[i]})'
        i += 1
    return p_facts, P_str

# INPUT: take vector x of expanded polynomial coefficients. 
# OUT: list of factor strings suitable for scypy evaluation. 
# Also a list of their algebraic multiplicities and a str value of
# all factors multiplied together (useful for keeping track of Pmin)
def factor_expanded_poly(x,rnd):
    _roots, _degens = find_poly_roots(x,rnd)
    _facts, _str = construct_factored_poly(_roots, _degens)
    return _facts, _degens, _str



#----------- Finding polynomial roots
# find where item occurs in a list
def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

# solve expanded polynomial w/ numpy roots function. Then locate/ count/ delete
# repeated values. returns list of roots and their multiplicities (or  in physics
# terminology, the degeneracies). Among mathematicians what I am calling the 
# degeneracy would usually be referred to as the algebraic multiplicity of the 
# eigenvalue.
# Note that scypy will not factor an expression if it has to resort to floats.
# Ergo, constructing the polynomial from the roots is the most tobust way to 
# go about constructing a scypy string to evaluate.
def find_poly_roots(x,rnd):
    roots_ = np.round(np.roots(x),rnd)
    i = 0; Roots = []; Degeneracies = []
    no_check_list = []
    
    # consolodate list (count/ eliminate repeated roots)
    while i < len(roots_):
        rooti  = roots_[i]; 
        
        if rooti not in no_check_list:
        
            # find repeated roots:
            repeats = find_indices(roots_, rooti)
            Roots.append(rooti)
            Degeneracies.append(len(repeats))
            
            # update no-check list so we don't count this root again
            no_check_list.append(rooti)
        
        i += 1
    return Roots, Degeneracies








#########################
#####     Extra      ####
#########################

# on page six the author claims, "First, note that each vector v ∈ V can be 
# represented as v = P(T)u for some polynomial P over the scalar field F of 
# degree less than k. (see page 6 or ref. 1 in the README doc.)

# This function accepts an arbitrary vector v, a matrix T with minimal polynomial
# pn that has coefficients xn. It deconstructs v into a linear combination of basis vectors
# of which are calculated at every step . This is to say v can be
# represented as 

# v = xn[0]u  + xn[1]Tu + xn[2]([T**2]u) + .... xn[K]([T**K]u) 
#   = xn[0]u0 + xn[1]u1 + xn[2]u2        + .... xn[K]uK

# Where K is the degree of the polynomial pn and u0,u1,...uK are basis vectors.
# of the subspace.

# RETURNS: matrix U whose rows are basis vectors u0,u1,...,uK. u0 is obviously of 
# particular interest (because ui = [T**i]u0).  This is essentially what we are 
# solving for, i.e. we are working in reverse; we know v, we now seek a basis in 
# which to represent it.
# Also prints out v after constructing it in terms of basis vectors as a check the 
# process is working.

def v_as_Pu(T,xn,v):
    
    
    K = len(xn)
    
    
    Tj = np.eye(len(T))
    T_powers = [Tj]
    j = 1
    while j < K:
        Tj = np.matmul(T,Tj)
        T_powers.append(Tj)
        j += 1
    

    i = K-1
    while i >= 0:
        Ti = T_powers[i]
        ui = np.matmul(Ti,v)/xn[0]
        j = i+1
        
        while j <= K-1:
            uj  =  globals()['u'+str(j)]
            ui += -(xn[j-i]*uj)/xn[0]
            j  +=  1
        globals()['u'+str(i)] = ui

        
        if i == K-1:
            U = ui
        else:
            
            U = np.vstack([U,ui])

        i += -1
    
    
    i = 0 ; v = 0
    while i < K:
        ui = globals()['u' + str(i)]
        v += xn[i]*ui
        i += 1
    print(f'\n v = {v}')
    print(f'\n U = \n{U}')
    print(f'basis vector u0: \n{u0}')
        









