import numpy as np
from funcs_.null_range_basis import null_basis, range_basis
from funcs_.helpful_scripts import *
import funcs_.Complex_solve as SLV
Tol_ = 0.0001




# Perform change of basis on matrix A.
def basis_transform(S,A):
    Sinv = inverse_(S)
    SM   = np.matmul(Sinv,A)
    SMS  = np.matmul(SM,S)
    return SMS


#===========================================
#======  rANK-NULL TRANSFORM  ==============
#===========================================



def range_null_transform(A_,I_,x ,rnd, Ndim):
    # INPUT: x - a vector of polynomial coefficients (real or imaginary)
    # Ndim is the dimension of matrix input A_. I_ is identity matrix.
    # rnd is a rounding parameter.
    
    # based on x, a scypy str will be constructed for evaluation with arguments
    # A_ and I_. Call the resulting matrix T. 
    
    # Basis for both the range and null space of T are found and hstacked together.
    # A change of basis transformation is then performed on the original matrix A_
    # which results in a decomposition such that
    
    #       p1(A_)*p2(A_) = {0}   ---> p1(A_) ⊕ p2(A_)  = 0 ⊕ 0
    
    # or in terms of our matrix T   
    
    #       T*p2(A_) = {0}   ---> T ⊕ p2(A_)  = {0} ⊕ {0}
    
    # The dimension of the null_space of T is returned as well.
    
    T_facts, T_degens, T_str = factor_expanded_poly(x,rnd)
    #print(f'[null_transform] T_facts = \n{T_facts}')
    #print(f'[null_transform] A_= \n{np.round(A_)}')
    #print(f'[null_transform] T_str = {T_str}')
    T = evaluate_str(T_str,A_,I_)
    #input(f'[null_transform] T{r}: \n{T}')
    T_null_basis, T_null_rank = null_basis(T)#[1:]
    T_range_basis, T_range_rank = range_basis(T)
    
    Sn = np.hstack([T_null_basis,T_range_basis])

        
    An = basis_transform(Sn,A_)
    An = An[T_null_rank:,T_null_rank:]
    Sn = add_rank_null_zeros(Sn,Ndim)
    
    return An,Sn, T_null_rank
    
# add zero columns and rows as neccessary to match dimension
# of original matrix. appended diagonal entries are set to 1.
def add_rank_null_zeros(A,N):
    DIM = A.shape; L = DIM[0]; W = DIM[1]
    if L < N:
        zero_rows = np.zeros([N-L,N])
        zero_cols = np.zeros([L,N-L])
        M = np.hstack([zero_cols,A])
        M = np.vstack([zero_rows,M])
        
        i = 0
        while i < N-L:
            M[i,i] = 1
            i += 1
        return M
    return A
#===========================================
#======  Find eigen basis of subspace =====
#===========================================


# See lemma 1 in read-me file.
def eigen_basis(T, u, k):
        j = 1 ; uj = u; U = uj
        while j < k:
            T_uj = np.matmul(T,uj)
            U    = np.hstack([U,T_uj])
            j += 1
        return U
       


def eigen_transform(A_,x,rnd):
    # INPUT: A_: a matrix. x = array of polynomial coefficients
    # so pn(A) =  ∑i (xi) A^i where xi is a real or imaginary coefficient.
    # rnd: rounding parameter.

    # This functon finds eigen-basis as described in lemma 1 of the read-me file.
    
    Ao  = np.matrix(np.copy(A_))
    I_  = np.eye(len(A_))

    # find roots of x and their algebraic multiplicities, here called degeneracies
    # which is a misnomer (see read-me file).
    p_roots, p_degens = find_poly_roots(x,rnd)
    
    # For each root, construct scypy string (x-root)**degeneracy
    # P_str is the ALL factors multiplied together.
    p_facts, P_str = construct_factored_poly(p_roots, p_degens)
    
    #print(f'\n p_roots: {p_roots}')
    #print(f' p_facts:  {p_facts}')
    #print(f' P_str:    {P_str}')
    #print(f' p_degens: {p_degens}')
    #print('\n====================')
    
    # Now we must do a shuffle routine. Call the nth factors fn. We have,
    
    #     F1(A_)*F2(A_)*...*FL(A_) = 0

    
    # We need to constuct a basis of the form {u,Tu,T^2u,..T^(k-1)u} 
    # where T^k u = 0 and T represents a MONIC polynomial term, e.g. (A-λ).
    # We factored by roots so we have all such terms.
    
    # Let fn(A_)**kn = Fn(A_)

    # Now call f1(A_) = T, p_degens[0] = k1
    # Then pick any non-zero column of u_str = f2(A_)*...*fL(A_) and call it u. 
    # Now find {u,Tu,T^2u,..T^(k-1)u}.
    
    # !!! The catch: We now have to shuffle f2(A_) to the first position, 
    # call it T, construct and evaluate the polynomial string u_str = f1(A_)*...*fL(A_) 
    # pick any u, and repeat the procedure to find an eigen-base for T = f2(A_). 
    
    # In short; we are decomposing TU = 0 in every way possible.
    
    L = len(p_facts)
    if L > 1 :
        l= 0
        while l < L:
            T_fact = p_facts[l]
            T  = evaluate_str(T_fact ,A_ ,I_)
            k  = p_degens[l]; 
            

            u_facts  = np.copy(p_facts)
            u_facts  = np.delete(u_facts,l)
            u_degens = np.copy(p_degens)
            u_degens = np.delete(u_degens,l)
            U_str    = construct_poly_facts(u_facts, u_degens)[1]
            u_mat    = evaluate_str(U_str ,A_ ,I_ )
            
            # find non-zero basis vector of u_mat to use for
            # u0 in constructing eigen-basis {u0,Tu0,T^2u0,...}
            u_range, rank_u = range_basis(u_mat)
            u  = u_range[:,1]
            ul_basis = eigen_basis(T, u, k)[:,::-1] #!!!!

            if l == 0:
                S = ul_basis
            if l > 0:
                S = np.hstack([S,ul_basis])

            l += 1
            
            
    # Here we have the case (T^k)I = 0
    if L == 1: 
        k  = p_degens[0]
        if k > 1:
            T_fact = p_facts[0]
            T  = evaluate_str(T_fact ,A_ ,I_)
            
            # construct unit vector e1 to use as u vector
            e1    = np.zeros(len(T))
            e1[0] = 1
            e1    = np.matrix(e1).transpose()
            

            S  = eigen_basis(T, e1, k)[:,::-1]
        if k == 1:
            S  = np.matrix(np.eye(len(A_)))
    an = basis_transform(S,A_)
    return S, an



# Adds zeros rows to top and bottom of a set of basis so that they have 
# the same dimension as N - presumably the length of hte original matrix.

# *Note: This is for EIGEN transform for which we need not worry about 
# adding zero COLUMNS.
def add_subspace_zeros(S,null_cnt,N):

    DIM = S.shape; L = DIM[0]; W = DIM[1]
    if L < N:
        top_zeros   = np.zeros([null_cnt,W]).astype('complex128')
        bottom_cols = np.zeros([N-(null_cnt+L),W]).astype('complex128')
        if len(top_zeros)> 0:
            S = np.vstack([top_zeros,S])
        if len(bottom_cols)> 0:
            S = np.vstack([S,bottom_cols])
    return S





      

















