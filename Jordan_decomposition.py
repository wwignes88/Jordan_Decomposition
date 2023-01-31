# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:27:26 2022

@author: Wayne
"""

from funcs_.Complex_Gauss import row_reduced
from funcs_.Complex_solve import solve as Isolve
from   sympy import * 
from   sympy.abc import x, y, z
import numpy as np
from   funcs_.helpful_scripts import *

Sleep_toggle = 0
input_toggle = 0


#---------------------------------
def find_Pmin(A):

    # A is presumed square matrix!!!
    A_ = np.matrix(A) 
    DIM = A.shape ; L = DIM[0] ; W = DIM[1] 

    # define basis vectors:
    i = 0
    while i < W:
        ei = np.zeros(W)
        ei[i] = 1
        ei = np.matrix(ei).transpose()
        globals()['e'+str(i+1)] = ei
        i += 1

    # initialize parameters
    I_     = np.matrix(np.eye(W)).astype('complex128')  
    zeros_ = np.matrix(np.zeros(W)).transpose() 
    Pmin   = np.matrix(np.copy(I_)).astype('complex128') 
    X = [] # polynomial factor(s) coefficient list
    n = 1 
    
    while  n <= W: 
    
        # find minimum dependent set of vectors ui = (A**i)(Pn)(en)
        # c.f. pg. 14, section 6      
        en  = globals()['e'+str(n)] ;  
        
        ui  = np.matmul(Pmin,en)  ; # u0
        Un  = ui ; U_ind_sys = []
        
        Dep = False ; i = 1
        while Dep == False:

            ui = np.matmul(A_,ui); # u1, u2,....
            Un = np.hstack([Un,ui])   
            
            Test = Isolve(Un, zeros_,1) # dependent variables set to 1
            
            if Test[0] == 'dependent':                
                print(f'\n============= n={n} \n')
                
                # coefficients of [expanded] pn factor
                xn = Test[1] 
                print(f'x{n}: {xn}')
                X.append(xn)

                # find roots of pn pn factor, construct polynomial factor string
                p_roots, p_degens = find_poly_roots(xn,rnd)
                p_facts, P_str = construct_factored_poly(p_roots, p_degens)
                
                # string representations of Pmin (to be updated @ each n val)
                if n == 1:
                    Pmin_str = f'{P_str}'
                if n > 1:
                    Pmin_str = P_str + '*'+ Pmin_str
                
                # functionalize/ evaluate Pmin_str
                Pmin = evaluate_str(Pmin_str,A_,I_)

                # print statements (for debugging)
                #print (f'[minimal dependent system] U{n}:\n{Un}')
                #print(f'\n p_roots (λ vals): {p_roots}')
                #print(f' p{n}_facts:  {p_facts}')
                #print(f' p{n}_str:    {P_str}')
                #print(f' p{n}_degens: {p_degens}')
                print(f'\nPmin_str:    {Pmin_str}\n')
                #print (f'\n============\nn = {n}:')
                Dep = True # exit loop
            i += 1 


        #-------------------------------
        # Find next n value using the criteria Pmin(A)*en ≠ 0


        j = n+1; Elim = True       
        while Elim == True and j <= W:
            ej      = globals()[f'e{j}']

            # check if  Pmin(A)*ej ≠ 0
            check   = np.matmul(Pmin,ej)
            if np.all(abs(check) == 0):
                print(f'Pmin(A)*e{j} == 0')

            else: 
                print(f'Pmin(A)*e{j} ≠ 0')
                Elim = False
                
                # Fail-safe: hopefully we'll never reach n=W without finding 
                # Pmin such that Pmin(A)*eW ≠ 0 
                if n == W:
                    raise ValueError("[Jordan_decomposition.py] No Pmin found that yeilds {0} matrix!!!!. check math")
        
                break
            j  += 1 ; 
        n = j; print(f'\nsetting n={n}...');
        #sleep(1)


    #----------------------------
    #sleep(2)
    print('\n====================')
    # Print final polynomial, pn coefficients list, and check Pmin(A) == 0.


    print('\nPmin(A) (should be {0} matrix) ' + f':\n{Pmin}')
    print(f'\nPmin    : {Pmin_str}')
    print(f'\npolynomial factor coefficients:\nX: {X}')
    return X



#########################################
######      Launch Example     ##########
#### find  Pmin = p1*p2*...*pn   ########
#########################################


# this example is taken from section 6 of 
# Attila M´at´es paper (ref [1] in READ_ME file).

A = np.matrix(' 0 1   0  0 -1 -1 ;\
               -3 8   5  5  2 -2 ;\
               1  0  -1  0 -1  0 ;\
               4 -10 -7 -6 -3  3 ;\
               -1 3   2  2  2 -1 ;\
               -2 6   4  4  2 -1')
           
DIM = A.shape; L = DIM[0]; W = DIM[1]
rnd = 2

# =====================================================
# Find minimal polynomial factor coefficients : a list of vectors corresponding
# to Pmin(A) = p1(A)p1(A)...pn(A)
X = find_Pmin(A)[::-1]

print('\n\n=============================')
print('=============================\n')

input(f'now to find subspace transforms. \n \
      Press ENTER key to continue')




#=======================================================
#============ NULL-SPACE TRANSFORMATION ================
#=======================================================
print('NULL_SPACE DECOMPOSITION:\n')

# We aim to transform so as to decompose Pmin(A) = p1(A)p1(A)...pn(A) 
# into  two matrices; T and U such that TU = 0. We then find the null-space
# of both T and U, construct a transformation basis by putting them together,
# then making the transformation TU --> T ⊕ U = {0} ⊕ {0}

# Using the polynomial factor coefficients in list X which was constructed in the previous
# section, we then find polynomial factors of U such that U can be written as U = T'U'
# and iterate the procedure until the minimal polynomial has been represented as,

# Pmin(A) = p1(A) ⊕ p1(A) ⊕ ... ⊕ pn(A)  = {0} ⊕ {0} ⊕ .... ⊕ {0} 

# Ergo, we keep track of null space of T at each step so we can later CALCULATE the 
# indices of the subspaces where we will perform eigen-value decomposition. Consecutive
# indices in a list of such indices will represent the upper left and bottom right corners
# of a given null-subspace.

from funcs_.transforms import (range_null_transform,
                               basis_transform,
                               eigen_transform,
                               add_subspace_zeros)
rnd = 2


# initialize null-space transformation matrix.
# *note: functions are designed to accept numpy MATRIX (not array) inputs.
An = np.matrix(np.copy(A)) 

# create original copies of An
Ao = np.matrix(np.copy(An)) ; Aoo = np.matrix(np.copy(Ao))
I_ = np.eye(L); In = np.copy(I_); 
Ndim = len(An) # dimension of matrix, presumed to be square matrix.
Snull = np.eye(L)  # initialize null-space transformation basis matrix 

T_null_ranks = [] # dimension of null space of T is 
                  # bottom right index of null-subspace blocks.
                  # The are not Jordan blocks. The next section will 
                  # reduce these blocks further into Jordan blocks.

n = 0
null_ct = 0
while n < len(X)-1:
    print('\n--------------------')
    
    # perform null-space transform on An. Dimensions of An will reduce at every n step.
    An,Sn, T_null_rankn = range_null_transform(An,In,X[n],rnd, Ndim)
    An = np.round(An)  # can round to desired accuracy. 
    In = np.eye(len(An)) # redefine Identity matrix to fit the size of new subspace.
 
    Snull  = np.matmul(Snull,Sn) # @ each step update transform matrix, i.e. S = ...S2*S1*I

    # index bottom-right corner of jordan blocks.
    T_null_ranks.append(T_null_rankn)
    
    A_transformed = basis_transform(Snull,Ao)
    print(f'A transformed @ {n+1}th subspace = \n{np.round(A_transformed)}')
    print(f'\nT{n+1} null space dimension  = {T_null_rankn}')
    
    # print block matrices identified
    subn1 = A_transformed [null_ct:null_ct+T_null_rankn,  null_ct:null_ct+T_null_rankn ]
    print(f'\nA{n} = \n{np.round(subn1)}') 
    subn2 = A_transformed [null_ct+T_null_rankn:, null_ct+T_null_rankn: ]
    input(f'\nA{n+1} = \n{np.round(subn2)}\n \
          Press ENTER key to continue') 
          
    null_ct += T_null_rankn
    
    n += 1
T_null_ranks.append(W)

print('\n--------------------')
# perform null-space transformation on unreduced matrix A.
M = basis_transform(Snull,Ao)
input(f'M after null-space transform ;\n{np.round(M)}\n \
      Press ENTER key to continue')


#=======================================================
#============ EIGEN-SPACE TRANSFORMATION ===============
#=======================================================
print('\n\n=============================\n')
print('EIGEN_SPACE DECOMPOSITION:\n')

# We aim to transform so as to decompose Pmin(A) = p1(A)p1(A)...pn(A) 
# into  two matrices; T and U such that TU = 0. We then find the null-space
# of both T and U, construct a transformation basis by putting them together,
# then making the transformation TU --> T ⊕ U = {0} ⊕ {0}


# As previously explained, the dimensions (or ranks) of the null-spaces allow us to 
# calculate the upper left and bottom right indices of the null-subspaces.
# For each n in the list 'T_null_ranks' we use consecutive indices to thes values.
# To this end, @ a given n val 'null_cnt' counts the sum of all indices to the left of n
# in this list. This is the upper left corner of the null-subspace we aim to decompose
# even further into an eigen-subspace. The bottom right corner will be null_cnt + the nth
# value in 'T_null_ranks'. 

# These indices allow us to replicate the null subspace matricies of

# Pmin(A) = p1(A) ⊕ p1(A) ⊕ ... ⊕ pn(A)  = {0} ⊕ {0} ⊕ .... ⊕ {0} 

# i.e. A1 = p1(N), A2 = p1(M) where p1(M) is the transformed matrix that was 
# constructed above. Subspaces are more abstract than tangible, so An is not the 
# subspace itself, but it does have equal dimensions to the null-subspace it inhabits.


# As Pmin is not a direct sum of block matricies, any eigen-bases we construct need
# to have dimension equal to the original matrix A (Ndim). Ergo, after an eigen-base is found
# for each of these blocks, zeros rows will need to be added above and below as needed to
# accomplish this. To this end having an index that tells us the verticle position of the 
# blcks is of course very handy. The function 'add_subspace_zeros' accomplishes this.

n = 0 ; null_cnt = 0 
while n < len(T_null_ranks):
    
    rank = T_null_ranks[n] ; 
    xn   = X[n]
    An   = M[null_cnt:null_cnt+rank, null_cnt:null_cnt+rank]
    print(f'\nA{n} : \n',np.round(An))
    Sn,an    = eigen_transform(An,xn,rnd)
    print(f'\nA{n} eigen decomposition: \n{np.round(an)}\n')    
    Sn = add_subspace_zeros(Sn,null_cnt,Ndim)

    null_cnt += rank
    
    # construct eigen-transform matrix.
    if n == 0: 
        Seigen = Sn
    if n > 0:
        Seigen = np.hstack([Seigen,Sn])
    n += 1
    input('--------------------------\n \
          Press ENTER key to continue')


print('\n\n=============================\n')
print('JORDAN MATRIX:\n')

# multiply S (rank-null transform) by SN (eigen-subspace transform).
S = np.matmul(Snull,Seigen)

J = basis_transform(S,Aoo)
print(f'\nJ= \n{np.round(J)}')





























