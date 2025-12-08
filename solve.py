import numpy as np
import generate
import time
import scipy.sparse
from sympy.physics.quantum.cg import CG
import copy



def precomp_Clebsch(L,P):
    #run this with P = 2*L to guarantee that you can run it for lower L,P.
    clebschL = np.zeros((L+1,2*L+1,L+1,2*L+1,2*L+2),dtype=np.complex128)
    clebschR = np.zeros((L+1,2*L+1,L+1,2*L+3),dtype=np.complex128)
    for n in range(-L,L+1):
        for l1 in range(0,L+1):
            for l2 in range(0,L+1):
                l3_u = 2*L+2
                l3_l = 0
                for l3 in range( l3_l , l3_u + 1):
                    clebschR[l1,n+L,l2,l3] = CG(l1, n, l2, -n, l3, 0).doit()

    for l1 in range(0,L+1):
        for l2 in range(0,L+1):
            for m1 in range(-l1, l1+1):
                for m2 in range(-l2,l2+1):
                    l3_l = max(abs(m1-m2), abs(l1-l2))
                    l3_u = min(l1+l2, P)
                    for l3 in range( l3_l , l3_u + 1):
                        clebschL[l1,m1+L,l2,m2+L,l3] = CG(l1, m1, l2, -m2, l3, m1-m2).doit()


    filename = 'precomputed_data/precomputed_ClebschL_L_'+str(L)+'_P_'+str(P)
    np.save(filename, clebschL)

    filename = 'precomputed_data/precomputed_ClebschR_L_'+str(L)+'_P_'+str(P)
    np.save(filename, clebschR)

def precomp_N(L):
    res = np.zeros((2*L+1,L+1),dtype=np.complex128)
    for n in range(-L,L+1):
            for l1 in range(0,L+1):
                res[n+L,l1] = generate.cN(n, l1)

    filename = 'precomputed_data/precomputed_N_L_'+str(L)
    np.save(filename, res)

def stackB(calliB):
    L = len(calliB)
    stackedB = np.zeros(((L)**2,(L)**2),dtype=np.complex128)
    for l1 in range(L):
        for l2 in range(L):
            stackedB[(l1)**2:(l1+1)**2,(l2)**2:(l2+1)**2] = calliB[l1][l2]

    return stackedB


def stackO(Os):
    L = len(Os)
    stackedO = np.zeros(((L)**2,(L)**2),dtype=np.complex128)
    for l in range(L):
            stackedO[(l)**2:(l+1)**2,(l)**2:(l+1)**2] = Os[l]

    return stackedO





def build_lsq_matrix(L,P,numeqs):
    loadL = 10 #replace by largest precomputed file in the folder
    loadP = 20 #replace by largest precomputed file in the folder
    cL = np.load('precomputed_data/precomputed_ClebschL_L_'+str(loadL)+'_P_'+str(loadP)+'.npy')
    cR = np.load('precomputed_data/precomputed_ClebschR_L_'+str(loadL)+'_P_'+str(loadP)+'.npy')
    N_array = np.load('precomputed_data/precomputed_N_L_'+str(loadL)+'.npy')
    A = scipy.sparse.lil_matrix((numeqs,(P+1)**2), dtype=np.complex128)
    indP = np.zeros((2,P+1)) # list containing the ranges for a given p:
             # indP[:,p] = the ranges of the B list variable
             # containing B_{p,u} for u ranging from -p to p
    minP = 0
    maxP = 1
    for p in range(P+1):
        indP[0,p] = minP
        indP[1,p] = maxP
        minP = maxP
        maxP = maxP + 2*(p+1)+1

    row_ind = 0
    for n in range(-L,L+1):
        for l1 in range(0,L+1):
            for l2 in range(0,L+1):
                for m1 in range(-l1, l1+1):
                    for m2 in range(-l2,l2+1):
                        l3_l = max(abs(m1-m2), abs(l1-l2))
                        l3_u = min(l1+l2, P)
                        if l3_u >= l3_l:    
                            for l3 in range( l3_l , l3_u + 1):
                                tmp = cL[l1,m1+loadL,l2,m2+loadL,l3]*cR[l1,n+loadL,l2,l3]
                                tmp1 = N_array[n+loadL,l1]
                                tmp2 = N_array[n+loadL,l2]
                                A[row_ind,int(indP[0,l3]+l3-m1+m2)] = (-1)**(m1+n)*tmp1 * tmp2\
                                    *tmp/(2*l3+1.0)

                            row_ind += 1 

    return A, indP



def build_lsq_vector(L,numeqs,calliBs,O,Q,P):
    b = np.zeros((numeqs,1), dtype=np.complex128)
    row_ind = 0
    for n in range(-L,L+1):
        for l1 in range(0,L+1):
            for l2 in range(0,L+1):
                tmp = calliBs[n+L][l1][l2]
                LHS = np.conj((O[l1]@Q[l1]).T)@tmp@O[l2]@Q[l2]
                for m1 in range(-l1, l1+1):
                    for m2 in range(-l2,l2+1):
                        l3_l = max(abs(m1-m2), abs(l1-l2))
                        l3_u = min(l1+l2, P)
                        if l3_u >= l3_l:    
                            b[row_ind] = LHS[l1+m1,l2+m2]
                            row_ind += 1

    # b = np.vstack((np.real(b),np.imag(b)))
    return b


def run_alternating(lst_M,calliM,maxiter,initialOs,P,L,tol=1e-10):
    O = initialOs
    numeqs = (2*L+1)*(L+1)**4 #use this for the first B cost function below

    Q = generate.list_real2complex_matrix(L)
    Qstacked = generate.stacked_real2complex_matrix(L)

    #cost functions for solving for B
    Amtx, indP = build_lsq_matrix(L,P,numeqs)  #all eqns, no random weights


    residuals_B = np.zeros((maxiter,1))
    residuals_O = np.zeros((maxiter,1))

    min_cost = 100000000
    sparse = True
    # sparse = False

    QB_list = generate.list_Q_for_B(P)
    QBH = scipy.sparse.csr_matrix(np.conj(generate.stacked_Q_for_B(P).T))
    smallest_O = initialOs.copy()
    for iter in range(maxiter):
        if iter%10 == 0:
            print("Iteration number " + str(iter))

        ### For real B
        OQ = copy.deepcopy(O)
        for ell in range(len(OQ)):
            OQ[ell] = OQ[ell]@np.conj(QB_list[ell].T)
        vecB = solveB_real(O,calliM,P,L,Q,QBH,A = Amtx, indP = indP,sparse=sparse)  #all eqns, no random weights

        for ell in range(len(vecB)):
            vecB[ell] = np.conj(QB_list[ell].T)@vecB[ell]

        B = vectorB2matrixB(vecB)
        lst_B = generate_listB(B,L)


        #measure the residual
        tmp_cost = evaluate_cost_function(O,lst_B,lst_M,Qstacked)

        residuals_B[iter] = tmp_cost


        if tmp_cost < min_cost:
            smallest_O = O
            smallest_B = B
            min_cost = tmp_cost

        if tmp_cost < tol:
            break

        
        O = solve_O_relaxed(lst_B,lst_M,Qstacked,sparse=sparse)

        new_res = evaluate_cost_function(O,lst_B,lst_M,Qstacked)
        # print("Just solved for O: ", new_res)
        residuals_O[iter] = new_res


    return O, B, residuals_O, residuals_B, smallest_O, iter


def evaluate_cost_function(O,lst_B,lst_M,Qstacked):
    cost = 0
    L = len(O) - 1
    stackedO = stackO(O)
    for n in range(0,2*L+1):
        cost += np.linalg.norm(stackedO@Qstacked@lst_B[n]@np.conj((stackedO@Qstacked).T) - lst_M[n])**2

    return np.sqrt(cost)



def blockdiag(A,L):
    B = np.zeros(A.shape, dtype=np.complex128)
    for l in range(L+1):
        B[(l)**2:(l+1)**2,(l)**2:(l+1)**2] = A[(l)**2:(l+1)**2,(l)**2:(l+1)**2]

    return B

def vectorB2matrixB(vecB):
    matB = [0]*len(vecB)
    for k in range(len(vecB)):
        l = vecB[k].shape[0]
        tmp = np.zeros((l,l),dtype=np.complex128)
        tmp[:,k] = vecB[k]
        matB[k] = tmp
    return matB


def solveB_real(O,calliBs,P,L,Q,QBH,A = None, indP = None, sparse = True):
    #calliBs is a list with calliBs[n] equals to the calligraphic B^n in the notes.
    # It is an L\times L list, where
    #calliBs[n][l1][l2] is the (2l1+1)x(2l2 +1) matrix (calligraphic B)_{l1,l2}
    #O is a list of orthogonal matrices, with O[l] = O_l from the notes

    #The variable B_{p,u} has p going from p = 0 to P and u from -p to p,
    #so in total, there are \sum_{p=0}^P (2p+1) = (P+1)^2 variables
    numeqs = (2*L+1)*(L+1)**4
    if A == None:
        t = time.time()
        A, indP = build_lsq_matrix(L,P,numeqs) #maybe just use a subset of the equations. Tradeoff speed vs accuracy
        print("...built A matrix in "+str(time.time()-t)+" seconds")

    # A = A@scipy.sparse.csr_matrix(np.conj(QB.T))
    A = A@QBH

    # t = time.time()
    
    b = build_lsq_vector(L,numeqs,calliBs,O,Q,P)

    if sparse:
        A = scipy.sparse.vstack((np.real(A), np.imag(A))) 
        b = np.vstack((np.real(b), np.imag(b)))
        tmp = scipy.sparse.linalg.lsqr(A,b, atol=1e-15, btol=1e-15, show=False)[0]
    if not sparse:

        print('not sparse',A.shape, b.shape)
        tmp = np.linalg.lstsq(A.todense(),b,rcond=-1)[0]
    tmp = tmp.reshape(-1,)

    res = [1j]*(P+1)
    for p in range(0,P+1):        
        res[p] = tmp[int(indP[0,p]):int(indP[1,p])]

    return res


#### Relaxation method for solving for O
def solve_O_relaxed(lst_B, lst_M, Q, sparse=True):

    L = int((len(lst_B) - 1)/2)
    Ans = [None]*(2*L+1)
    bns = [None]*(2*L+1)
    Anssparse = [None]*(2*L+1)
    bnssparse = [None]*(2*L+1)

    indn_new = delete_col_index_for_O_new(lst_M[0].shape[0])


    for n in range(-L,L+1):
        #for lstsq
        if not sparse:
            tmp = np.kron(np.eye((L+1)**2), lst_M[n+L]) - np.kron((Q@lst_B[n+L]@np.conj(Q.T)).T, np.eye((L+1)**2))
            Ans[n+L] = tmp[:,indn_new]
            bns[n+L] = get_RHS_for_O(tmp, (L+1)**2)#lst_M[0].shape[0])

        #For lsqr
        if sparse:
            tmp = scipy.sparse.kron(scipy.sparse.identity((L+1)**2,format='csr',dtype=np.complex128), lst_M[n+L],format='csr') \
                - scipy.sparse.kron((Q@lst_B[n+L]@np.conj(Q.T)).T, scipy.sparse.identity((L+1)**2,format='csr',dtype=np.complex128),format='csr')
            Anssparse[n+L] = tmp[:,indn_new]
            bnssparse[n+L] = np.array(get_RHS_for_O(tmp,(L+1)**2,sparse=True))

    #for lstsq
    if not sparse:
        A = np.vstack(Ans)
        b = np.vstack(bns)

        #If we want to impose X to be real
        A = np.vstack((np.real(A), np.imag(A)))
        b = np.vstack((np.real(b), np.imag(b)))

        x = np.linalg.lstsq(A,b,rcond=-1)[0]


    #For lsqr
    if sparse:
        Asparse = scipy.sparse.vstack(Anssparse, dtype=np.complex128)
        bsparse = np.vstack(bnssparse)

        Asparse = scipy.sparse.vstack((np.real(Asparse), np.imag(Asparse))) 
        bsparse = np.vstack((np.real(bsparse), np.imag(bsparse)))
        x = scipy.sparse.linalg.lsqr(Asparse,bsparse,atol=1e-15, btol=1e-15, show=False)[0]


    X = reshape_for_O(x,L)

    return X



def get_RHS_for_O(A,m,sparse=False):
    b = 0
    # tmp = np.array(A.todense(),dtype=np.complex128)
    for i in range(4):
        #(i,i) are the elements we have set to 1; the corresponding
        # linear index is m*i+i, since we use zero-indexing
        b += -A[:,m*i+i]
    #For lsqr
    if sparse:
        b = np.array(b.todense())
    return np.reshape(b, [-1,1])


def reshape_for_O(x,L):

    # The below is when fixing top two blocks of O
    X = [None]*(L+1)
    X[0] = np.array([[1]])
    X[1] = np.eye(3)
    ind = 0
    for l in range(L-1):
        X[l+2] = procrustes(np.reshape(x[ind:ind+(2*l+5)**2], [2*l+5,2*l+5], order='F'))
        ind += (2*l+5)**2
        # print(X[l+2].shape)
    return X



def block_procrustes(A,L):
    B = np.zeros(A.shape, dtype=np.complex128)
    for l in range(L+1):
        B[(l)**2:(l+1)**2,(l)**2:(l+1)**2] = procrustes(A[(l)**2:(l+1)**2,(l)**2:(l+1)**2])

    return B


def delete_col_index_for_O_new(m):
    # Q is a square matrix
    # ones in index denote the indices in the deletion set
    d = m**2
    ind = []
    for k in range(d):
        i = k % m
        j = k // m 
        l = np.floor(np.sqrt(j)) # also equal to the index of block that i belongs to    
        if j > 3 and i >= l**2 and i < (l+1)**2: #when fixing top 2 blocks
            ind.append(k)

    return ind


def generate_fB(B,L,P):
    fB = np.zeros(shape = (1, (L+1)**2), dtype = np.complex128)
    for l in range(min(L+1,P+1)):
        fB[0, (l**2):((l+1)**2)] = generate.frakB(B, l)

    return fB

def listB2calliB(lst_B,L):
    calliB = [[[None for _ in range(L+1)] for _ in range(L+1)] for _ in range(2*L+1)]
    for n in range(-L,L+1):
        for l1 in range(L+1):
            for l2 in range(L+1):
                calliB[n+L][l1][l2] = lst_B[n+L][(l1**2):((l1+1)**2), (l2**2):((l2+1)**2)]
    return calliB


def generate_listB(B,L):
    # generate list of calliB for all -L\leq n\leq L:
    lst_B = generate.calliB_all_n(B, L)
    return lst_B



def generate_calliB(B,L):
    calliB = [[[None for _ in range(L+1)] for _ in range(L+1)] for _ in range(2*L+1)]
    for n in range(-L,L+1):
        for l1 in range(L+1):
            for l2 in range(L+1):
                calliB[n+L][l1][l2] = generate.calliB(B, l1, l2, n)
    return calliB


def procrustes(A):
    U,_,V = np.linalg.svd(A)
    return U@V  


