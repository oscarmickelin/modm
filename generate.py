import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import factorial,lpmv
from sympy.physics.quantum.cg import CG
from sympy import N
from scipy.io import loadmat
from aspire.basis.basis_utils import sph_bessel


from fle_2d import FLEBasis2D
from fast_cryo_pca import FastPCA
import utils_cwf_fast_batch as utils
import aspire.utils.rotation
from aspire.source.simulation import Simulation
from aspire.volume import Volume
from aspire.operators import ScalarFilter
from aspire.operators import RadialCTFFilter
import scipy.special as spl

import cvxpy as cp
from aspire.noise import CustomNoiseAdder


from scipy.stats import vonmises_fisher 
import gurobipy
import os
import copy

import scipy.optimize

import torch
import torch_harmonics as th
import mrcfile
import os
import gzip
import shutil
import urllib.request

def fetch_emdb(emdb_id):

    url = 'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-'+str(emdb_id)+'/map/emd_'+str(emdb_id)+'.map.gz'
    target_folder = 'structures/'+str(emdb_id)
    if not(os.path.isdir(target_folder)):
        os.makedirs(target_folder)
    else:
        print('Folder exists')

    target_gz = target_folder+'/'+str(emdb_id)+'map.gz'
    target_map = str(emdb_id)+'/'+str(emdb_id)+'.map'

    if not(os.path.exists(target_gz)):
        urllib.request.urlretrieve(url, target_gz)
    else:
        print('... .gz exists')


    if not(os.path.exists(target_map)):
        with gzip.open(target_gz, 'rb') as f_in:
            with open(target_map, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    else:
        print('...... .map exists')

    with mrcfile.open(target_map) as mrc:
        return mrc.data

def load_data(s, fetch=0):
    if fetch == 0:
        data=loadmat("structures/"+s)
        return data['V']
    else:
        return fetch_emdb(s[4:])

def rot_matrix_to_axis(vectors, angles):
    Rs = np.zeros((len(angles),3,3))
    for i in range(len(angles)):
        v3 = vectors[i].T
        # randomly generate v2 
        #  to v3
        v2_tmp = np.random.rand(3)
        v2_tmp = v2_tmp - np.inner(v2_tmp, v3)*v3
        v2 = v2_tmp/np.linalg.norm(v2_tmp)
        # v1 equals to the cross product of v2 and v3
        v1 = np.cross(v2, v3)
        R1 = np.transpose(np.array([v1, v2, v3]))
        # randomly generate the inplane rotation angle theta
        theta = angles[i]
        R2 = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta),0], [0, 0, 1]])
        Rs[i,:,:] = (R1@R2)

    return Rs




def generate_moment(V,angles,pixel_size, sn_ratio, batch_size, num_imgs, defocus_ct, eps):
    #angles = None gives uniform distribution
    # For non-uniform, use generate_random_rotations
    # in https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/utils/rotation.py


    #######################################
    ## Parameters
    #######################################
    img_size = V.shape[0]


    # Number of defocus groups.
    # Specify the CTF parameters

    voltage = 300  # Voltage (in KV)
    defocus_min = 1e4  # Minimum defocus value (in angstroms)
    defocus_max = 3e4  # Maximum defocus value (in angstroms)

    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    # create CTF indices for each image, e.g. h_idx[0] returns the CTF index (0 to 99 if there are 100 CTFs) of the 0-th image
    h_idx = utils.create_ordered_filter_idx(num_imgs, defocus_ct)
    dtype = np.float32

    # Create filters. This is a list of CTFs. Each element corresponds to a UNIQUE CTF
    h_ctf = [
        RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]



    # We prefer that our various arrays have consistent dtype.
    vols = Volume(V.astype(dtype))

    vols = vols.downsample(img_size)
    vols = utils.mask_volume(vols, img_size, radius=img_size//2)

    # Create a simulation object with specified filters and the downsampled 3D map

    # this is for generating CTF-affected clean projections.
    # We use this to determine the noise variance so that our simulated images have targeted SNR
    source_ctf_clean = Simulation(
        L=img_size,
        n=num_imgs,
        vols=vols,
        offsets=0.0,
        amplitudes=1.0,
        unique_filters=h_ctf,
        filter_indices=h_idx,
        dtype=dtype,
    )

    # determine noise variance to create noisy images with certain SNR
    noise_var = utils.get_noise_var_batch(source_ctf_clean, sn_ratio, batch_size)

    # create noise filter
    noise_filter = CustomNoiseAdder(noise_filter = ScalarFilter(dim=2, value=noise_var))


    # create simulation object for noisy images
    source = Simulation(
        L=img_size,
        n=num_imgs,
        vols=vols,
        unique_filters=h_ctf,
        filter_indices=h_idx,
        offsets=0.0,
        amplitudes=1.0,
        dtype=dtype,
        angles=angles,
        noise_adder=noise_filter,
    )

    # Fourier-Bessel expansion object
    fle = FLEBasis2D(img_size, img_size, eps=eps)
    
    # get clean sample mean and covariance
    mean_clean = utils.get_clean_mean_batch(source, fle, batch_size)
    covar_clean = utils.get_clean_covar_batch(source, fle, mean_clean, batch_size, dtype)

    # options for covariance estimation
    options = {
        "whiten": False,
        "single_pass": False, # whether estimate mean and covariance together (single pass over data), not separately
        "noise_var": noise_var, # noise variance
        "batch_size": batch_size,
        "dtype": dtype
    }

    # create fast PCA object
    fast_pca = FastPCA(source, fle, options)

    # # two passes over data for covariance estimation
    mean_est = fast_pca.estimate_mean()
    _, covar_est = fast_pca.estimate_mean_covar(mean_est=mean_est)

    err_mean = np.linalg.norm(mean_clean-mean_est)/np.linalg.norm(mean_clean)
    _, err_covar = utils.compute_covar_err(covar_est, covar_clean)

    print(err_mean, err_covar)


    return mean_est, covar_est, fle


def get_Al_tilde_analytical(V,angles,pixel_size,L, r_grid, phi_grid, sn_ratio, batch_size, num_imgs, defocus_ct, eps,J_fle):
    mean_est, covar_est, fle = generate_moment(V,angles,pixel_size, sn_ratio, batch_size, num_imgs, defocus_ct, eps)

    m2_for_kam = m2_cr2pf_analytical(mean_est, covar_est, fle, pixel_size, r_grid, phi_grid)

    A = kam(m2_for_kam,L,fle.L, phi_grid,r_grid,J_fle)

    return A, mean_est, fle




def kam(m2_for_kam,L,N,phi_grid,r_grid,J_fle):
    A = [0]*(L+1)
    phi_grid_size = len(phi_grid)

    weight = np.pi/(phi_grid_size)
    for l in range(L+1):
        print(l)
        C = 0
        for pind in range(phi_grid_size):
            p = phi_grid[pind]
            
            if p <= np.pi:  #want the integral from 0 to pi
                if pind % 2 == 0:
                    C += 2/3*weight*m2_for_kam[pind,:,:]*spl.eval_legendre(l,np.cos(p))*np.sin(p)
                else:
                    C += 4/3*weight*m2_for_kam[pind,:,:]*spl.eval_legendre(l,np.cos(p))*np.sin(p)

        C *= (2*l+1)/(4*np.pi)

        Csym = (C + np.conj(C).T)/2

        lam, V = np.linalg.eig(Csym)
        lam[lam < 0] = 0

        idx = lam.argsort()[::-1]   
        lam = lam[idx]
        V = V[:,idx]

        A[l] = np.real(V[:,:2*l+1]@np.sqrt(np.diag(lam[:2*l+1])))

    return A 




def f_complex(n,k,r,t,lam):
    return 2*np.sqrt(np.pi)*(-1)**k*(-1j)**n*lam*spl.jv(n,2*np.pi*r)/((2*np.pi*r)**2 - lam**2)*np.exp(1j*n*t)

def f_real(n,k,r,t,lam):
    if n == 0:
        tmp = f_complex(n,k,r,t,lam)
    elif n > 0:
        tmp = np.sqrt(2)*np.real(f_complex(n,k,r,t,lam))
    else:
        tmp = np.sqrt(2)*np.imag(f_complex(n,k,r,t,lam))
    tmp = tmp*(-1j)**n

    return tmp


def m2_cr2pf_analytical(mean_est, covar_est, fle, voxel_size, r_grid, phi_grid):

    path_to_module = os.path.dirname(__file__)
    zeros_path = os.path.join(path_to_module, "jn_zeros_n=3000_nt=2500.mat")
    data = loadmat(zeros_path)
    lmds = data["roots_table"]
    m2 = np.zeros((len(phi_grid),len(r_grid),len(r_grid)), dtype=np.complex128)


    for ind in range(len(covar_est)):
        n = int((-1)**ind*np.floor((ind+1)/2))
        c = copy.deepcopy(covar_est[ind])
        klen = c.shape[0]
        if n == 0:
            c = c + mean_est[:klen].reshape(-1,1)@np.conj(mean_est[:klen].reshape(1,-1))

        tmp = 0
        for k1 in range(klen):
            lmd1 = lmds[np.abs(n),k1]
            for k2 in range(klen):
                lmd2 = lmds[np.abs(n),k2]
                tmp += c[k1,k2]*f_complex(n,k1+1,r_grid.reshape(-1,1),0,lmd1)*np.conj(f_complex(n,k2+1,r_grid.reshape(1,-1),0,lmd2))

        for tind in range(len(phi_grid)):
            t = phi_grid[tind]
            angt = (-1j)**n*np.exp(1j*n*t)
            if n == 0:
                ang = np.real(angt*tmp)
            elif n > 0:
                ang = 2*np.real(angt*tmp)*np.real((-1j)**n)
            else:
                ang = 2*np.imag(angt*tmp)*np.imag((-1j)**n)

            m2[tind,:,:] += ang

    m2 /= (fle.h)**2
    return m2


def m2_cr2pf_analytical_complex(mean_est, covar_est, fle, voxel_size, r_grid, phi_grid):

    path_to_module = os.path.dirname(__file__)
    zeros_path = os.path.join(path_to_module, "jn_zeros_n=3000_nt=2500.mat")
    data = loadmat(zeros_path)
    lmds = data["roots_table"]
    m2 = np.zeros((len(phi_grid),len(r_grid),len(r_grid)), dtype=np.complex128)


    for ind in range(len(covar_est)):
        n = int((-1)**ind*np.floor((ind+1)/2))
        c = copy.deepcopy(covar_est[ind])
        klen = c.shape[0]
        if n == 0:
            c += mean_est[:klen].reshape(-1,1)@np.conj(mean_est[:klen].reshape(1,-1))
        tmp = 0
        for k1 in range(klen):
            lmd1 = lmds[np.abs(n),k1]
            for k2 in range(klen):
                lmd2 = lmds[np.abs(n),k2]
                tmp += c[k1,k2]*f_complex(n,k1+1,r_grid.reshape(-1,1),0,lmd1)*np.conj(f_complex(n,k2+1,r_grid.reshape(1,-1),0,lmd2))

        for tind in range(len(phi_grid)):
            t = phi_grid[tind]
            angt = np.exp(1j*n*t)
            if n == 0:
                ang = np.real(angt*tmp)
            elif n > 0:
                ang = 2*np.real(angt*tmp)
            else:
                ang = 2*np.imag(angt*tmp)

            m2[tind,:,:] += ang

    m2 /= (fle.h)**2
    return m2





def list_real2complex_matrix(L):
    res = [None]*(L+1)
    for l in range(L+1):
        res[l] = real2complex_matrix(l)
        
    return res

def stacked_real2complex_matrix(L):
    res = np.zeros(((L+1)**2, (L+1)**2), dtype=np.complex128)
    for l in range(L+1):
        res[(l)**2:(l+1)**2,(l)**2:(l+1)**2] = real2complex_matrix(l)

    return res

def real2complex_matrix(l):

    A = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
    for m in range(2*l+1):
        if m < l:
            A[m,m] = -1j/np.sqrt(2)
            A[m,2*l-m] = 1/np.sqrt(2)
        if m > l:
            A[m,m] = 1/np.sqrt(2)*(-1)**(m-l)
            A[m,2*l-m] = 1j/np.sqrt(2)*(-1)**(m-l)

    A[l,l] = 1
    if l%2 == 1:
        A = 1j*A

    A = np.conj(A.T)
    return A

def Q_for_B(l):
    A = np.zeros((2*l+1, 2*l+1), dtype=np.complex128)
    for m in range(2*l+1):
        if m < l:
            A[m,m] = -1j/np.sqrt(2)
            A[m,2*l-m] = 1/np.sqrt(2)
        if m > l:
            A[m,m] = 1/np.sqrt(2)*(-1)**(m-l)
            A[m,2*l-m] = 1j/np.sqrt(2)*(-1)**(m-l)

    A[l,l] = 1
    A = np.conj(A.T)


    return A


def list_Q_for_B(L):
    res = [None]*(L+1)
    for l in range(L+1):
        res[l] = Q_for_B(l)
        
    return res

def stacked_Q_for_B(L):
    res = np.zeros(((L+1)**2, (L+1)**2), dtype=np.complex128)
    for l in range(L+1):
        res[(l)**2:(l+1)**2,(l)**2:(l+1)**2] = Q_for_B(l)

    return res


def makeJmatrix_fle(rs,fle,L):
    J = [None]*(L+1)
    for l in range(L+1):
        tmp = fle.lmds[fle.idlm_list[l][0]]
        rmat = rs.reshape(-1,1) * tmp.reshape(1,-1)
        radial_ell = np.zeros_like(rmat, dtype=np.complex128)
        for ik in range(0, radial_ell.shape[1]):
            radial_ell[:, ik] = sph_bessel(l, rmat[:, ik])
            
        nrm = np.reshape(fle.cs[fle.idlm_list[l][0]], [1,-1])
        radial_ell = radial_ell * nrm
        J[l] = radial_ell

    return J


def makeAmatrix_fle(Abessel,fle,L,J):
    A = np.zeros((J[0].shape[0], (L+1)**2), dtype = np.complex128)
    for l in range(0,L+1):
        A[:, (l**2):((l+1)**2)]= J[l]@Abessel[:np.max(fle.ks[fle.idlm_list[l][0]]), (l**2):((l+1)**2)]

    return A


def Ahelp(A,lst_B,L):
    Ahelp = [1j]*(2*L+1)

    for n in range(-L,L+1):
        Ahelp[n+L] = A@lst_B[n+L]@np.conj(A.T)
    return Ahelp

def formula2moment(Ahelp,p1,p2,L):    
    m2 = 0
    for n in range(-L,L+1):
        m2 += np.exp(1j*n*(p1-p2))*Ahelp[n+L]
    return m2




def moment_tilde2G(m2_tilde,L,Alt_stacked,phi_grid, w,r_grid):
    Gs = [None]*(2*L+1)
    tmps = [None]*(2*L+1)
    for n in range(-L,L+1):
        print('moment2G, ', n)
        G = np.zeros(((L+1)**2, (L+1)**2), dtype=np.complex128)
        tmp = 0
        n_phi = len(phi_grid)
        for pind in range(n_phi):
            p = phi_grid[pind]
            tmp += w[pind]*m2_tilde[pind,:,:]*np.exp(-1j*n*p)

        tmp = np.diag(r_grid)@tmp@np.diag(r_grid)
        tmpA = np.diag(r_grid)@copy.deepcopy(Alt_stacked)
        ## use the fact that many blocks of G should be zero to regularize
        G = optimize_pinv_explicit(tmp, tmpA, n, L)

        Gs[n+L] = G
        tmps[n+L] = tmp

    return Gs,tmps






def optimize_pinv_explicit(tmp, A, n, L):
    M,b = build_pinv_matrix(A,tmp,n,L)
    x = scipy.optimize.lsq_linear(M,b,tol=1e-14)
    xv = rebuild_pinv_matrix(x['x'],A,n,L)
    return xv


def build_pinv_matrix(A,rhs,n,L):
    M = np.kron(np.conj(A),A)
    MR = copy.deepcopy(M)
    MI = copy.deepcopy(M)
    xshape = [A.shape[1], A.shape[1]]
    to_del = []

    for i in range(0,A.shape[1]):
        for j in range(i+1,A.shape[1]):
            ind1 = np.ravel_multi_index((i,j),xshape,order='F')
            ind2 = np.ravel_multi_index((j,i),xshape,order='F')
            MR[:,ind1] += MR[:,ind2]
            MI[:,ind1] -= MI[:,ind2]
            to_del.append(ind2)

    for l1 in range(0,np.abs(n)):
        tmp1 = [i for i in range(l1**2,(l1+1)**2) for j in range(0,xshape[1])]
        tmp2 = [j for i in range(l1**2,(l1+1)**2) for j in range(0,xshape[1])]
        tmp = np.ravel_multi_index(np.array([tmp1,tmp2]),xshape,order='F')
        to_del = np.concatenate([to_del,tmp])

    for l2 in range(0,np.abs(n)):
        tmp1 = [i for i in range(0,xshape[1]) for j in range(l2**2,(l2+1)**2)]
        tmp2 = [j for i in range(0,xshape[1]) for j in range(l2**2,(l2+1)**2)]
        tmp = np.ravel_multi_index(np.array([tmp1,tmp2]),xshape,order='F')
        to_del = np.concatenate([to_del,tmp])

    for l1 in range(np.abs(n)+1, L+1, 2):
        tmp1 = [i for i in range(l1**2,(l1+1)**2) for j in range(0,xshape[1])]
        tmp2 = [j for i in range(l1**2,(l1+1)**2) for j in range(0,xshape[1])]
        tmp = np.ravel_multi_index(np.array([tmp1,tmp2]),xshape,order='F')
        to_del = np.concatenate([to_del,tmp])

    for l2 in range(np.abs(n)+1, L+1, 2):
        tmp1 = [i for i in range(0,xshape[1]) for j in range(l2**2,(l2+1)**2)]
        tmp2 = [j for i in range(0,xshape[1]) for j in range(l2**2,(l2+1)**2)]
        tmp = np.ravel_multi_index(np.array([tmp1,tmp2]),xshape,order='F')
        to_del = np.concatenate([to_del,tmp])



    MR = np.delete(MR, to_del, axis=1)
    MI = np.delete(MI, to_del, axis=1)

    M = np.vstack( (np.hstack((np.real(MR),-np.imag(MI))), np.hstack((np.imag(MR),np.real(MI)))) )

    b = np.vstack( (np.real(rhs).reshape(-1,1,order='F'), np.imag(rhs).reshape(-1,1,order='F')) ).reshape(-1,)
    
    return M,b

def rebuild_pinv_matrix(x,A,n,L):
    cut = int(x.shape[0]/2)
    xr = x[:cut]
    xi = x[cut:]
    xv = np.zeros((A.shape[1],A.shape[1]), dtype=np.complex128)

    i = 0
    j = 0

    #put in zeros
    ind = 0
    while ind < cut:
        should_be_zero = 0
        if i < (np.abs(n))**2 or j < (np.abs(n))**2:
            should_be_zero = 1
        
        for l1 in range(np.abs(n)+1, L+1, 2):
            if (l1**2 <= i) and (i < (l1+1)**2):
                should_be_zero = 1

            if (l1**2 <= j) and (j < (l1+1)**2):
                should_be_zero = 1

        if should_be_zero:
            pass
        else:
            xv[i,j] = xr[ind] + 1j*xi[ind]
            ind += 1
        
        if i == j:
            i = 0
            j += 1
        else:
            i += 1

    xv = xv + np.conj(np.triu(xv,1).T)

    return xv



def optimize_pinv(tmp, A, n, L):
    s = A.shape[1]
    x = cp.Variable((s,s), complex=True)
    objective = cp.Minimize(cp.sum_squares(A @ x @ np.conj(A.T) - tmp))

    constraints = [x == cp.conj(x.T)]
    for l1 in range(0,np.abs(n)):
       constraints += [x[(l1)**2:(l1+1)**2,:].flatten() == 0]

    for l2 in range(0,np.abs(n)):
       constraints += [x[:,(l2)**2:(l2+1)**2].flatten() == 0]

    for l1 in range(np.abs(n)+1, L+1, 2):
        constraints += [x[(l1)**2:(l1+1)**2,:].flatten() == 0]

    for l2 in range(np.abs(n)+1, L+1, 2):
        constraints += [x[:,(l2)**2:(l2+1)**2].flatten() == 0]

    prob = cp.Problem(objective, constraints)
    env = gurobipy.Env()
    env.setParam('BarConvTol', 1e-12)
    result = prob.solve(solver=cp.GUROBI,verbose=False,env=env)
    return x.value


def moment2G(Ahelp,rs,L,n_phi):
    Gs = [None]*(2*L+1)
    for n in range(-L,L+1):
        G = np.zeros((rs.shape[0], rs.shape[0]), dtype=np.complex128)
        tmp = 0
        p1s = np.linspace(0,2*np.pi, n_phi)
        p2s = np.linspace(0,2*np.pi, n_phi)
        for p1 in p1s:
            for p2 in p2s:
                tmp += formula2moment(Ahelp, p1, p2, L)*np.exp(-1j*n*(p1-p2))
        G = tmp/(n_phi**2)
        Gs[n+L] = G

    return Gs

def formula2moment_vec(Ahelp,dp,L):    
    m2 = 0
    for n in range(-L,L+1):
        m2 += Ahelp[n+L].reshape(-1,1)@np.exp(1j*n*dp).reshape(1,-1)
    return m2

def moment2G_vec(Ahelp,rs,L,n_phi):
    Gs = [None]*(2*L+1)
    p1s = np.linspace(0,2*np.pi, n_phi, endpoint=False)
    p2s = np.linspace(0,2*np.pi, n_phi, endpoint=False)
    ###
    dp_unique = np.concatenate((-np.flip(p1s), p1s[1:])).reshape(1,-1)
    mom = formula2moment_vec(Ahelp, dp_unique, L)
    counts = np.concatenate((list(range(1,n_phi+1)), np.flip(list(range(1,n_phi))))).reshape(1,-1)
    for n in range(-L,L+1):
        tmp = np.reshape(np.sum(mom*np.exp(-1j*n*dp_unique)*counts.reshape(1,-1), axis=1), (rs.shape[0], rs.shape[0]))
        G = tmp/(n_phi**2)
        Gs[n+L] = G

    return Gs

def G2M(G,A):
    epsilon = 1e-3
    tmp = np.linalg.pinv(A)
    res = [None]*len(G)
    for n in range(len(G)):
        res[n] = tmp@G[n]@np.conj(tmp.T)

    return res




## Go from c to A
def c2A_fle(fle,c):
    ind = 0
    ind_ang = 0
    ind_radial = 0

    c = c.reshape(-1,)
    global_k_max = max(fle.ks)
    A = np.zeros(
        shape=(global_k_max, (fle.lmax+1)**2), dtype=c.dtype
    )
    
    for ell in range(0, fle.lmax + 1):
        for m in range(-ell, ell + 1):
            md = 2*np.abs(m) - ( m < 0)
            k_max = max(fle.ks[fle.idlm_list[ell][md]])
            leftover = global_k_max - k_max
            idx_radial = ind_radial + np.arange(0, k_max)
            idx = ind + np.arange(0, len(idx_radial))
            A[:,ind_ang:ind_ang+1] = np.vstack((np.array([c[idx]]).T, np.zeros((leftover,1)))) #for each l,m these are the indices, so stack them for each l. varying m, padding columns with zeros so they all have the same length
            ind += len(idx)
            ind_ang += 1

        ind_radial += len(idx_radial)
    return A

def A2c_fle(fle,A):
    ind_ang = 0
    c_un = np.empty(0)
    print(A.shape)
    for ell in range(0, fle.lmax +1):
        for m in range(-ell, ell + 1):
            md = 2*np.abs(m) - ( m < 0)
            k_max = max(fle.ks[fle.idlm_list[ell][md]])
            c_un = np.concatenate((c_un,  A[0:k_max, ind_ang:ind_ang+1].flatten()))
            ind_ang += 1
    return c_un





def cN(n, l):
    """
    (n,l) requires -l \leq n \leq l, l \equiv n (mod 2) to be nonzero
    N(n, l) = (-1)^n N(-n, l)
    Output N_l^n
    """ 
    if abs(n) > l or (l-n) % 2 == 1:
        return 0
    else:
         return lpmv(n, l, 0) * np.sqrt((2*l+1)/4/np.pi) * np.sqrt(factorial(l-n) / factorial(l+n))





def generating_mixture_vonMises(means, covs, pis):
    """
    Inputs: means means of Gaussians
            covs covariance matrices of Gaussians
            pis probabilities of choosing each Gaussian
    Output: a random vector on the unit sphere according to mixture of vonMises distritions
    """
    means = means
    covs = covs
    pis = pis
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
    assert np.isclose(acc_pis[-1], 1)
    
    r = np.random.uniform(0, 1)
    # select Gaussian
    k = 0
    for i, threshold in enumerate(acc_pis):
        if r < threshold:
            k = i
            break
    selected_mean = means[k]
    selected_cov = covs[k]

    tmp = np.linalg.inv(selected_cov)
    mu = tmp @ (selected_mean.reshape(3,))
    k = np.linalg.norm(mu)
    a = vonmises_fisher(mu / k, k)
    x = a.rvs(1).reshape(3,)
    return x
    


def R_inplane_invariant(v):
    """
    Inputs: v point on the unit sphere representing the viewing direction
    Output: randomly generate a rotation according to inplane invariant distribution
    """
    v3 = v
    # randomly generate v2 perpendiculat to v3
    v2_tmp = np.random.rand(3)
    v2_tmp = v2_tmp - np.inner(v2_tmp, v3)*v3
    v2 = v2_tmp/np.linalg.norm(v2_tmp)
    # v1 equals to the cross product of v2 and v3
    v1 = np.cross(v2, v3)
    R1 = np.transpose(np.array([v1, v2, v3]))
    # randomly generate the inplane rotation angle theta
    theta = np.random.uniform(0, 2*np.pi)
    R2 = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta),0], [0, 0, 1]])
    
    return R1@R2



def rho(theta,phi,means,covs,pis):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    v = np.array([[x,y,z]])
    val = 0
    for j in range(len(means)):
        selected_mean = means[j]
        selected_cov = covs[j]
        tmp = np.linalg.inv(selected_cov)
        mu = tmp @ (selected_mean.reshape(3,))
        k = np.linalg.norm(mu)
        vmf = vonmises_fisher(mu / k, k)
        val += pis[j]*vmf.pdf(v.reshape(3,))

    return np.squeeze(val)



def B_inplane_invariant(P,means,covs,pis, n_theta=50):
    n_phi = n_theta + 1
    sht = th.RealSHT(n_theta, n_phi, lmax=P + 1, grid="equiangular", csphase=True).to("cpu")
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    cost, _ = th.quadrature.clenshaw_curtiss_weights(n_theta, -1, 1)
    theta = np.flip(np.arccos(cost))
    rho_mat = np.zeros((n_theta, n_phi))

    for i in range(n_theta):
        for j in range(n_phi):
            rho_mat[i,j] = rho(theta[i], phi[j],means,covs,pis)

    B = sht(torch.DoubleTensor(rho_mat).to("cpu")).cpu().numpy()
    B = np.conj(B)
    B = B[:P+1,:P+1]

    Bmats = [0]*(P+1)


    for p in range(P+1):
        Bmats[p] = np.zeros((2*p+1,2*p+1),dtype=np.complex128)
        Bmats[p][p:,p] = B[p,:p+1]
        Bmats[p][:p,p] = np.conj(B[p,1:p+1][::-1])*((-1)**np.arange(1,p+1))*(-1)**(p+1)
        up = np.arange(-p, p+1)
        Bmats[p][:,p] = Bmats[p][:,p]*(2*p+1)*(-1)**np.abs(up)*np.sqrt(4*np.pi/(2*p+1))

    return Bmats





def visualize_distr(B,npoints=100,filename='test'):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    P = len(B) - 1
    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1, projection='3d')


    u = np.linspace( 0, 2 * np.pi, npoints)
    v = np.linspace( 1/100, np.pi - 1/100, npoints )

    # create the sphere surface
    XX = 1 * np.outer( np.cos( u ), np.sin( v ) )
    YY = 1 * np.outer( np.sin( u ), np.sin( v ) )
    ZZ = 1 * np.outer( np.ones( np.size( u ) ), np.cos( v ) )

    WW = np.zeros(XX.shape, dtype=np.complex128)
    inte = 0
    for i in range( len( XX ) ):
        for j in range( len( XX[0] ) ):
            x = XX[ i, j ]
            y = YY[ i, j ]
            z = ZZ[ i, j ]
            v3 = np.array([x,y,z]).reshape(3,)
            R_matrix = R_inplane_invariant(v3)
            R = Rotation.from_matrix(R_matrix)
            R_angles = R.as_euler('zyz')
            for p in range(P+1):
                for up in range(2*p+1):
                    tmp2 = (-1)**(up-p)*np.sqrt(4*np.pi/(2*p+1))*np.conj(spl.sph_harm(up-p, p, R_angles[2], R_angles[1]))
                    WW[ i, j ] +=  B[p][up,p]*tmp2
                inte += WW[i,j]*np.sin(v[j])
    myheatmap = np.real(WW)

    myheatmap = myheatmap / np.max(np.abs(myheatmap.flatten()))

    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.YlGnBu( myheatmap ) )

    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(filename+"P"+str(P)+".png", bbox_inches='tight')

    return myheatmap




def B_norm(B):
    L = len(B)
    res = [0 for i in range(L)] 
    for l in range(L):
        res[l] = np.linalg.norm(B[l][:,l]) #/np.sqrt(2*l+1)
    return(res)


def frakB(B, l):
    """
    Inputs: B a list of length P where P is band limit;
            l;
    Output: frak B_l row vector of length (2*l + 1)
    """
    loadL = 20 #replace by largest precomputed file in the folder
    loadP = 40 #replace by largest precomputed file in the folder
    N_array = np.load('precomputed_data/precomputed_N_L_'+str(loadL)+'.npy')
    P = len(B) - 1
    fB = np.zeros(shape = (1,2*l+1), dtype = complex) 
    for m in range(-l, l+1):
        # fB[0, l+m] = cN(0, l) * B[l][m+l, l] / (2*l + 1)
        fB[0, l+m] = N_array[loadL,0] * B[l][m+l, l] / (2*l + 1)
    return(fB)



def calliC(l1, l2, m1, m2, n1, n2, l3):
    """
    calligraphic_C(l1, l2, m1, m2, n1, n2, l3) =  C(l1, m1, l2, m2, l3, m1+m2) 
                                                    * (l1, n1, l2, n2, l3, n1+n2)

    Inputs: l1, l2, m1, m2, n1, n2, l3
    Output: calliC outputs calligraphic C in sympy class
            calliC_enum outputs calligraphic C in numpy float64 
    """
    C1 = CG(l1, m1, l2, m2, l3, m1+m2)
    C2 = CG(l1, n1, l2, n2, l3, n1+n2)
    C = C1 * C2
    return(C.doit())



def calliC_enum(l1, l2, m1, m2, n1, n2, l3):
    """
    calligraphic_C(l1, l2, m1, m2, n1, n2, l3) =  C(l1, m1, l2, m2, l3, m1+m2) 
                                                    * (l1, n1, l2, n2, l3, n1+n2)

    Inputs: l1, l2, m1, m2, n1, n2, l3
    Output: calliC outputs calligraphic C in sympy class
            calliC_enum outputs calligraphic C in numpy float64 
    """
    C1 = CG(l1, m1, l2, m2, l3, m1+m2)
    C2 = CG(l1, n1, l2, n2, l3, n1+n2)
    C = C1 * C2
    C_enum = N(C.doit())
    return(np.float64(C_enum))



def calliB_all_n(B, L, loadL = 10, loadP = 20):
    """
    Inputs: B a list of length P where P is band limit;
            l1;
            l2;
            n = n1 = -n2
    Output: calligraphic B^n_(l1, l2) matrix of dim (2*l1 + 1) * (2*l2 + 1)
    """
    
    cL = np.load('precomputed_data/precomputed_ClebschL_L_'+str(loadL)+'_P_'+str(loadP)+'.npy')
    cR = np.load('precomputed_data/precomputed_ClebschR_L_'+str(loadL)+'_P_'+str(loadP)+'.npy')
    N_array = np.load('precomputed_data/precomputed_N_L_'+str(loadL)+'.npy')

    P = len(B) - 1
    lst_B = list(range(2*L+1))
    cB = np.zeros(shape = ((L+1)**2,(L+1)**2,2*L+1), dtype = np.complex128)
    for l1 in range(L+1):
        for l2 in range(L+1):
            #faster
            tmp_N1 = N_array[loadL-L:loadL+L+1,l1]
            tmp_N2 = N_array[loadL-L:loadL+L+1,l2]
            for m1 in range(-l1, l1+1):
                for m2 in range(-l2, l2+1):
                    l3_l = max(abs(m1-m2), abs(l1-l2))
                    l3_u = min(l1+l2, P)
        
                    tmp_c = cL[l1,m1+loadL,l2,m2+loadL,l3_l:l3_u+1]*cR[l1,loadL-L:loadL+L+1,l2,l3_l:l3_u+1]
                    vec = np.array([B[l3][-m1+m2+l3, l3] / (2*l3+1) for l3 in range(l3_l,l3_u+1)]).reshape(-1,)
                    vec2 = ((-1)**abs(m1+np.arange(loadL-L,loadL+L+1))).reshape(-1,1)
                    cB_tmp = np.sum(vec2*tmp_c*vec,axis=1)
                    # cB_tmp = 0
                    # for l3 in range(l3_l, l3_u+1):
                    #     #fastest
                    #     tmp_c = cL[l1,m1+loadL,l2,m2+loadL,l3]*cR[l1,loadL-L:loadL+L+1,l2,l3]
                    #     tmp = (-1)**abs(m1+np.arange(loadL-L,loadL+L+1)) * tmp_c * B[l3][-m1+m2+l3, l3] / (2*l3+1)
                    #     cB_tmp = cB_tmp + tmp 


                    cB[(l1**2)+m1+l1, (l2**2)+m2+l2,:] = cB_tmp * tmp_N1 * tmp_N2

    for n in range(2*L+1):
        lst_B[n] = cB[:,:,n]

    return(lst_B)





def NG_matrices(n,l1,l2,l3):
    """
    return the matrix involving the products of cN and CG coefficients for calliB(n,l1,l2) for l2\geq l1\geq n\geq 0 
        and each l3\geq |l1-l2|
    The dim should be (2l1+1)*(2l2+1)
    \sum_{|l1-l2|\leq l3\leq |l1+l2|} NG_matrix \circ normalB_matrix gives calliB(n,l1,l2)   
    """
    if l3 < abs(l1-l2):
        return "Error: l3 must be greater than abs(l1-l2)"
    else:
        CG_mat = np.zeros((2*l1 + 1,2*l2 + 1))
        for m1 in range(2*l1 + 1):
            for m2 in range(2*l2 + 1):
                CG_mat[m1][m2] = calliC(l1, l2, m1-l1, l2-m2, n, -n, l3) 
        NG_mat = CG_mat * cN(n, l1) * cN(n, l2) / (2*l3 + 1)
        return NG_mat        
        









