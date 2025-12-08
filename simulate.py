import solve, generate
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from scipy.ndimage import zoom

from aspire.volume import Volume
from utils_BO import align_BO
from aspire.utils.rotation import Rotation

import matplotlib.pyplot as plt
import finufft
from fle_3d import FLEBasis3D

import os
from scipy.io import loadmat



def to_angular_order_fle(fle, a):
    a_ordered = np.zeros_like(a)
    ind = 0
    for l in range(0,fle.lmax + 1):
        for m in range(-l, l+1):
            md = 2*np.abs(m) - ( m < 0)
            tmp = fle.idlm_list[l][md]
            if l % 2 == 0:
                a_ordered[ind:ind+len(tmp)] = np.real(a[tmp])
            else:
                a_ordered[ind:ind+len(tmp)] = np.imag(a[tmp])
            ind += len(tmp)
    return a_ordered


def to_eigen_order_fle(fle, a):
    a_ordered = np.zeros_like(a)
    ind = 0
    for l in range(0,fle.lmax + 1):
        for m in range(-l, l+1):
            md = 2*np.abs(m) - ( m < 0)
            tmp = fle.idlm_list[l][md]
            if l % 2 == 0:
                a_ordered[tmp] = a[ind:ind+len(tmp)]
            else:
                a_ordered[tmp] = 1j*a[ind:ind+len(tmp)]
            ind += len(tmp)
    return a_ordered


def run_simulation_with_images(M,L,P,tol=1e-10):

    print('setting parameters')

    sn_ratio = 0.1

    batch_size = 20000
    num_imgs = 100000
    defocus_ct = 20000

    eps = 1e-6


    num_langevins = 8
    means = [None]*num_langevins
    for k in range(num_langevins):
        means[k] = np.random.randn(3,)
        means[k] = means[k]/np.linalg.norm(means[k])

    covs = [0.05*np.diag(np.ones(3,dtype=np.float64))]*num_langevins
    pis = [0]*num_langevins


    pis = np.random.rand(1,num_langevins).T
    pis = pis/np.sum(pis)


    print('generating rotations')


    dir_samples = np.array([generate.generating_mixture_vonMises(means, covs, pis ) for _ in range(num_imgs)])
    angle_samples = np.random.uniform(0.0, 2*np.pi, num_imgs)

    sampled_rotations = generate.rot_matrix_to_axis(dir_samples, angle_samples)

    angles_nu = Rotation(sampled_rotations).angles


    print('computing B coefficients')
    B = generate.B_inplane_invariant(P, means, covs, pis, n_theta=4*P)

    print('loading data')
    #with mrcfile.open("./structures/emd_32743.map") as mrc:
    #    V = mrc.data
    #pixel_size = 0.832*512/M

    with mrcfile.open("./structures/emd_2660.map") as mrc:
        V = mrc.data
    pixel_size = 1.34*360/M


    print('expanding FLEBasis')
    V = zoom(V, (M/V.shape[0],M/V.shape[1],M/V.shape[2]))
    V = V/np.max(V)
    x = np.linspace(-1,1,M,endpoint=False)*2*np.pi/2
    X,Y,Z = np.meshgrid(x,x,x)
    Vf = finufft.nufft3d1(Y.flatten(),X.flatten(),Z.flatten(), np.complex128(V).flatten(), (M,M,M), isign=-1, eps = 1e-6)

    fle3 = FLEBasis3D(M,int(1.0*M),1e-9,max_l=L+1, mode="real")


    c_fle = fle3.expand(Vf)

    print('...done')
    cc_fle = to_angular_order_fle(fle3, c_fle)
    Atrunc = generate.c2A_fle(fle3,cc_fle)


    print('truncating structure')

    # ##### #truncate the structure for the moment calculation
    Vftrunc = fle3.evaluate(c_fle)
    x = np.linspace(-1,1,M,endpoint=False)*2*np.pi/2

    X,Y,Z = np.meshgrid(x,x,x)
    Vband = (1/M**3)*finufft.nufft3d2(Y.flatten(),X.flatten(),Z.flatten(), np.complex128(Vftrunc), isign=1, eps = 1e-9)
    V = Vband.reshape([M,M,M])
    # #####



    initialOs = [0]*(L+1)

    #initialize O
    for l in range(L+1):
        #random initialization for O
        tmp = np.random.randn((2*l+1),(2*l+1))
        U,_,VT = np.linalg.svd(tmp)
        initialOs[l] = U@VT


    num_r = 2*M
    r_grid = np.linspace(0,1,num_r)
    r_eval = np.linspace(0,1,num_r)*M/4


    phi_grid_size = M
    phi_grid = np.linspace(0, np.pi, 2*phi_grid_size, endpoint=False)

    print('creating J-matrices')

    J_fle = generate.makeJmatrix_fle(r_grid,fle3,L)


    print('computing uniform moment')

    Alt,mean_unif,_ = generate.get_Al_tilde_analytical(V,None,pixel_size,L, r_eval, phi_grid, sn_ratio, batch_size, num_imgs, defocus_ct, eps,J_fle)


    path_to_module = os.path.dirname(__file__)
    zeros_path = os.path.join(path_to_module, "jn_zeros_n=3000_nt=2500.mat")
    data = loadmat(zeros_path)
    lmds = data["roots_table"]

    klen = len(mean_unif)

    tmp = 0
    for k1 in range(min(klen,lmds.shape[1])):
        lmd1 = lmds[0,k1]
        tmp += mean_unif[k1]*generate.f_complex(0,k1+1,r_eval.reshape(-1,1),0,lmd1)
    mean_unif = tmp
    ### possibly change sign of Alt[0]s
    s1 = np.sum(Alt[0].flatten()*mean_unif.flatten()[:len(Alt[0].flatten())])
    if s1 < 0:
        Alt[0] = -Alt[0]
    ##
    for l in range(len(Alt)):
        Alt[l] = Alt[l]*M**(2.5)*np.pi


    Qstacked = generate.stacked_real2complex_matrix(L)
    Alt_stacked = np.zeros((num_r, (L+1)**2), dtype=np.complex128)

    for l in range(L+1):
        Alt_stacked[:, l**2:(l+1)**2] = Alt[l]

    phi_grid2 = np.linspace(0, 2*np.pi, 1*phi_grid_size, endpoint=False)
    w = np.ones(len(phi_grid2))/(len(phi_grid2))

    mean_est, covar_est, fle = generate.generate_moment(V,angles_nu,pixel_size, sn_ratio, batch_size, num_imgs, defocus_ct, eps)

    m2_tilde = generate.m2_cr2pf_analytical(mean_est, covar_est, fle, pixel_size, r_eval, phi_grid2)

    m2_tilde = m2_tilde / ( (fle3.h**2) * 1/M**2 )


    Astacked = generate.makeAmatrix_fle(Atrunc@Qstacked,fle3,L,J_fle)

    lst_M_tmp = solve.generate_listB(B,L)
    Ahelp = generate.Ahelp(Astacked,lst_M_tmp,L)


    Astacked_measured = Alt_stacked
    lst_M,tmps = generate.moment_tilde2G(m2_tilde, L, Astacked_measured, phi_grid2, w, r_grid)
    calliM = solve.listB2calliB(lst_M,L)


    Os,Bnew, residuals_O, residuals_B, smallest_O, iter = solve.run_alternating(lst_M,calliM,maxiter,initialOs,P,L,tol=tol)


    plt.figure()
    plt.semilogy(residuals_O, label='residual after solving for O')
    plt.semilogy(residuals_B, label='residual after solving for B')
    plt.legend()
    plt.xlabel('Iteration number')
    plt.ylabel('Residual at given iteration')
    plt.savefig('simulation_results/residual_plot_'+str(sn_ratio)+'_L='+str(L)+'_M='+str(M)+'_P='+str(P)+'_maxiter='+str(maxiter)+'.pdf')


    suffix = ''

    rJ_fle = [1j]*len(J_fle)
    for l in range(len(J_fle)):
        rJ_fle[l] = r_grid.reshape(-1,1)*J_fle[l]
        
    err = postprocess(L,P,Os,smallest_O,initialOs,\
                        r_grid.reshape(-1,1)*Astacked_measured,fle3,rJ_fle,Atrunc,M,suffix,maxiter,sn_ratio)


    return err




def postprocess(L,P,Os,smallest_O,initialOs,Astacked_measured,fle3,J_fle,Atrunc,M,suffix,maxiter,sn_ratio):

    for l in range(min(L,P)+1):
        Os[l] = solve.procrustes(np.real(Os[l]))
        smallest_O[l] = solve.procrustes(np.real(smallest_O[l]))
    
    print("Evaluating result")


    Osmallest = np.zeros(((L+1)**2, (L+1)**2),dtype=np.float64)
    for l in range(L+1):
        Osmallest[l**2:(l+1)**2, l**2:(l+1)**2] = smallest_O[l]

    tmp = Astacked_measured@Osmallest
    Asmallest = np.zeros((np.max(fle3.ks),(L+1)**2),dtype=np.complex128)

    for l in range(L+1):
        print(np.linalg.pinv(J_fle[l]).shape, np.linalg.cond(J_fle[l]))
        Asmallest[:np.max(fle3.ks[fle3.idlm_list[l][0]]),l**2:(l+1)**2] = np.linalg.pinv(J_fle[l])@tmp[:,l**2:(l+1)**2]

    Vtrunc = fle3.evaluate(to_eigen_order_fle(fle3,generate.A2c_fle(fle3,Atrunc)))
    Vsmallest = fle3.evaluate(to_eigen_order_fle(fle3,generate.A2c_fle(fle3,Asmallest)))
    Vsmallest_minus = fle3.evaluate(-to_eigen_order_fle(fle3,generate.A2c_fle(fle3,Asmallest)))

    x = np.linspace(-1,1,M,endpoint=False)*2*np.pi/2
    X,Y,Z = np.meshgrid(x,x,x)

    Vtrunc = (1/M**3)*finufft.nufft3d2(Y.flatten(),X.flatten(),Z.flatten(), np.complex128(Vtrunc), isign=1, eps = 1e-6).reshape([M,M,M])
    Vsmallest = (1/M**3)*finufft.nufft3d2(Y.flatten(),X.flatten(),Z.flatten(), np.complex128(Vsmallest), isign=1, eps = 1e-6).reshape([M,M,M])
    Vsmallest_minus = (1/M**3)*finufft.nufft3d2(Y.flatten(),X.flatten(),Z.flatten(), np.complex128(Vsmallest_minus), isign=1, eps = 1e-6).reshape([M,M,M])


    print("norms", np.linalg.norm(Vtrunc), np.linalg.norm(Vsmallest), np.linalg.norm(Vsmallest_minus))

    Vtrunc = np.float32(Vtrunc)
    Vsmallest = np.float32(Vsmallest)
    Vsmallest_minus = np.float32(Vsmallest_minus)

    Vtrunc = Vtrunc/np.linalg.norm(Vtrunc)
    Vsmallest = Vsmallest/np.linalg.norm(Vsmallest)
    Vsmallest_minus = Vsmallest_minus/np.linalg.norm(Vsmallest_minus)

    print("... result evaluated")
    print("Aligning result")

    params = ['wemd',min(M,128),500,True]
    print(np.squeeze(Vtrunc).shape, np.squeeze(Vsmallest).shape)
    [_,R_rec] = align_BO(Volume(np.squeeze(Vtrunc)),Volume(np.squeeze(Vsmallest)),params)
    Valigned = Volume(np.squeeze(Vsmallest)).rotate(Rotation(R_rec)).to_vec().reshape(M,M,M)

    [_,R_rec_flipped] = align_BO(Volume(np.squeeze(Vtrunc)),Volume(np.squeeze(Vsmallest)).flip(),params)
    Valigned_flipped = Volume(np.squeeze(Vsmallest)).flip().rotate(Rotation(R_rec_flipped)).to_vec().reshape(M,M,M)


    print('errors', np.linalg.norm(Valigned - Vtrunc), np.linalg.norm(Valigned_flipped - Vtrunc))

    Vres = Valigned
    err = np.linalg.norm(Valigned - Vtrunc)/np.linalg.norm(Vtrunc)
    print(err)

    if np.linalg.norm(Valigned_flipped - Vtrunc) < np.linalg.norm(Vres - Vtrunc):
            Vres = Valigned_flipped
            err = np.linalg.norm(Valigned_flipped - Vtrunc)/np.linalg.norm(Vtrunc)
    print(err)


    with mrcfile.new('simulation_results/trunc_res_'+str(sn_ratio)+'_L='+str(L)+'_M='+str(M)+'_P='+str(P)+'_maxiter='+str(maxiter)+str(suffix)+'.mrc', overwrite=True) as mrc:
        mrc.set_data(np.float32(np.real(np.squeeze(np.float32(Vres)))))
    with mrcfile.new('simulation_results/ground_trunc_'+str(sn_ratio)+'_L='+str(L)+'_M='+str(M)+'_P='+str(P)+'_maxiter='+str(maxiter)+str(suffix)+'.mrc', overwrite=True) as mrc:
        mrc.set_data(np.float32(np.real(np.squeeze(np.float32(Vtrunc)))))


    return err


def run_sweep(Ms, Ls, maxiter, tol):
    e = np.zeros((len(Ms), len(Ls)))
    Mind = 0
    for M in Ms:
        Lind = 0
        for L in Ls:
            print('Running sweep for L = ', L, ' and M = ', M)
            P = 2*L
            e[Mind, Lind] = run_simulation_with_images(M,L,P,tol=tol)
            Lind += 1
        Mind += 1
    return e

if __name__ == '__main__':
    Ms = [64]
    Ls = [5]
    maxiter=1500
    tol = 1e-10
    e = run_sweep(Ms, Ls, maxiter, tol=tol)
    print(e)
    plt.figure()
    plt.imshow(e)
    plt.colorbar()

    plt.savefig('simulation_results/sweep.pdf')

