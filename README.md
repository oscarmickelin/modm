# The method-of-double-moments for cryo-EM

Code accompanying the paper https://arxiv.org/abs/2511.07438


Download the files at https://github.com/yunpeng-shi/fast-cryoEM-PCA
Download the files:
https://github.com/nmarshallf/fle_2d/blob/main/src/fle_2d/jn_zeros_n%3D3000_nt%3D2500.mat
https://github.com/oscarmickelin/fle_3d/blob/main/jl_zeros_l%3D3000_k%3D2500.mat
and put them in the same folder as fast-cryoEM-PCA
Install fle_3d
Install https://github.com/RuiyiYang/BOTalign/
Install (including obtaining a license file) gurobi using the instructions at https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python


## Installation
```bash
conda create --name modm python=3.9 pip
conda activate modm
pip install aspire
pip install "numpy<2" scipy finufft torch==1.12.0 torch_harmonics==0.6.3 fle-2d sympy cvxpy
pip3 install "cvxpy[GUROBI]"
```
