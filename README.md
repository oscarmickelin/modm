# The method-of-double-moments for cryo-EM

Code accompanying the paper https://arxiv.org/abs/2511.07438


## Installation
Run the following commands
```bash
#create conda environment
conda create --name modm python=3.9 pip
conda activate modm

#Install dependencies
pip install aspire  fle-2d sympy cvxpy
pip3 install "cvxpy[GUROBI]"
```

Install fle_3d by following the instructions at https://github.com/oscarmickelin/fle_3d

Install BOTalign by following the instructions at https://github.com/RuiyiYang/BOTalign/

Install fast-cryoEM-PCA by following the instructions at https://github.com/yunpeng-shi/fast-cryoEM-PCA

Install (including obtaining a license file) gurobi using the instructions at https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python

Download the files:

https://github.com/nmarshallf/fle_2d/blob/main/src/fle_2d/jn_zeros_n%3D3000_nt%3D2500.mat

https://github.com/oscarmickelin/fle_3d/blob/main/jl_zeros_l%3D3000_k%3D2500.mat

https://github.com/oscarmickelin/fle_3d/blob/main/cs_l%3D3000_k%3D2500.mat

and put them in the same folder as fast-cryoEM-PCA
