
#%%
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from time import time
from pycuda.compiler import SourceModule

    
#%%
def set_module():
    return SourceModule("__global__ void scalar_multiply_kernel( int *outvect, int scalar,int *vec){int j = threadIdx.x;outvect[j] =  scalar*vec[j];}")

#%%
ker = set_module()
smg = ker.get_function("scalar_multiply_kernel")
t_vec = np.random.randint(low = 0, high = 3, size = 1500).astype(np.int32)
t_vec_gpu = gpuarray.to_gpu(t_vec)
o_vec = gpuarray.empty_like(t_vec_gpu)
smg(o_vec,np.int32(2),t_vec_gpu,block=(150,1,1),grid=(int(1500/150),1,1))
print(np.allclose(o_vec.get(),2*t_vec))

#%%
print(o_vec.get())

print(2*t_vec)

#%%
