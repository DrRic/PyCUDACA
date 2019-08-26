
#%%
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from time import time
from pycuda.compiler import SourceModule


#%%
host_data = np.float32(np.random.random(50000000))
    
#%%
def set_module():
    return SourceModule("__global__ void scalar_multiply_kernel( float *outvect, float scalar,float *vec){int i = threadIdx.x;outvect[i] =  scalar*vec[i];}")

#%%
ker = set_module()
smg = ker.get_function("scalar_multiply_kernel")
t_vec = np.random.rand(1048).astype(np.float32)
t_vec_gpu = gpuarray.to_gpu(t_vec)
o_vec = gpuarray.empty_like(t_vec_gpu)
smg(o_vec,np.float32(2),t_vec_gpu,block=(1048,1,1),grid=(1,1,1))
print(np.allclose(o_vec.get(),2*t_vec))

#%%
def big_calc_ker(host_data):
    t1 = time()
    for _ in range(10):
        host_x2 = host_data * np.float32(2)
    t2 = time()
    print(t2-t1)
    dev_data = gpuarray.to_gpu(host_data)
    dev_data_X2 = gpuarray.empty_like(dev_data)
    gpu_ker(dev_data,dev_data_X2)
    t1 = time()
    for _ in range(10):
        gpu_ker(dev_data,dev_data_X2)
    t2 = time()
    print(t2-t1)
    from_dev = dev_data_X2.get()
    #gpuarray.free()
    
    


#%%
for _ in range(10):
    big_calc_ker(host_data)


#%%



