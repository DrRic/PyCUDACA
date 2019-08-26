

#%%
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from time import time
from pycuda.elementwise import ElementwiseKernel


#%%
host_data = np.int32([1,2,3,4])
print(host_data)
    


#%%
gpu_ker = ElementwiseKernel(
"int *in, int *out",
"out[i] = 2*in[i]",
"gpu_ker"
)


#%%
dev_data = gpuarray.to_gpu(host_data)
dev_data_X2 = gpuarray.empty_like(dev_data)
gpu_ker(dev_data,dev_data_X2)
from_dev = dev_data_X2.get()
print(from_dev)
    
    





#%%
