#%%
import numpy as np
from time import time
import matplotlib.pyplot as plt


#%%
def mandlebrot(width,height,r_low,r_high,i_low,i_high,max_iter):
    r_vals = np.linspace(r_low,r_high,width)
    i_vals = np.linspace(i_low,i_high,height)
    m_graph = np.ones((height,width),dtype=np.float32)
    for x in range(width):
        for y in range(height):
            c= np.complex64(r_vals[x]+i_vals[y]* 1j )
            z= np.complex64(0)
            for i in range(max_iter):
                z=z**2 +c
                if(np.abs(z) >2):
                    m_graph[x,y] = 0
                    break
    return m_graph


#%%
t1 = time()
m = mandlebrot(512,512,-2,2,-2,2,256)
t2 = time()
m_time = t2-t1
t1 = time()
plt.imshow(m,extent=(-2,2,-2,2))
plt.show()
t2 = time()
p_time= t2-t1
print(m_time,p_time)


#%%



