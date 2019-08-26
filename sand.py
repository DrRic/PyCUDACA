#%%
def sum(n):
    sum = 0
    for i in range(1,n+1):
        sum+=i
    print(sum)
    print(int(((n**2)+n)/2))
    print(int((n*(n+1)/2)))

#%%
sum(100)

#%%
from functools import reduce
a = [1,2,31,4]
print(reduce(lambda x,y:max(x,y),a))

#%%
