# %%
import numpy as np
# %%
h = 32
t = 1
N = 50000
alpha = 1
# %%
def mesh_norm(N, t, h):
    return 1/(N*(h-t+1)**2)**(1/(3*(t**2)))
# %%
def grad_norm(t, h, alpha):
    return 1 / (t/h)**(alpha/(3*(t**2)))
# %%
def tile_noise(t,h):
    return -np.log(t**2 / h**2)
# %%
def 