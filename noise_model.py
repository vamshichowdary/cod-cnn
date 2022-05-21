# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
c10_accs_ts = np.array([8,12,16,18,24,30,32])
c10_accs_r18 = np.array([60.73, 85.93, 91.87, 92.91, 93.28, 92.42, 89.88])
c10_accs_r9 = np.array([75.24, 87.44, 91.55, 90.95, 92.38, 89.1, 90.39])

c100_accs_ts_r9 = np.array([8,12,16,18,24,30,32])
c100_accs_r9 = np.array([20.04, 55.41, 61.72, 49.15, 68.72, 61.36, 68.74])
c100_accs_ts_r18 = np.array([8,12,16,24,32])
c100_accs_r18 = np.array([22.64, 53.53, 65.13, 72.78, 64.01])

s10_accs_ts_r18 = np.array([8,16,24,32,48,56,64,96])
s10_accs_r18 = np.array([24.79, 53.79, 68.89, 79.29, 80.78, 80.04, 79.95, 71.86])
s10_accs_ts_r9 = np.array([8,16,24,32,48,56,96])
s10_accs_r9 = np.array([38.22, 52.09, 65.34, 71.28, 78.00, 80.19, 80.51])

im200_accs_ts = np.array([16,24,32,48,56,64])
im200_accs = np.array([21.67, 47.98, 57.30, 57.91, 57.16, 51.45])

im1k_accs_ts = np.array([56, 112, 168, 196, 224, 256])
im1k_accs = np.array([32.49, 63.31, 70.82, 71.27, 70.95, 69.95])
im1k_accs_top5 = np.array([57.17, 85.21, 89.86, 89.99, 89.87, 89.13])
# %%
class noise_model_params:
    def __init__(self):
        self.h = None
        self.N = None
        self.k = None
        self.alpha = 3.0
    
    @property
    def input_resolution(self):
        return self.h
    
    @input_resolution.setter
    def input_resolution(self, value):
        self.h = value
        if type(value) == int:
            self.t = np.arange(0.1, self.h, 0.1)
# %%
class cifar10(noise_model_params):
    """
    Args:
        s (int): stride to calculate effective number of tiles per image
    """
    def __init__(self, s):
        super().__init__()

        self.input_resolution = 32
        self.N = 50000
        self.k = 10
        self.s = s

class cifar100(noise_model_params):
    """
    Args:
        s (int): stride to calculate effective number of tiles per image
    """
    def __init__(self, s):
        super().__init__()
        
        self.input_resolution = 32
        self.N = 50000
        self.k = 100
        self.s = s

class stl10(noise_model_params):
    """
    Args:
        s (int): stride to calculate effective number of tiles per image
    """
    def __init__(self, s):
        super().__init__()
        
        self.input_resolution = 96
        self.N = 5000
        self.k = 10
        self.s = s

class imagenet200(noise_model_params):
    """
    Args:
        s (int): stride to calculate effective number of tiles per image
    """
    def __init__(self, s):
        super().__init__()
        
        self.input_resolution = 64
        self.N = 100000
        self.k = 200
        self.s = s

class imagenet1k(noise_model_params):
    """
    Args:
        s (int): stride to calculate effective number of tiles per image
    """
    def __init__(self, s):
        super().__init__()
        
        self.input_resolution = 256
        self.N = 1200000
        self.k = 1000
        self.s = s
# %%
def gen_err(t,h,N,s,k,alpha):
    """
    Args:
        t (np.array): tile size
        h (int): original image size
        N (int): number of training samlples in dataset
        s (int): stride to calculate effective number of tiles per image
        k (int): number of classes
        alpha (float): parameter to control effective dimension of tiles in mesh norm
    """
    #m1 = lambda x : -np.log(x)             ## grad norm dependence
    m1 = lambda x:1./x 
    m2 = np.sqrt                    ## class dependence of training error
    m3 = lambda x : -np.log(x)      ## tile area ratio dependence of training error

    t_eff = ( ((h-t)/s)+1 )**2      ## effective number of tiles in a single image

    def mesh_norm(const=1.):
        d_eff = alpha / (3 * (t**2))
        return const / ( (N * t_eff) ** d_eff )

    def grad_norm(const=1.):
        d = 1 / (3 * (t**2))
        return const * m1( ((t)/h)**d )

    def train_err(const=1.):
        t_err = const * m2( k ) * m3(t/h)
        return t_err
        #t_err = const * grad_norm()
        # if D == 0:
        #     return t_err
        # else:
        #     nd = np.where((N*t_eff/D)-1 > 0, (N*t_eff/D)-1, 0)
        #     return t_err * np.sqrt( nd / (N*t_eff/D) )
    
    return  (mesh_norm()*grad_norm() + train_err(const=0.5)) / np.sqrt(t_eff)
    # return  (train_err(const=0.5)) / np.sqrt(t_eff)
    #return  (grad_norm()) / np.sqrt(t_eff)
# %%
c10 = cifar10(s=1)
c100 = cifar100(s=1)
s10 = stl10(s=5)
im200 = imagenet200(s=3)
im1k = imagenet1k(s=4)

im1k_512 = imagenet1k(s=8)
im1k_512.input_resolution = 512
# %%
c10_err = gen_err(**(c10.__dict__))
c100_err = gen_err(**(c100.__dict__))
s10_err = gen_err(**(s10.__dict__))
im200_err = gen_err(**(im200.__dict__))
im1k_err = gen_err(**(im1k.__dict__))
im1k_512_err = gen_err(**(im1k_512.__dict__))
# %%
plt.figure(figsize=(10,8))

# plt.plot(c10.t, c10_err, label='cifar10 noise model', color='orangered')
# plt.plot(c10_accs_ts, (100-c10_accs_r9)/100, '--o', label='cifar10 experimental - resnet9', color='orangered')
# plt.plot(c10_accs_ts, (100-c10_accs_r18)/100, ':o', label='cifar10 experimental - resnet18', color='orangered')

# plt.plot(c100.t, c100_err, label='cifar100 noise model', color='dodgerblue')
# plt.plot(c100_accs_ts_r9, (100-c100_accs_r9)/100, '--o', label='cifar100 experimental - resnet9', color='dodgerblue')
# plt.plot(c100_accs_ts_r18, (100-c100_accs_r18)/100, ':o', label='cifar100 experimental - resnet18', color='dodgerblue')

# plt.plot(s10.t, s10_err, label='noise model', color='g')
# plt.plot(s10_accs_ts_r9, (100-s10_accs_r9)/100, '--o', label='experimental - resnet9', color='g')
# plt.plot(s10_accs_ts_r18, (100-s10_accs_r18)/100, ':o', label='experimental - resnet18', color='g')

# plt.plot(im200.t, im200_err, label='noise model', color='c')
# plt.plot(im200_accs_ts, (100-im200_accs)/100, '--o', label='experimental - resnet34', color='c')

plt.plot(im1k.t, im1k_err, label='noise model, resolution=256', color='m')
plt.plot(im1k_512.t, im1k_512_err, label='noise model, resolution=512', color='thistle')
plt.plot(im1k_accs_ts, (100-im1k_accs)/100, '--o', label='experimental - resnet18, resolution=256', color='m')
#plt.plot(im1k_accs_ts, (100-im1k_accs_top5)/100, ':o', label='experimental - resnet18, resolution=256', color='m')

#plt.xticks(np.arange(0,512,10))
plt.legend()
plt.ylim([0,1])

plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
plt.legend(fontsize=14)
#legend = plt.legend(fontsize=14, loc='upper right',bbox_to_anchor=(0.85, 1))

plt.savefig('imnet-1k.png')
# %%

# %%

# %%
s10_32_accs_ts = np.array([8,16,32])
s10_32_accs = np.array([48.55, 69.18, 70.93])
# %%
s10_32 = stl10(s=2)
s10_32.input_resolution = 32
# %%
s10_32_err = gen_err(**(s10_32.__dict__))
# %%
plt.figure(figsize=(10,8))

plt.plot(s10.t, s10_err, label='noise model, resolution=96', color='g')
plt.plot(s10_accs_ts_r9, (100-s10_accs_r9)/100, '--o', label='experimental, resolution=96', color='g')
#plt.plot(s10_accs_ts_r18, (100-s10_accs_r18)/100, ':o', label='experimental - resnet18', color='g')


plt.plot(s10_32.t, s10_32_err, label='noise model, resolution=32', color='orangered')
plt.plot(s10_32_accs_ts, (100-s10_32_accs)/100, '--o', label='experimental, resolution=32', color='orangered')
plt.legend()
plt.ylim([0,1])
plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
#plt.legend(fontsize=14)
legend = plt.legend(fontsize=14, loc='upper right',bbox_to_anchor=(0.85, 1))
plt.savefig('stl10_diff_res_curve.png')
# %%
