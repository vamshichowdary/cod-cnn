# %%
from flops import add_flops_counting_methods
from torchvision.models import resnet18
# %%
fcn = resnet18(num_classes=1000)
# %%
import torch
# %%
x = torch.randn(1024,3,224,224)
# %%
x = x.to('cuda')
# %%
fcn.to('cuda')
# %%
fcn = add_flops_counting_methods(fcn)
fcn = fcn.cuda().train()

fcn.start_flops_count()
# %%
_ = fcn(x) # works now
# %%
fcn.compute_average_flops_cost() 
# %%
