# %%
import numpy as np
# %%
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
# %%
experiment_id = "KyEmeEI2SYWqMLoHOpxofw"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# %%
df = experiment.get_scalars()
df
# %%
print(df["run"].unique())
print(df["tag"].unique())
# %%
stl10_train_losses = df[df.run.str.endswith("resnet18_1/losses_train") & df.run.str.contains("stl10_cropped")]
stl10_val_accs = df[df.run.str.endswith("resnet18_1/Accuracy_val") & df.run.str.contains("stl10_cropped")]
# %%
#cifar10_train_losses.loc[:,'value'] = cifar10_train_losses.value.astype(np.float)
# %%
def get_train_loss_runs(run):
    return run.split('cropped_')[1].split('_')[0]
# %%
#cifar10_train_tiles = cifar10_train_losses.run.apply(get_train_loss_runs)
stl10_train_tiles = stl10_train_losses.run.apply(get_train_loss_runs)
# %%
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=cifar10_train_losses, x="step", y="value",
             hue=cifar10_train_tiles, palette=sns.color_palette("Set1", cifar10_train_tiles.nunique()) ).set_title("train_loss")
# %%
stl10_train_losses = stl10_train_losses.loc[:,'value'].to_numpy()
stl10_val_accs = stl10_val_accs.loc[:,'value'].to_numpy()
# %%
#c = cifar10_train_losses.reshape(-1,400)
#c = cifar10_val_accs.reshape(-1,400)
c = stl10_train_losses.reshape(-1,400)
# c = stl10_val_accs.reshape(-1,400)
# %%
plt.figure(figsize=(7,7))
plt.plot(c[0], label='16')
plt.plot(c[1], label='24')
plt.plot(c[2], label='32')
plt.plot(c[3], label='48')
#plt.plot(c[4], label='56')
#plt.plot(c[5], label='64')
#plt.plot(c[6], label='8')
plt.plot(c[7], label='96')
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
legend = plt.legend(fontsize=12)
legend.set_title("tile size",prop={'size':14})
plt.savefig('stl10_train_losses.png')
# %%
