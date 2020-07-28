import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn')

res = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/'
                        'ISIC_2020/run_exp/efficient-siimd/kaggle_skin_6/efficient-siimd_final.model')['plot_stuff']

train_loss = res[0]
val_loss = res[2]
train_auc = res[1]['auc']
val_auc = res[3]['auc']

fig = plt.figure(figsize=(6, 4.2))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax.plot(range(len(train_loss)), train_loss, color="indianred", linewidth=1.6, label="loss_train")
ax.plot(range(len(val_loss)), val_loss, color="steelblue", linewidth=1.6, label="loss_val")

ax2.plot(range(len(train_auc)), train_auc, color="indianred", linewidth=1.5, label="train auc")
ax2.plot(range(len(val_auc)), val_auc, color="steelblue", linewidth=1.5, label="val auc")
ax2.set_ylim([0.7, 1.0])
ax.set_xlabel("epoch")
ax.set_ylabel("loss")

ax.legend(loc='lower left')
ax2.legend(loc='upper right')
plt.title('single model training results')
plt.tight_layout()
# plt.savefig('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/ISIC_2020/run_exp/efficient-siimd/kaggle_skin_5/train.png', dpi=300)
plt.show()