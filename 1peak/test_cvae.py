import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import CVAE
import matplotlib.pyplot as plt
from data_loader import load_data


x_dim = 1001
h_dim1 = 512
h_dim2 = 256
z_dim = 50
c_dim = 1
encoder_size = [1001,1024,2048,1024,512,256,128,20]
decoder_size = [20,128,256,512,1024,2048,1024,1001]
model = CVAE.CVAE(encoder_size=encoder_size,decoder_size=decoder_size,c_dim=c_dim)
model.load_state_dict(torch.load('./model/cvae_best_mean.pth'))

def sample(c):
    # 生成一个随机向量z
    z = torch.randn(3, encoder_size[-1])
    print(z)
    tensor_list = []
    # 用模型的解码器生成样本
    for i in range(len(z)):
        sample = model.decoder(z[i].unsqueeze(0), c)
        tensor_list.append(sample)
    merged_tensor = torch.cat(tensor_list, dim=0)
    return merged_tensor




# 生成样本

wlmin = 380
wlmax = 800
num = 1001




#wl = torch.linspace(wlmin,wlmax,num)


data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')

# reshape data_b_list elements and remove first column
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]


y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)

# Preprocessing
#y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
n_samples = len(y)
min_index = np.argmin(y, axis=1)

min_index = min_index.reshape(-1, 1)
print(min_index)
#print(min_index[160])
wl = np.array(data_b_list[0])[:, 0]
min_index_wave = wl[min_index]
min_index_wave = torch.tensor(min_index_wave)
print(min_index_wave)
# 转换为NumPy数组
tensor_np = min_index_wave.numpy()

# 保存为文本文件
np.savetxt('./data/min_wave.txt', tensor_np, fmt='%.5f', delimiter='\t')
for i in range(len(min_index_wave)):
    if min_index_wave[i] == 532.0400:
        print(i)
        print(" ")


def spec(a,b):
    # 选择一个条件向量
    wl = np.array(data_b_list[0])[:, 0]
    a = (a - 380) / (800 - 380)
    c = torch.tensor([[a]])
    with torch.no_grad():
        model.eval()
        #spe, mu, log_var = model(b, c)
        spe = sample(c)
        for i in range(len(spe)):
            #print(spe)
            plt.figure()
            plt.plot(wl,b,label='true')
            plt.plot(wl,spe[i],label='prediction')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()
    #print(spe)
    min_index = np.argmin(spe, axis=1)

    min_index = min_index.reshape(-1, 1)
    print(min_index)
    wl = np.array(data_b_list[0])[:, 0]
    min_index_wave = wl[min_index]
    min_index_wave = torch.tensor(min_index_wave)
    print(min_index_wave)
    return spe

def target_trans(wlmin=380, wlmax=800, num=1001, plot=False):

    wl = np.linspace(wlmin, wlmax, num) #生成等间距1*num个点
    #spe = np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)) # ture or false
    #spe = np.logical_not(np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)))
    # spe = np.logical_not(np.logical_and(wl >= 532, wl < 533))
    # spe = spe.astype(np.float) #转化为float类型,532
    spe = np.logical_not(np.logical_and(wl >= 520, wl < 535 ))
    spe = spe.astype(float) #转化为float类型,532
    return torch.from_numpy(spe)

def lorentzian_trans(center,wlmin=380, wlmax=800, num=1001, gamma=15, plot=False):
    wl = np.linspace(wlmin, wlmax, num)
    #center = 532
    #spe = gamma / ((wl - center) ** 2 + gamma ** 2)  # Lorentzian function
    spe = 1 / (1 + ((wl - center) / gamma)**2)
    spe = (spe-1) * -1

    if plot:
        plt.plot(wl, spe)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Intensity')
        plt.show()

    return torch.from_numpy(spe).float()

a = 406.8800
# spec(a)
with torch.no_grad():
    model.eval()
    a = 649.6103
    a = (a - 380) / (800 - 380)
    b = 406.5985
    b = (b - 380) / (800 - 380)
    a = torch.tensor([[a]])
    b = torch.tensor([[b]])
    z = torch.randn(1, encoder_size[-1])
    sample_a = model.decoder(z, a)
    sample_b = model.decoder(z, b)
    sample_c = lorentzian_trans(center = 650)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(wl, sample_a.squeeze(0), label='CTS', linewidth=2)
    ax.plot(wl, sample_c, label='ITS', linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transmittance', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.legend(fontsize=10, frameon=True, prop=dict(weight='bold'))
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout(pad=1.2)
    fig.savefig('./spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.show()
    plt.close(fig)
    model.eval()


