import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import CVAE_2p
import matplotlib.pyplot as plt
from data_loader_2P import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths based on this file location (so it works no matter where you `cd` to)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cstdata_10496', 'data')


x_dim = 1001
h_dim1 = 512
h_dim2 = 256
z_dim = 50
c_dim = 2
encoder_size = [1001,1024,2048,1024,512,256,128,20]
decoder_size = [20,128,256,512,1024,2048,1024,1001]
model = CVAE_2p.CVAE(encoder_size=encoder_size,decoder_size=decoder_size,c_dim=c_dim)
model.to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'cvae_best.pth'), map_location=device))

def sample(c):
    # 生成一个随机向量z
    z = torch.randn(3, encoder_size[-1]).to(device)
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


data_a_list, data_b_list = load_data(DATA_DIR)

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
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
np.savetxt(os.path.join(BASE_DIR, 'data', 'min_wave.txt'), tensor_np, fmt='%.5f', delimiter='\t')
for i in range(len(min_index_wave)):
    if min_index_wave[i] == 532.0400:
        print(i)
        print(" ")


def spec(peak1,peak2,spec):
    # 选择一个条件向量
    wl = np.array(data_b_list[0])[:, 0]
    peak1 = (peak1 - 380) / (800 - 380)
    peak2 = (peak2 - 380) / (800 - 380)
    a = torch.tensor([[peak1]]).to(device)
    b = torch.tensor([[peak2]]).to(device)
    c = torch.cat((a,b),1).to(device)
    with torch.no_grad():
        model.eval()
        #spe, mu, log_var = model(b, c)
        spe = sample(c).to(device)
        for i in range(len(spe)):
            # Debug plot (disabled by default to avoid blocking in batch/headless runs)
            # Set to True if you explicitly want to visualize every generated spectrum.
            PLOT_EACH_SPE = False
            if PLOT_EACH_SPE:
                plt.figure()
                plt.plot(wl, spec, label='true')
                plt.plot(wl, spe[i].cpu().numpy(), label='prediction')
                plt.xlabel('Wavelength')
                plt.ylabel('Intensity')
                plt.legend()
                plt.show()
            else:
                plt.close('all')
    #print(spe)
    min_index = np.argmin(spe.cpu().numpy(), axis=1)

    min_index = min_index.reshape(-1, 1)
    print(min_index)
    wl = np.array(data_b_list[0])[:, 0]
    min_index_wave = wl[min_index]
    min_index_wave = torch.tensor(min_index_wave)
    print(min_index_wave)
    return spe

# a = 406.8800
# # spec(a)
# with torch.no_grad():
#     model.eval()
#     peak1 = 500
#     peak1 = (peak1 - 380) / (800 - 380)
#     peak2 = 450
#     peak2 = (peak2 - 380) / (800 - 380)
#     peak1 = torch.tensor([[peak1]])
#     peak2 = torch.tensor([[peak2]])
#     z = torch.randn(1, encoder_size[-1])
#     a = torch.cat((peak1, peak2), dim=1)
#     sample_a = model.decoder(z, a)
#     #sample_b = model.decoder(z, b)
#     plt.figure()
#     plt.plot(wl, sample_a.squeeze(0), label='500,450')
#     #plt.plot(wl, sample_b.squeeze(0), label='406.5985')
#     plt.xlabel('Wavelength')
#     plt.ylabel('Intensity')
#     plt.legend()
#     plt.show()
#     model.eval()


