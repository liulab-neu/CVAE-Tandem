import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from data_loader import load_data
import CVAE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_dim = 1001
h_dim1 = 512
h_dim2 = 256
#z_dim = 50
c_dim = 1
encoder_size = [1001,1024,2048,1024,512,256,128,20]
decoder_size = [20,128,256,512,1024,2048,1024,1001]
model = CVAE.CVAE(encoder_size=encoder_size,decoder_size=decoder_size,c_dim=c_dim).to(device)
# for name, param in model.named_parameters():
#     print(name, param.shape)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size= 100 * 12, gamma=0.97)


# 定义训练函数
def train(epoch,train_loader):
    model.train()
    train_loss = 0
    print('======= Epoch:{} Train ========'.format(epoch))
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        # 这里假设标签是one-hot编码
        recon_batch, mu, log_var,z = model(data, labels)
        loss = model.loss_function(recon_batch, data, mu, log_var,labels).to(device)
        #print(f"loss:{loss},lr:{scheduler.get_last_lr()[-1]}")

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()




    print('====> Epoch: {} Average train loss: {:.4f}'.format(
        epoch, train_loss / (batch_idx+1)))
    return train_loss / (batch_idx+1)


# 定义验证函数
def validate(val_loader):
    model.eval()
    val_loss = 0
    print('======= Validation ========')
    with torch.no_grad():
        for batch_idx,(data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, log_var,z = model(data, labels)
            val_loss += model.loss_function(recon_batch, data, mu, log_var,labels).to(device).item()

    val_loss /= batch_idx+1

    print('====> Average validation loss: {:.4f}'.format(val_loss))
    return val_loss

data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')

# reshape data_b_list elements and remove first column
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]


y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)
y_related = torch.load('./data/y_related.pt')
y_mean = torch.mean(y_related, dim=1)

# Preprocessing
#y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
n_samples = len(y_mean)
min_index = np.argmin(y_mean, axis=1)

min_index = min_index.reshape(-1, 1)
#min_index_wave = (min_index/(encoder_size[0]-1))*(800-380)+380
wavelengths = np.array(data_b_list[0])[:, 0]
min_index_wave = wavelengths[min_index]
min_index_wave_scale = (min_index/(encoder_size[0]-1))

#print(min_index_wave)


# 初始化 FWHM 的数组
fwhms = np.zeros(n_samples)

# 对每个样本计算 FWHM
for i in range(n_samples):
    y_sample = y_mean[i]

    peak_value = y_sample[min_index[i]]
    #print(peak_value)
    # 计算峰值的一半
    half_max = (peak_value + 1) / 2
    print(half_max)
    # 确保 half_max 是 Tensor
    half_max = torch.tensor(half_max, dtype=torch.float32)

    # 找到左侧的半高点
    left_half_max_index = np.where(y_sample[:min_index[i]] <= half_max)[0]
    if left_half_max_index.size > 0:
        left_half_max_index = left_half_max_index[-1]
    else:
        left_half_max_index = 0

    # 找到右侧的半高点
    right_half_max_index = np.where(y_sample[min_index[i]:] >= half_max)[0]
    if right_half_max_index.size > 0:
        right_half_max_index = right_half_max_index[0] + min_index[i]
    else:
        right_half_max_index = len(y_sample) - 1

    # 计算 FWHM
    fwhms[i] = wavelengths[right_half_max_index] - wavelengths[left_half_max_index]

# 输出所有样本的 FWHM
# print("FWHMs for each sample:", fwhms)
# min_fwhm_value = np.min(fwhms)
# print(min_fwhm_value)

# labels = F.one_hot(torch.randint(0, 10, (n_samples,)), num_classes=c_dim).float()
# print(labels)
dataset = TensorDataset(y_mean, min_index_wave_scale)
n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)


#训练并验证模型
output = open('./1peak/CVAE_loss.txt', 'w')
best_val_loss = float('inf')
train_loss_total = []
val_loss_total = []
for epoch in range(1, 2000 + 1):
    train_loss = train(epoch,train_loader)
    val_loss = validate(val_loader)
    train_loss_total.append(train_loss)
    val_loss_total.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './1peak/model/cvae_best_mean.pth')
    output.write(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n")

# train_loss_total = [...]
# val_loss_total = [...]
output.close()

plt.figure(figsize=(10, 6))
plt.plot(train_loss_total, label='Train Loss')
plt.plot(val_loss_total, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


# 保存图像
plt.savefig('./1peak/loss_curve.png')

# 显示图像
plt.show()



# z_values = []
# with torch.no_grad():
#     for data, labels in dataset:
#         recon_batch, mu, log_var,z = model(data, labels)
#         z_values.append(z.detach().cpu().numpy())
# file_path = './data/z_value.pt'
# os.makedirs(os.path.dirname(file_path), exist_ok=True)
# torch.save(z_values)