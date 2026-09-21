import torch
import torch.nn as nn
import tandem
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
import matplotlib.pyplot as plt
import webbrowser

data_a_list, data_b_list = load_data('D:/Yuxiao Li/batch3')
data_a_array = []
for data in data_a_list:
    # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
    values = [data['R'], data['n_host']]
    data_a_array.append(values)
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
X = torch.tensor(data_a_array, dtype=torch.float32)
y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)
num = len(y)
idx = np.arange(num)
np.random.shuffle(idx)
r = 0.8
train_idx = idx[:int(num*r)]
test_idx = idx[int(num*r):]
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
y_related = torch.load('./data/y_related.pt')
y_mean = torch.mean(y_related, dim=1)
y_train_relate = y_mean[train_idx]
y_test_relate = y_mean[test_idx]

# Make predictions
n_input = 2  # 12
n_classes = 1001  # 3
fnn_size = [n_input, 64, 128, 256, 512, 1024,2048,1024, n_classes]
inn_size = [n_classes,1024,2048,1024,512,256,128,64, n_input]
# The 1peak dataset runs R 2.5-6.5 and n 1.5-3.05; this said 6/3.
structure_min = torch.tensor([2.5, 1.5])
structure_max = torch.tensor([6.5, 3.05])
tandem_net = tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size,training=False,structure_max=structure_max,structure_min=structure_min)
tandem_net.restore_FNN('C:/Users/liyux/OneDrive - Northeastern University/tandem/1peak/model/DNN_tandem_FNN_label.ckpt')
model = tandem_net.fnn



# 定义后处理函数，用于找出最低峰的波长
def find_lowest_peak(spectrum):
    min_index = torch.argmin(spectrum, dim=1)  # 找出最低峰的索引
    #min_index_wave = (min_index.float() / (1001 - 1)) * (800 - 380) + 380
    min_index_wave = (min_index.float() / (1001 - 1))
    #min_index_wave = min_index_wave.requires_grad_(True)
    return min_index_wave.unsqueeze(-1)  # 将min_index_wave转换为标量

def find_lowest_peak_value(spectrum):
    min_value = torch.min(spectrum, dim=1).values  # 找出最低峰的索引
    return min_value.unsqueeze(-1)  # 将min_index_wave转换为标量


def find_lowest_peak_width(spectrum, threshold_ratio=0.5):
    min_value, min_index = torch.min(spectrum, dim=1)
    threshold_value = min_value / threshold_ratio
    print(min_value)
    print(threshold_value)

    left_index = torch.zeros_like(min_index)
    right_index = torch.zeros_like(min_index)

    for i in range(spectrum.shape[0]):
        # 寻找左侧索引
        for j in range(min_index[i], -1, -1):
            if spectrum[i, j] > threshold_value[i]:
                left_index[i] = j
                break

        # 寻找右侧索引
        for j in range(min_index[i], spectrum.shape[1]):
            if spectrum[i, j] > threshold_value[i]:
                right_index[i] = j
                break

    # 计算宽度
    print(right_index,left_index)
    width = right_index - left_index
    relative_width = width.float() / (spectrum.shape[1] - 1)

    return relative_width.unsqueeze(-1)


# 然后你可以使用这个函数来创建一个新的模型，它基于原始模型，但输出的是最低峰的波长
class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model

    def forward(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        original_output = self.model(x)
        #lowest-peak wavelength
        #new_output = find_lowest_peak(original_output)
        #lowest-peak width
        #new_output = find_lowest_peak_width(original_output)
        # lowest-peak value
        new_output = find_lowest_peak_value(original_output)
        #normalized_output = torch.sigmoid(new_output)  # 进行归一化
        return new_output.detach().numpy()


new_model = NewModel(model)
print(X[127])
print(X[1311])
print(new_model(X[127].unsqueeze(0)),new_model(X[1311].unsqueeze(0)))


# 现在你可以用SHAP来解释新模型的预测结果了
# Use 'model' instead of 'original_model'
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

background_np = background.numpy()
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()

explainer = shap.Explainer(new_model, X_train_np)  # Use 'background' instead of 'X_train'
#shap_values = explainer.shap_values(X_test_np)
shap_value = explainer(X_test_np)


feature_names = ["R", "n_host"]

#shap.summary_plot(shap_values, X_test_np, feature_names = feature_names)  # Fix the typo
plt.figure(figsize=(10,6))
shap.summary_plot(shap_value,X_test,feature_names = feature_names)
shap.summary_plot(shap_value,X_test,feature_names = feature_names, plot_type= 'bar')
plt.show()
