import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap

from data_loader_2P import load_data
import tandem_2p

# ===================== 全局设置 =====================
# 光谱采样信息（根据你数据来，如果不同请改这里）
LAMBDA_MIN = 380.0
LAMBDA_MAX = 800.0
N_POINTS   = 1001

# 选择要解释的物理量：
#   'lambda1' : 较短波长那个谷的位置 (nm)
#   'lambda2' : 较长波长那个谷的位置 (nm)
#   'val1'    : 较短波长那个谷的透射值
#   'val2'    : 较长波长那个谷的透射值
#   'val_mean': 两个谷透射值的平均
MODE = "val2"

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ===================== 1. 读取 2-peak 数据 =====================
data_a_list, data_b_list = load_data('./2peak/data/cstdata_10496/data')

data_a_array = []
for data in data_a_list:
    # 三个结构参数：R, n_host1, n_host2
    values = [data['R'], data['n_host1'], data['n_host2']]
    data_a_array.append(values)

# 光谱：每个 data_b 是 [lambda, T]，这里取 T
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]

X = torch.tensor(data_a_array, dtype=torch.float32)               # [N, 3]
y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)       # [N, 1001]

num = len(y)
idx = np.arange(num)
np.random.shuffle(idx)

# train / test 划分
r = 0.8
train_idx = idx[:int(num * r)]
test_idx  = idx[int(num * r):]

X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

# ===================== 2. 加载 tandem FNN 模型 =====================
n_input   = 3      # R, n_host1, n_host2
n_classes = N_POINTS

fnn_size = [n_input, 64, 128, 256, 512, 1024, 2048, 1024, n_classes]
inn_size = [n_classes, 1024, 2048, 1024, 512, 256, 128, 64, n_input]

# Bounds come from the deployed checkpoints' sidecar, not a literal:
# a stale pair here silently mis-denormalises every structure the
# inverse network predicts (this file said 3.0, and Shap.py also
# said R=6.0, while the checkpoints were trained with 6.5/3.05).
import dual_peak_utils as _bounds
structure_min = torch.as_tensor(_bounds.STRUCTURE_MIN)
structure_max = torch.as_tensor(_bounds.STRUCTURE_MAX)

tandem_net = tandem_2p.tandem_network(
    INN_size=inn_size,
    FNN_size=fnn_size,
    training=False,
    structure_max=structure_max,
    structure_min=structure_min
)

# 把 checkpoint 加载到对应 device
state_dict = torch.load('./2peak/model/DNN_tandem_FNN_label.ckpt',
                        map_location=device)
tandem_net.fnn.load_state_dict(state_dict)
tandem_net = tandem_net.to(device)
model = tandem_net.fnn                  # 前向网络：输入结构 → 输出光谱
print("Model device:", next(model.parameters()).device)

# ===================== 3. 从光谱中找两个最低谷 =====================
def find_two_lowest_peaks(spectrum: torch.Tensor):
    """
    spectrum: [batch, N_POINTS]
    返回:
      lambda_two: [batch, 2]   两个谷的波长 (nm)，按波长从小到大排序
      value_two:  [batch, 2]   对应的透射值，跟 lambda_two 对应
    """
    # 找两个最小值（largest=False）
    vals_raw, idx_raw = torch.topk(spectrum, k=2, dim=1, largest=False)  # [batch, 2]

    # 按 index（相当于波长）排序：小 index → 短波长
    idx_sorted, order = torch.sort(idx_raw, dim=1)        # [batch, 2]
    vals_sorted = torch.gather(vals_raw, 1, order)        # [batch, 2]

    # index → 波长 (nm)
    lambda_two = LAMBDA_MIN + idx_sorted.float() / (N_POINTS - 1) * (LAMBDA_MAX - LAMBDA_MIN)

    return lambda_two, vals_sorted


# ===================== 4. NewModel：输出你想解释的那个标量 =====================
class NewModel(nn.Module):
    def __init__(self, model, mode="val_mean"):
        """
        mode 选项：
          'lambda1' : 较短波长那个谷的位置 (nm)
          'lambda2' : 较长波长那个谷的位置 (nm)
          'val1'    : 较短波长那个谷的透射值
          'val2'    : 较长波长那个谷的透射值
          'val_mean': 两个谷透射值的平均
        """
        super(NewModel, self).__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        self.model.eval()

        # shap 传进来的是 numpy，这里统一成 float32 tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()

        # 把输入搬到和 model 一样的 device
        model_device = next(self.model.parameters()).device
        x = x.to(model_device)

        with torch.no_grad():
            spectrum = self.model(x)                 # [batch, N_POINTS]
            lambda_two, val_two = find_two_lowest_peaks(spectrum)  # [batch, 2], [batch, 2]

            if self.mode == "lambda1":
                out = lambda_two[:, 0:1]            # [batch, 1]
            elif self.mode == "lambda2":
                out = lambda_two[:, 1:2]
            elif self.mode == "val1":
                out = val_two[:, 0:1]
            elif self.mode == "val2":
                out = val_two[:, 1:2]
            elif self.mode == "val_mean":
                out = val_two.mean(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # 返回 numpy，先搬回 CPU
        return out.detach().cpu().numpy()


new_model = NewModel(model, mode=MODE)   # 不要 .to(device)，内部已经处理好

print("Example inputs:")
print("X[0] =", X[0])
print("X[1] =", X[1])
print("NewModel outputs:", new_model(X[0].unsqueeze(0)),
      new_model(X[1].unsqueeze(0)))

# ===================== 5. 用 SHAP 解释 new_model =====================
# 背景样本：从训练集里抽 100 个（numpy 格式留给 SHAP）
background    = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
background_np = background.numpy()
X_test_np     = X_test.numpy()

explainer  = shap.Explainer(new_model, background_np)
shap_value = explainer(X_test_np)   # shap.Explanation

feature_names = ["R", "n_host1", "n_host2"]

# shap.Explanation:
#   shap_value.values: [N_sample, N_feature]
#   shap_value.data  : [N_sample, N_feature]
shap_vals = shap_value.values
feat_vals = shap_value.data

print("shap_vals shape:", shap_vals.shape)
print("feat_vals shape:", feat_vals.shape)

# ===================== 6. 数值打印：SHAP≈0 时的特征值 =====================
# 如果 MODE 是 'val*'，输出通常在 [0,1]，eps 可以设小一点；
# 如果是 'lambda*'，输出是 nm，eps 可以设成 1~5 看情况。
if MODE.startswith("val"):
    eps = 0.01
else:
    eps = 1.0

for j, name in enumerate(feature_names):
    sv = shap_vals[:, j]
    fv = feat_vals[:, j]

    mask = np.abs(sv) < eps
    if mask.sum() > 0:
        approx_val = fv[mask].mean()
        std_val    = fv[mask].std()
        print(f"{name}: SHAP≈0 (|shap|<{eps}) 时的大致特征值 ≈ {approx_val:.4f} ± {std_val:.4f}")
    else:
        print(f"{name}: 没有 |shap|<{eps}) 的点，可以把 eps 设大一点再试。")

# ===================== 7. 画 feature value vs SHAP 散点图 =====================
for j, name in enumerate(feature_names):
    sv = shap_vals[:, j]
    fv = feat_vals[:, j]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(fv, sv, c=fv, cmap='coolwarm', s=20)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)  # SHAP = 0 水平线

    # 标注 “SHAP≈0 时的大致特征值”
    mask = np.abs(sv) < eps
    if mask.sum() > 0:
        approx_val = fv[mask].mean()
        plt.axvline(approx_val, color='gray', linestyle=':', linewidth=1.5)
        plt.text(approx_val, 0, f"{approx_val:.3f}",
                 rotation=90, va='bottom', ha='right')

    plt.xlabel(f"{name} (feature value)", fontsize=20)
    ylabel_text = "SHAP value (impact on Output)"

    plt.ylabel(ylabel_text, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    cbar = plt.colorbar(sc)
    cbar.set_label(f"{name} value", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

# ===================== 8. SHAP summary plot（全局重要性） =====================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_value, X_test_np, feature_names=feature_names)
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_value, X_test_np, feature_names=feature_names, plot_type='bar')
plt.show()
