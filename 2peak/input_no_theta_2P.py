import numpy as np
import tandem_2p
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator
from data_loader_2P import load_data
from test_cvae_2p import spec
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# Plot style (match ./1peak/input_no_theta.py)
# -----------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

SHOW_PLOT = False
SAVE_PLOT = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FIG_DIR = os.path.join(BASE_DIR, 'figures')
FIG_DPI = 300
SAVE_PDF = True

PPT_BLUE = '#2E75B6'
PPT_GREEN = '#70AD47'
PPT_ORANGE = '#ED7D31'


def _autoscale_axes(ax, x, ys, *, xpad_frac=0.02, ypad_frac=0.06, force_01=False):
    """Auto-fit axes to data with small margins."""
    x = np.asarray(x, dtype=float)
    if x.size:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        xpad = (xmax - xmin) * xpad_frac if xmax > xmin else 1.0
        ax.set_xlim(xmin - xpad, xmax + xpad)

    y_min = np.inf
    y_max = -np.inf
    for y in ys:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            continue
        y_min = min(y_min, np.nanmin(y))
        y_max = max(y_max, np.nanmax(y))

    if np.isfinite(y_min) and np.isfinite(y_max):
        if force_01 and y_min >= -0.05 and y_max <= 1.05:
            ax.set_ylim(0.0, 1.0)
        else:
            ypad = (y_max - y_min) * ypad_frac if y_max > y_min else 0.1
            ax.set_ylim(y_min - ypad, y_max + ypad)


def setup_figure(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', labelsize=16, length=4, width=1, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, width=1, direction='in')
    ax.xaxis.set_major_locator(FixedLocator([400, 600, 800]))
    ax.xaxis.set_minor_locator(FixedLocator([500, 700]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_xlim(380, 800)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    return fig, ax


def save_fig(fig, basepath):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIG_DIR, basepath + '.png'), dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.10)
    if SAVE_PDF:
        fig.savefig(os.path.join(FIG_DIR, basepath + '.pdf'), bbox_inches='tight', pad_inches=0.10)


# mode = 'FNN'
# mode = 'tandem'
mode = input('Enter mode (FNN, tandem): ').strip()

def target_trans(wlmin=380, wlmax=800, num=1001, plot=False):

    wl = np.linspace(wlmin, wlmax, num) #生成等间距1*num个点
    #spe = np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)) # ture or false
    #spe = np.logical_not(np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)))
    # spe = np.logical_not(np.logical_and(wl >= 532, wl < 533))
    # spe = spe.astype(np.float) #转化为float类型,532
    spe = np.logical_not(np.logical_and(wl >= 520, wl < 535 ))
    spe = spe.astype(np.float32) #转化为float类型,532
    return torch.from_numpy(spe)

wlmin = 380
wlmax = 800
num = 1001

target = target_trans(wlmin,wlmax,num)
wl = torch.linspace(wlmin,wlmax,num)


def lorentzian_trans(centers,wlmin=380, wlmax=800, num=1001, gamma=15, plot=False):
    wl = np.linspace(wlmin, wlmax, num)
    spe = np.zeros_like(wl)

    for center in centers:
        spe += 1 / (1 + ((wl - center) / gamma) ** 2)
    spe = (spe - 1) * -1

    if plot:
        plt.plot(wl, spe)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Intensity')
        plt.show()

    return torch.from_numpy(spe).float()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_a_list, data_b_list = load_data(os.path.join(BASE_DIR, 'data', 'cstdata_10496', 'data'))
data_a_array = []
for data in data_a_list:
    # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
    values = [data['R'], data['n_host1'],data['n_host2']]
    data_a_array.append(values)
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
X = torch.tensor(data_a_array, dtype=torch.float32).to(device)
y = torch.tensor(reshaped_data_b_list, dtype=torch.float32).to(device)
# y_related = torch.load('./data/y_related.pt')
# y_mean = torch.mean(y_related, dim=1)
#print(y[0])

def lowest_wavelength(x):
    # 初始化 numpy 数组来存储结果
    lowest_peaks_per_group = np.empty((len(x), 2))


    # 对每组数据取负值以找到最低的
    y_neg = -x
    peaks_neg, _ = find_peaks(y_neg,prominence=0.01)


    # 检查峰值数量并存储结果
    if len(peaks_neg) == 1:
        # 如果只有一个峰值，复制它
        lowest_peaks_per_group = [peaks_neg[0], peaks_neg[0]]
    elif len(peaks_neg) > 1:
        # 如果有多个峰值，选择最低的两个
        lowest_peaks = np.argsort(x[peaks_neg])[:2]
        lowest_peaks = np.sort(lowest_peaks)
        lowest_peaks_per_group = peaks_neg[lowest_peaks]

    min_index = lowest_peaks_per_group
    wavelengths = np.array(data_b_list[0])[:, 0]
    min_index_wave = wavelengths[min_index]
    min_index = torch.from_numpy(min_index)
    min_index_wave = torch.from_numpy(min_index_wave)
    return min_index,min_index_wave


# Make predictions
n_input = 3  # 12
n_classes = 1001  # 3
fnn_size = [n_input, 64, 128, 256, 512, 1024,2048,1024, n_classes]
inn_size = [n_classes,1024,2048,1024,512,256,128,64, n_input]
# fnn_size = [n_input, 64,128, 256, 512, 1024,2048, n_classes]
# inn_size = [n_classes,2048,1024,512,256,128,64, n_input]
# Bounds come from the deployed checkpoints' sidecar, not a literal:
# a stale pair here silently mis-denormalises every structure the
# inverse network predicts (this file said 3.0, and Shap.py also
# said R=6.0, while the checkpoints were trained with 6.5/3.05).
import dual_peak_utils as _bounds
structure_min = torch.as_tensor(_bounds.STRUCTURE_MIN).to(device)
structure_max = torch.as_tensor(_bounds.STRUCTURE_MAX).to(device)
tandem_net = tandem_2p.tandem_network(INN_size=inn_size, FNN_size=fnn_size,training=False,structure_max=structure_max,structure_min=structure_min).to(device)


if mode == 'tandem':


    tandem_net.restore_INN(os.path.join(BASE_DIR, 'model', 'DNN_tandem_INN_label.ckpt'))
    tandem_net.restore_FNN(os.path.join(BASE_DIR, 'model', 'DNN_tandem_FNN_label.ckpt'))

    spe = lorentzian_trans(centers = [615])
    # for CVAE + tandem
    peak1 = float(input("Enter the wavelength of peak1: ").strip())
    peak2 = float(input("Enter the wavelength of peak2: ").strip())
    input_values = spec(peak1,peak2,spe)
    input_tensor = input_values.to(device)
    # # for only tandem
    # peak1 = float(input("Enter the wavelength of peak1: ").strip())
    # peak2 = float(input("Enter the wavelength of peak2: ").strip())
    # input_values = lorentzian_trans([peak1,peak2],plot=True)
    # input_values = input_values.unsqueeze(0)
    # input_tensor = input_values.to(device)
    ## for test
    # input_values = spe
    # input_values = y[493]
    # input_values = torch.tensor(input_values, dtype = torch.float32)
    # input_tensor = input_values.unsqueeze(0)
    loss = nn.MSELoss()


    pre_layer, pre_response = tandem_net.test(X,input_tensor,'tandem')
    #print(f'pre_layer{pre_layer},true:{X[400]}')
    #array1 = spe

    for i in range(len(pre_response)):
        # min_index = np.argmin(pre_response[i])
        #
        #
        # min_value = pre_response[i].flatten()[min_index]
        # wavelengths = np.array(data_b_list[0])[:, 0]
        #min_index_wave = wavelengths[min_index]
        loss_peak_lowest = 9999999
        loss_lowest = 999999
        min_index,min_index_wave = lowest_wavelength(pre_response[i])
        response_tensor = torch.tensor(pre_response[i], dtype=torch.float32).to(device)
        input_tensor[i] = torch.tensor(input_tensor[i],dtype=torch.float32).to(device)
        peak = np.array([peak1,peak2])
        peak = np.sort(peak)
        peak = torch.tensor(peak)
        loss_peak = tandem_net.loss_fn(min_index_wave,peak)
        loss = tandem_net.loss_fn(response_tensor,input_tensor[i])
        # plt.figure()
        # wl = np.array(data_b_list[0])[:, 0]
        # plt.plot(wl, pre_response[i], label='prediction, wavelength:{}'.format(min_index_wave))
        # # plt.plot(wl,y_mean[493],label='target')
        # plt.plot(wl, input_tensor[i].cpu().numpy(), label='target')
        # plt.xlabel('Wavelength')
        # plt.ylabel('Intensity')
        # plt.legend()
        # plt.show()
        if loss_peak < loss_peak_lowest:
            loss_peak_lowest = loss_peak
            loss_lowest = loss
            t = i
        elif loss_peak == loss_peak_lowest:
            if loss < loss_lowest:
                loss_peak_lowest = loss_peak
                loss_lowest = loss
                t = i
        #print("最小值:", min_value)
        #print("最小值的索引:", np.unravel_index(min_index, pre_response[i].shape))    #pre_layer_tensor = torch.from_numpy(pre_layer)
    min_index,min_index_wave = lowest_wavelength(pre_response[t])
    print(f'pre_layer{pre_layer[t]}')
    print("最小值索引：", min_index, "最小波长：", min_index_wave)
    fig, ax = setup_figure()
    wl = np.array(data_b_list[0])[:, 0]

    y_pred = pre_response[t]
    y_tgt = input_tensor[t].detach().cpu().numpy()

    ax.plot(wl, y_pred, color=PPT_BLUE, linewidth=2.2, solid_capstyle='round')
    ax.plot(
        wl,
        y_tgt,
        color=PPT_ORANGE,
        linewidth=2.0,
        linestyle=(0, (5, 3)),
        alpha=0.95,
        solid_capstyle='round',
        dash_capstyle='round',
    )
    _autoscale_axes(ax, wl, [y_pred, y_tgt], force_01=False)

    for j in range(len(min_index)):
        xj = wl[min_index[j]]
        yj = y_pred[min_index[j]]
        ax.scatter(
            xj,
            yj,
            color=PPT_BLUE,
            s=45,
            zorder=5,
            edgecolors='white',
            linewidths=1.2,
        )
        ax.annotate(
            f'{min_index_wave[j].item():.1f}',
            xy=(xj, yj),
            xytext=(-28, 12),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.2),
            textcoords='offset points',
            ha='right',
            va='bottom',
        )

    target_wavelengths = [peak1, peak2]
    print(f'target_wavelengths:{target_wavelengths}')
    min_index, min_index_wave = lowest_wavelength(y_tgt)
    for j in range(len(min_index)):
        xj = target_wavelengths[j]
        yj = y_tgt[min_index[j]]
        ax.scatter(
            xj,
            yj,
            color=PPT_ORANGE,
            s=45,
            zorder=5,
            edgecolors='white',
            linewidths=1.2,
        )
        ax.annotate(
            f'{xj:.1f}',
            xy=(xj, yj),
            xytext=(16, 12),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.2),
            textcoords='offset points',
            ha='left',
            va='bottom',
        )

    ax.set_xlabel('Wavelength(nm)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Transmittance', fontsize=20, fontweight='bold')

    if SAVE_PLOT:
        save_fig(fig, 'tandem_pred_2peak')
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)

    print(f'loss:{loss_lowest},loss_peak:{loss_peak_lowest}')

else:
    #Make predictions
    #checkpoint_path = "D:/Photosynthesis design with DL/model/DNN_tandem_FNN_no_theta.ckpt"

    # 使用torch.load()函数加载checkpoint文件
    #tandem_net.fnn.load_state_dict(torch.load(checkpoint_path))
    #checkpoint = torch.load(checkpoint_path)

    # 查看checkpoint中的内容
    #print(checkpoint.keys())
    tandem_net.restore_FNN(os.path.join(BASE_DIR, 'model', 'DNN_tandem_FNN_label.ckpt'))
    #
    # response = tandem_net.test(X, y_mean, "FNN")
    # response_tensor = torch.tensor(response, dtype=torch.float32)
    # err_rms = tandem_net.loss_fn(response_tensor,y_mean)
    # print(f'Test set Mse:{err_rms}')

    # input_values = [
    #     {'R': 4, 'n_host': 1.6},
    #     {'R': 5, 'n_host': 1.8},
    #     {'R': 6, 'n_host': 2.0}
    # ]

    input_values = [
        #{'R':2.7982302, 'n_host':2.3380327},
        #{'R': 4.8887362, 'n_host': 2.3616776}
        {'R': 2.559, 'n_host1': 2.899, 'n_host2':1.7253}
        #{'R': 3.798861, 'n_host': 2.3244426}

    ]

    input_array = []
    for data in input_values:
        values = [data['R'], data['n_host1'],data['n_host2']]
        input_array.append(values)

    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)


    with torch.no_grad():
        predictions = tandem_net.test(input_tensor,1,'FNN')

    wavelengths = np.array(data_b_list[0])[:, 0]  # Wavelengths from the first data_b_list item

    # Plotting

    spec1 = np.loadtxt(
        os.path.join(ROOT_DIR, 'Filter_CST_DataGen_Share', 'data', '430_620.b'),
        delimiter='\t',
    )
    # spec2 = np.loadtxt('./data/2D_CST/00002.b',
    #                   delimiter='\t')
    # spec3 = np.loadtxt('./data/2D_CST/00003.b',
    #                   delimiter='\t')
    wavelength1 = spec1[:, 0]
    intensity1 = spec1[:, 1]
    # wavelength2 = spec2[:, 0]
    # intensity2 = spec2[:, 1]
    # wavelength3 = spec3[:, 0]
    # intensity3 = spec3[:, 1]
    # intensity = (intensity1 + intensity2 + intensity3)/3
    min_index = np.argmin(intensity1)
    print(min_index)
    min_value = intensity1.flatten()[min_index]
    min_index_wave = wavelength1[min_index]

    mse = nn.MSELoss()
    for i, data in enumerate(input_values):
        print(f"Input: R={data['R']} n_host1={data['n_host1']} n_host2={data['n_host2']}")
        print("Prediction:", predictions[i])
        fig, ax = setup_figure()
        min_index_p = np.argmin(predictions[i])
        print(min_index_p)
        min_index_wave_p = wavelengths[min_index_p]

        ax.plot(
            wavelengths,
            predictions[i][:len(wavelengths)],
            color=PPT_BLUE,
            linewidth=2.2,
            solid_capstyle='round',
        )
        ax.plot(
            wavelength1,
            intensity1,
            color=PPT_ORANGE,
            linewidth=2.0,
            linestyle=(0, (5, 3)),
            alpha=0.95,
            solid_capstyle='round',
            dash_capstyle='round',
        )
        _autoscale_axes(ax, wavelengths, [predictions[i][:len(wavelengths)], intensity1], force_01=False)

        ax.set_xlabel('Wavelength(nm)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Transmittance', fontsize=20, fontweight='bold')

        if SAVE_PLOT:
            save_fig(fig, f'FNN_pred_2peak_{i}')
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)

        print(mse(torch.tensor(intensity1),torch.tensor(predictions[i])))


    # # 绘制光谱
    # plt.figure(figsize=(10, 6))
    # plt.plot(wavelength, intensity, label='Intensity vs. Wavelength')
    # plt.xlabel('Wavelength')
    # plt.ylabel('Intensity')
    # plt.title('Spectrum')
    # plt.legend()
    # plt.show()

