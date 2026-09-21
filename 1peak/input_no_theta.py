from ast import mod
import argparse
import numpy as np
import tandem
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from matplotlib.ticker import FixedLocator
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
from data_loader import load_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the direct Tandem inverse design for one wavelength.'
    )
    parser.add_argument(
        '--target-wavelength',
        type=float,
        default=410.0,
        help='Requested resonance wavelength in nm.',
    )
    return parser.parse_args()


ARGS = parse_args()

# 图像显示与保存：True 弹窗显示，False 不弹窗
SHOW_PLOT = False
SAVE_PLOT = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_WAVELENGTH = float(ARGS.target_wavelength)
RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(BASE_DIR, f'{TARGET_WAVELENGTH:g}', 'tandem')
FIG_DIR = OUTPUT_DIR
DATA_DIR = OUTPUT_DIR
FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.7)  # 约 89 mm，适合期刊单栏
SAVE_PDF = False  # 子图阶段只保存 PNG，整合多面板图时再统一导出

# PPT 常用色（主题色/标准色）
PPT_BLUE = '#2E75B6'
PPT_GREEN = '#70AD47'
PPT_ORANGE = '#ED7D31'

def _autoscale_axes(ax, x, ys, *, xpad_frac=0.02, ypad_frac=0.06, force_01=True):
    """Auto-fit axes to data with small margins.

    When force_01=True and the data roughly lies in [0, 1], lock to [0, 1]
    for a clean, comparable transmittance scale. When False, fully auto-scale.
    """
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


# 期刊单栏图：小字号、细线条、完整边框，缩放后仍保持清晰
def setup_figure(figsize=SINGLE_COLUMN_SIZE):
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(
        axis='both',
        which='major',
        length=3,
        width=0.7,
        direction='in',
        top=True,
        right=True,
        pad=2,
    )
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_xlim(380, 800)
    ax.set_ylim(-0.025, 1.025)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return fig, ax

def save_fig(fig, basepath):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(os.path.join(FIG_DIR, basepath + '.png'), dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.04)
    if SAVE_PDF:
        fig.savefig(os.path.join(FIG_DIR, basepath + '.pdf'), bbox_inches='tight', pad_inches=0.04)


def save_tandem_results(
    index,
    wavelengths,
    target_response,
    predicted_response,
    predicted_structure,
    target_wavelength,
    predicted_min_index,
    mse,
):
    """Save plotting data and scalar prediction results for later analysis."""
    os.makedirs(DATA_DIR, exist_ok=True)

    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    target_response = np.asarray(target_response, dtype=float).reshape(-1)
    predicted_response = np.asarray(predicted_response, dtype=float).reshape(-1)
    predicted_structure = np.asarray(predicted_structure, dtype=float).reshape(-1)

    data_path = os.path.join(DATA_DIR, f'tandem_pred_{index}_data.txt')
    plotting_data = np.column_stack(
        (wavelengths, target_response, predicted_response)
    )
    np.savetxt(
        data_path,
        plotting_data,
        delimiter='\t',
        fmt='%.10e',
        header='wavelength_nm\ttarget_transmittance\tpredicted_transmittance',
        comments='',
    )

    predicted_min_wavelength = wavelengths[predicted_min_index]
    predicted_min_value = predicted_response[predicted_min_index]
    result_path = os.path.join(DATA_DIR, f'tandem_pred_{index}_results.txt')
    with open(result_path, 'w', encoding='utf-8') as result_file:
        result_file.write(f'run_tag\t{RUN_TAG}\n')
        result_file.write(f'target_wavelength_nm\t{target_wavelength:.10f}\n')
        if predicted_structure.size > 0:
            result_file.write(f'predicted_R\t{predicted_structure[0]:.10f}\n')
        if predicted_structure.size > 1:
            result_file.write(f'predicted_n_host\t{predicted_structure[1]:.10f}\n')
        for parameter_index, value in enumerate(predicted_structure[2:], start=2):
            result_file.write(
                f'predicted_parameter_{parameter_index}\t{value:.10f}\n'
            )
        result_file.write(f'predicted_min_index\t{predicted_min_index}\n')
        result_file.write(
            f'predicted_min_wavelength_nm\t{predicted_min_wavelength:.10f}\n'
        )
        result_file.write(
            f'predicted_min_transmittance\t{predicted_min_value:.10e}\n'
        )
        result_file.write(f'wavelength_error_nm\t{predicted_min_wavelength - target_wavelength:.10f}\n')
        result_file.write(f'mse\t{mse:.10e}\n')
        result_file.write(f'rmse\t{np.sqrt(mse):.10e}\n')

    structure_path = os.path.join(
        DATA_DIR,
        f'tandem_pred_{index}_structure.txt',
    )
    np.savetxt(
        structure_path,
        predicted_structure[:2].reshape(1, -1),
        delimiter='\t',
        fmt='%.10f',
        header='R\tn_host',
        comments='',
    )

    return data_path, result_path, structure_path


# mode = 'FNN'
mode = 'tandem'

def target_trans(wlmin=380, wlmax=800, num=1001, plot=False):

    wl = np.linspace(wlmin, wlmax, num) #生成等间距1*num个点
    #spe = np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)) # ture or false
    #spe = np.logical_not(np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700)))
    # spe = np.logical_not(np.logical_and(wl >= 532, wl < 533))
    # spe = spe.astype(np.float) #转化为float类型,532
    spe = np.logical_not(np.logical_and(wl >= 520, wl < 535 ))
    spe = spe.astype(float) #转化为float类型,532
    return torch.from_numpy(spe)

wlmin = 380
wlmax = 800
num = 1001

target = target_trans(wlmin,wlmax,num)
wl = torch.linspace(wlmin,wlmax,num)


def lorentzian_trans(
    center,
    wlmin=380,
    wlmax=800,
    num=1001,
    gamma=15,
    plot=False,
    wavelengths=None,
):
    if wavelengths is None:
        wl = np.linspace(wlmin, wlmax, num)
    else:
        wl = np.asarray(wavelengths, dtype=float)
    #center = 532
    #spe = gamma / ((wl - center) ** 2 + gamma ** 2)  # Lorentzian function
    spe = 1 / (1 + ((wl - center) / gamma)**2)
    spe = (spe-1) * -1

    if plot:
        fig, ax = setup_figure()
        ax.plot(wl, spe, color=PPT_BLUE, linewidth=1.5)
        ax.set_xlim(380, 800)
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transmittance')
        if SAVE_PLOT:
            save_fig(fig, 'lorentzian_spec')
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)

    return torch.from_numpy(spe).float()


data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')
data_a_array = []
for data in data_a_list:
    # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
    values = [data['R'], data['n_host']]
    data_a_array.append(values)
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
X = torch.tensor(data_a_array, dtype=torch.float32)
y = torch.from_numpy(np.asarray(reshaped_data_b_list, dtype=np.float32))
y_related = torch.load(
    os.path.join(BASE_DIR, 'data', 'y_related.pt'),
    map_location='cpu',
)
y_mean = torch.mean(y_related, dim=1)
# print(y[0])


# Make predictions
n_input = 2  # 12
n_classes = 1001  # 3
fnn_size = [n_input, 64, 128, 256, 512, 1024,2048,1024, n_classes]
inn_size = [n_classes,1024,2048,1024,512,256,128,64, n_input]
structure_min = torch.tensor([2.5,1.5])
structure_max = torch.tensor([6.5,3.05])
tandem_net = tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size,training=False,structure_max=structure_max,structure_min=structure_min).to(device)


if mode == 'tandem':
    tandem_net.restore_INN(
        os.path.join(BASE_DIR, 'model', 'DNN_tandem_INN_label.ckpt')
    )
    tandem_net.restore_FNN(
        os.path.join(BASE_DIR, 'model', 'DNN_tandem_FNN_label.ckpt')
    )
    target_wavelength = TARGET_WAVELENGTH

    model_wavelengths = np.array(data_b_list[0])[:, 0]
    spe = lorentzian_trans(
        center=target_wavelength,
        wavelengths=model_wavelengths,
    )
    # for CVAE + tandem
    # from test_cvae import spec
    # input_values = spec(target_wavelength,spe)
    # input_tensor = input_values.to(device)

    # for only tandem
    input_values = spe
    #input_values = y_mean[100]
    input_values = input_values.float()
    input_tensor = input_values.unsqueeze(0)

    pre_layer, pre_response = tandem_net.test(X.to(device),input_tensor.to(device),'tandem')
    #print(f'pre_layer{pre_layer},true:{X[400]}')
    print(f'pre_layer{pre_layer}')
    array = spe

    for i in range(len(pre_response)):
        predicted_response = np.asarray(pre_response[i]).reshape(-1)
        predicted_min_index = int(np.argmin(predicted_response))


        predicted_min_value = predicted_response[predicted_min_index]
        if not data_b_list:
            print("data_b_list is empty")
        else:
            print("data_b_list is not empty")

        wavelengths = np.array(data_b_list[0])[:, 0]
        predicted_min_wavelength = wavelengths[predicted_min_index]
        print("最小值索引：", predicted_min_index, "最小波长：", predicted_min_wavelength)
        print("最小值:", predicted_min_value)
        #print("最小值的索引:", np.unravel_index(min_index, pre_response[i].shape))    #pre_layer_tensor = torch.from_numpy(pre_layer)


        wl = np.array(data_b_list[0])[:, 0]
        target_response = input_tensor[i].detach().cpu().numpy()
        response_tensor = torch.as_tensor(
            predicted_response, dtype=torch.float32, device=device
        )
        mse = tandem_net.loss_fn(
            response_tensor, input_tensor[i].to(device)
        ).detach().cpu().item()

        fig, ax = setup_figure()
        ax.plot(
            wl,
            target_response,
            color=PPT_ORANGE,
            linewidth=1.35,
            linestyle=(0, (4, 2.4)),
            solid_capstyle='round',
            dash_capstyle='round',
            zorder=2,
        )
        ax.plot(
            wl,
            predicted_response,
            color=PPT_BLUE,
            linewidth=1.5,
            solid_capstyle='round',
            zorder=3,
        )

        min_index = np.argmin(np.abs(wl - target_wavelength))
        min_value = input_tensor[i].detach().cpu().numpy()[min_index]
        ax.scatter(
            wl[min_index],
            min_value,
            marker='D',
            facecolors='white',
            edgecolors=PPT_ORANGE,
            s=16,
            zorder=5,
            linewidths=0.8,
        )
        ax.scatter(
            wl[predicted_min_index],
            predicted_response[predicted_min_index],
            color=PPT_BLUE,
            s=15,
            zorder=6,
            edgecolors='white',
            linewidths=0.5,
        )
        ax.annotate(
            f'{target_wavelength:.1f}',
            xy=(wl[min_index], min_value),
            xytext=(20, 27),
            textcoords='offset points',
            fontsize=6.7,
            color='#333333',
            ha='left',
            va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                color='#555555',
                linewidth=0.55,
                shrinkA=2,
                shrinkB=2,
                mutation_scale=6,
            ),
            annotation_clip=True,
        )
        ax.annotate(
            f'{predicted_min_wavelength:.1f}',
            xy=(
                wl[predicted_min_index],
                predicted_response[predicted_min_index],
            ),
            xytext=(45, 9),
            textcoords='offset points',
            fontsize=6.7,
            color='#333333',
            ha='left',
            va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                color='#555555',
                linewidth=0.55,
                shrinkA=2,
                shrinkB=2,
                mutation_scale=6,
            ),
            annotation_clip=True,
        )

        ax.set_xlim(380, 800)
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transmittance')
        if SAVE_PLOT:
            save_fig(fig, f'tandem_pred_{i}')
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)

        data_path, result_path, structure_path = save_tandem_results(
            index=i,
            wavelengths=wl,
            target_response=target_response,
            predicted_response=predicted_response,
            predicted_structure=pre_layer[i],
            target_wavelength=target_wavelength,
            predicted_min_index=predicted_min_index,
            mse=mse,
        )
        print(f'loss:{mse}')
        print(f'绘图数据已保存：{data_path}')
        print(f'结果摘要已保存：{result_path}')
        print(f'结构参数已保存：{structure_path}')
        print(f'图片已保存：{FIG_DIR}')

else:
    #Make predictions
    #checkpoint_path = "D:/Photosynthesis design with DL/model/DNN_tandem_FNN_no_theta.ckpt"

    # 使用torch.load()函数加载checkpoint文件
    #tandem_net.fnn.load_state_dict(torch.load(checkpoint_path))
    #checkpoint = torch.load(checkpoint_path)

    # 查看checkpoint中的内容
    #print(checkpoint.keys())
    tandem_net.restore_FNN('./model/DNN_tandem_FNN_label.ckpt')
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
        {'R': 4.387, 'n_host':  2.825}
        #{'R': 3.798861, 'n_host': 2.3244426}

    ]

    input_array = []
    for data in input_values:
        values = [data['R'], data['n_host']]
        input_array.append(values)

    input_tensor = torch.tensor(input_array, dtype=torch.float32)


    with torch.no_grad():
        predictions = tandem_net.test(input_tensor.to(device),1,'FNN')

    wavelengths = np.array(data_b_list[0])[:, 0]  # Wavelengths from the first data_b_list item

    # Plotting
    spec1 = np.loadtxt('../Filter_CST_DataGen_Share/data/00001.b',
                      delimiter='\t')  # 假设数据列之间是由空格分隔的，如果是其他分隔符，如逗号，请更改 delimiter 参数为 ','
    spec2 = np.loadtxt('../Filter_CST_DataGen_Share/data/00002.b',
                      delimiter='\t')
    spec3 = np.loadtxt('../Filter_CST_DataGen_Share/data/00003.b',
                      delimiter='\t')
    wavelength1 = spec1[:, 0]
    intensity1 = spec1[:, 1]
    wavelength2 = spec2[:, 0]
    intensity2 = spec2[:, 1]
    wavelength3 = spec3[:, 0]
    intensity3 = spec3[:, 1]
    intensity = (intensity1 + intensity2 + intensity3)/3
    # intensity = intensity1
    min_index = np.argmin(intensity)
    print(min_index)
    min_value = intensity.flatten()[min_index]
    min_index_wave = wavelength1[min_index]

    for i, data in enumerate(input_values):
        print(f"Input: R={data['R']} n_host={data['n_host']}")
        print("Prediction:", predictions[i])
        min_index_p = np.argmin(predictions[i])
        print(min_index_p)
        min_index_wave_p = wavelengths[min_index_p]

        fig, ax = setup_figure()
        ax.plot(
            wavelengths,
            predictions[i][:len(wavelengths)],
            color=PPT_BLUE,
            linewidth=1.5,
            solid_capstyle='round',
        )
        ax.plot(
            wavelength1,
            intensity,
            color=PPT_ORANGE,
            linewidth=1.35,
            linestyle=(0, (4, 2.4)),
            solid_capstyle='round',
            dash_capstyle='round',
        )
        ax.set_xlim(380, 800)
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transmittance')
        if SAVE_PLOT:
            save_fig(fig, f'FNN_pred_{i}')
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)
        # print()
    
        # plt.figure(figsize=(8, 6))    
        # plt.plot(wavelength1, intensity1,label='0 degree'.format(min_index_wave),linewidth=2.5)
        # plt.plot(wavelength1, intensity2,label='30 degree'.format(min_index_wave),linewidth=2.5)
        # plt.plot(wavelength1, intensity3,label='60 degree'.format(min_index_wave),linewidth=2.5)
        # plt.xlabel('Wavelength (nm)',fontsize=15, fontweight='bold')
        # plt.ylabel('Transmittance',fontsize=15, fontweight='bold')
        # plt.xticks(fontsize=15, fontweight='bold')                 # x刻度
        # plt.yticks(fontsize=15, fontweight='bold')                 # y刻度
        # ax = plt.gca()
        # for spine in ax.spines.values():
        #     spine.set_color('black')
        #     spine.set_linewidth(1.5)

        # ax.tick_params(axis='both', colors='black', width=2, length=6)
        # plt.legend(prop={'weight': 'bold', 'size': 12})
        # plt.show()
        # print()


    # # 绘制光谱
    # plt.figure(figsize=(10, 6))
    # plt.plot(wavelength, intensity, label='Intensity vs. Wavelength')
    # plt.xlabel('Wavelength')
    # plt.ylabel('Intensity')
    # plt.title('Spectrum')
    # plt.legend()
    # plt.show()
