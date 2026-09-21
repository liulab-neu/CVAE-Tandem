import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from torch.optim.lr_scheduler import StepLR
from data_loader_2P import load_data
import utils1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # gamma and beta are learnable affine transform parameters
        self.gamma = nn.Parameter(torch.Tensor(1, num_features).uniform_())
        self.beta = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x):
        mean = x.mean([0, 2], keepdim=True)
        var = x.var([0, 2], keepdim=True)

        # Compute the normalized input (zero mean and unit variance)
        x_normalized = (x - mean) / (torch.sqrt(var + self.eps))

        # Scale and shift
        out = self.gamma * x_normalized + self.beta

        return out




def _wavelength_axis():
    """The shared 1001-point wavelength axis, from whichever layout is present.

    Tries the packed bundles first, then a per-sample .npz, then the archived
    text export, so the module keeps working as datasets are repacked or
    retired.
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if name.endswith('.npz'):
            with np.load(os.path.join(base, name)) as bundle:
                if 'wavelength_nm' in bundle.files:
                    return bundle['wavelength_nm'].astype(float)
    for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        directory = os.path.join(base, entry, 'data')
        if not os.path.isdir(directory):
            continue
        first_npz = os.path.join(directory, '00001.npz')
        if os.path.isfile(first_npz):
            with np.load(first_npz) as sample:
                return sample['wavelength_nm'].astype(float)
        first_text = os.path.join(directory, '00001.b')
        if os.path.isfile(first_text):
            return np.loadtxt(first_text, usecols=0)
    raise FileNotFoundError(f'No wavelength-axis source under {base}')



def build_dense(sizes, norm='bn', groups=32):
    """Dense stack with a selectable normalisation layer.

    ``bn`` is the original BatchNorm1d. ``gn`` uses GroupNorm, which keeps no
    running statistics and so behaves identically in train and eval mode --
    relevant here because the training loop only ever populates the BatchNorm
    statistics during the first epoch and then runs the remaining epochs in
    eval mode, freezing statistics taken from an untrained network.
    """
    layers = []
    for index in range(len(sizes) - 2):
        width = sizes[index + 1]
        layers.append(nn.Linear(sizes[index], width))
        if norm == 'gn':
            layers.append(nn.GroupNorm(min(groups, width), width))
        elif norm == 'bn':
            layers.append(nn.BatchNorm1d(width))
        elif norm != 'none':
            raise ValueError(f'Unknown norm: {norm}')
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class tandem_network(nn.Module):
    def __init__(self, INN_size, FNN_size, starter_learning_rate=0.001,
                 decay_step=1000 * 100, decay_rate=0.5, norm='bn',
                 structure_min = torch.tensor([2.5,1.5,1.5]),
                 # The n_host1 entry used to read 1.5, i.e. equal to its minimum,
                 # which makes the normalisation divide by zero if a caller ever
                 # relies on the default.
                 structure_max = torch.tensor([6.5, 3.05, 3.05]),
                 training=True):
        super(tandem_network, self).__init__()
        assert INN_size[0] == FNN_size[-1]
        assert INN_size[-1] == FNN_size[0]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.global_step = 0
        self.lr = starter_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.inn_size = INN_size
        self.fnn_size = FNN_size
        self.training = training
        self.structure_min = structure_min.to(self.device)
        self.structure_max = structure_max.to(self.device)

        # Only the wavelength axis is needed, and it is identical for every
        # sample, so read one spectrum rather than the whole dataset -- this
        # used to call load_data() on 10k .a/.b pairs on every construction.
        self.wl = _wavelength_axis()

        self.norm = norm
        self.build_net()
        self.optimizer_fnn = optim.Adam(self.fnn.parameters(), lr=self.lr)
        self.scheduler_fnn = StepLR(self.optimizer_fnn, step_size=decay_step, gamma=decay_rate)
        # self.optimizer_inn = optim.Adam([{'params':self.awl.parameters()},{'params':self.inn.parameters(), 'lr':self.lr}])
        self.optimizer_inn = optim.Adam([{'params': self.inn.parameters(), 'lr': self.lr}])
        self.scheduler_inn = StepLR(self.optimizer_inn, step_size=decay_step, gamma=decay_rate)
        # ``self.scheduler`` used to be assigned twice, leaving the forward
        # network without any decay.  Alias kept for external callers.
        self.scheduler = self.scheduler_inn
        self.loss_fn = nn.MSELoss()
        # Both cases weight the spectral term by 300.  The dip position is
        # normalised to [0, 1] and differentiable here, so the two physics
        # terms start from 1 and the ablation grid justifies where they land.
        self.w_spectral = 300.0
        self.w_peak_wavelength = 1.0
        self.w_peak_intensity = 1.0




    def loss2(self,dataset1,dataset2):
        #tensor_seq = self.data(dataset)
        #tensor_seq = utils1.process_data(dataset)
        merged_tensor = torch.cat((dataset1, dataset2), dim=1)
        mean_mse = torch.mean(torch.mean(merged_tensor, dim=1)**2)
        var_mse = torch.mean(torch.var(merged_tensor,dim=1)**2)
        # print(mse)
        return mean_mse,var_mse

    # def lowest_wavelength(self,x):
    #     min_index = torch.argmin(x, axis=1)
    #
    #     min_index = min_index.reshape(-1, 1)
    #     min_index_wave = (min_index / (1001 - 1))
    #     #print(min_index_wave)
    #     return min_index,min_index_wave

    def lowest_wavelength(self, x):
        """Normalised positions of both dips, differentiable with respect to ``x``.

        A single ``argmin`` would only locate the deeper of the two notches, so
        the peak-wavelength term would constrain one resonance and let the
        other drift; ``find_peaks`` keeps both.  Its indices carry no gradient,
        so each position is refined by interpolating the parabola through the
        three samples around the dip, which makes it a smooth function of the
        spectrum values.  The pair is sorted by wavelength so that prediction
        and target are always compared resonance-to-resonance rather than in
        order of depth.
        """
        n = x.shape[1]
        detached = x.detach().cpu().numpy()
        located = np.empty((len(x), 2), dtype=np.int64)
        for i, row in enumerate(detached):
            peaks, _ = find_peaks(-row, prominence=0.01)
            if len(peaks) == 0:
                fallback = int(np.argmin(row))
                located[i] = [fallback, fallback]
            elif len(peaks) == 1:
                located[i] = [peaks[0], peaks[0]]
            else:
                deepest = peaks[np.argsort(row[peaks])[:2]]
                located[i] = np.sort(deepest)
        min_index = torch.from_numpy(located).to(x.device)
        return min_index, self.interpolate_positions(x, min_index)

    def interpolate_positions(self, x, min_index):
        """Sub-grid dip positions, differentiable with respect to ``x``.

        Split out so a caller that already knows where the dips are -- the
        training loop caches them for the fixed target spectra -- can skip the
        peak search, which is the dominant cost of the loss.
        """
        n = x.shape[1]
        clamped = min_index.clamp(1, n - 2)
        rows = torch.arange(x.shape[0], device=x.device).reshape(-1, 1)
        left = x[rows, clamped - 1]
        centre = x[rows, clamped]
        right = x[rows, clamped + 1]
        denominator = left - 2.0 * centre + right
        flat = denominator.abs() < 1e-8
        safe = torch.where(flat, torch.ones_like(denominator), denominator)
        delta = torch.where(flat, torch.zeros_like(denominator),
                            0.5 * (left - right) / safe).clamp(-0.5, 0.5)
        return (clamped.to(x.dtype) + delta) / (n - 1)



    def build_net(self):
        self.response = nn.Linear(self.inn_size[0], self.inn_size[0]).to(self.device)  # Fixed: Use nn.Linear for response
        self.design = nn.Linear(self.inn_size[-1], self.inn_size[-1]).to(self.device)  # Fixed: Use nn.Linear for design
        self.fnn = self.build_dense(self.fnn_size).to(self.device)
        self.inn = self.build_dense(self.inn_size).to(self.device)
        #self.td = self.build_dense(self.fnn_size)

    def build_dense(self, Size):
        return build_dense(Size, getattr(self, 'norm', 'bn'))

    def forward(self, x, mode):
        if mode == 'FNN':
            pred_fnn = self.fnn(x)
            return pred_fnn
        elif mode == 'INN':
            layer_m = self.inn(x)
            return layer_m
        elif mode == 'tandem':
            layer_m = self.inn(x)
            pred_td = self.fnn(layer_m * (self.structure_max - self.structure_min) + self.structure_min)
            return layer_m, pred_td


    def train(self, design, response,mode):
        design_tensor = torch.tensor(design, dtype=torch.float32)
        response_tensor = torch.tensor(response, dtype=torch.float32)


        if mode == 'FNN':
            # Deliberately no self.fnn.train(): test() puts the modules in eval
            # mode and nothing switches them back, so from the first validation
            # onwards the BatchNorm layers keep their running statistics
            # frozen.  That frozen behaviour is what the design was tuned on.
            self.optimizer_fnn.zero_grad()
            pred_fnn = self.forward(design_tensor,mode)
            loss = self.loss_fn(pred_fnn, response_tensor)
            min_index, c_hat = self.lowest_wavelength(response_tensor)
            min_index, c = self.lowest_wavelength(pred_fnn)
            label_loss = self.loss_fn(c_hat, c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = pred_fnn[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss = self.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)
            loss_sum = (self.w_spectral * loss
                        + self.w_peak_wavelength * label_loss
                        + self.w_peak_intensity * lowest_wavelength_loss)
            loss_sum.backward()
            #add label(low_wavelength) loss
            self.optimizer_fnn.step()
            self.scheduler_fnn.step()
        elif mode == 'INN':
            self.optimizer_inn.zero_grad()
            layer_m = self.forward(response_tensor,mode)
            loss = self.loss_fn(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
            print(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
            loss.backward()
            self.optimizer_inn.step()
            self.scheduler_inn.step()
        elif mode == 'tandem':
            # Frozen BatchNorm statistics, as in the FNN branch above.
            self.optimizer_inn.zero_grad()
            self.eval_mode('FNN')
            layer_m, pred_td = self.forward(response_tensor,mode)
            #loss = self.loss_fn(pred_td, response_tensor)
            min_index, c_hat = self.lowest_wavelength(response_tensor)
            min_index, c = self.lowest_wavelength(pred_td)
            loss = self.loss_fn(pred_td, response_tensor)
            label_loss = self.loss_fn(c_hat,c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = pred_td[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss = self.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)
            #loss2 = 10*loss2
            #loss.backward()
            loss_sum = (self.w_spectral * loss
                        + self.w_peak_wavelength * label_loss
                        + self.w_peak_intensity * lowest_wavelength_loss)
            loss_sum.backward()
            self.optimizer_inn.step()
            self.scheduler_inn.step()
        else:
            raise ValueError('mode should be FNN, INN or tandem.')


    def calculate_rse(self,y_true, y_pred):
        return torch.sum(torch.abs(y_true - y_pred)) / torch.sum(torch.abs(y_true))


    def show_loss(self, design, response,mode):
        design_tensor = torch.tensor(design, dtype=torch.float32)
        response_tensor = torch.tensor(response, dtype=torch.float32)


        if mode == 'FNN':
            pred_fnn = self.forward(design_tensor,mode)
            loss = self.loss_fn(pred_fnn, response_tensor)
            error = self.calculate_rse(response_tensor,pred_fnn )
            min_index, c_hat = self.lowest_wavelength(response_tensor)
            min_index, c = self.lowest_wavelength(pred_fnn)
            label_loss = self.loss_fn(c_hat, c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = pred_fnn[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss = self.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)
            return loss,label_loss,lowest_wavelength_loss,1-error
        elif mode == 'INN':
            layer_m = self.forward(response_tensor,mode)
            loss = self.loss_fn(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
            return loss
        elif mode == 'tandem':
            layer_m, pred_td = self.forward(response_tensor,mode)
            #loss = self.loss_fn(pred_td, response_tensor)
            loss1 = self.loss_fn(pred_td, response_tensor)
            error = self.calculate_rse(response_tensor, pred_td)
            min_index, c_hat = self.lowest_wavelength(response_tensor)
            min_index, c = self.lowest_wavelength(pred_td)
            label_loss = self.loss_fn(c_hat, c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = pred_td[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss = self.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)
            return loss1,label_loss,lowest_wavelength_loss, 1-error
        else:
            raise ValueError('mode should be FNN, INN or tandem.')



    def show_lr(self):
        return self.scheduler.get_last_lr()[-1]

    def test(self, design, response, mode):
        design_tensor = torch.tensor(design, dtype=torch.float32)
        response_tensor = torch.tensor(response, dtype=torch.float32)

        if mode == 'FNN':
            self.eval_mode(mode)
            with torch.no_grad():
                 return self.fnn(design_tensor).detach().cpu().numpy()

            #return self.fnn(design_tensor).detach().numpy()
        elif mode == 'INN':
            self.eval_mode(mode)
            with torch.no_grad():
                return ((self.inn(response_tensor)) * (self.structure_max - self.structure_min) + self.structure_min).detach().cpu().numpy()
        elif mode == 'tandem':
            self.eval_mode(mode)
            with torch.no_grad():
            #return self.td(self.inn(response_tensor)).detach().numpy()
                pre_layer = ((self.inn(response_tensor)) * (self.structure_max - self.structure_min) + self.structure_min)
                response = self.fnn(pre_layer).detach().cpu().numpy()
                return pre_layer.detach().cpu().numpy(),response
        else:
            raise ValueError('mode should be FNN, INN or tandem.')

    def save(self, filename):
        torch.save({
            'inn': self.inn.state_dict(),
            'fnn': self.fnn.state_dict(),
        }, filename)

    def save_FNN(self, filename):
        torch.save(self.fnn.state_dict(), filename)

    def save_INN(self, filename):
        torch.save(self.inn.state_dict(), filename)

    # def copy_fnn(self):
    #     self.td.load_state_dict(self.fnn.state_dict())

    def restore(self, filename):
        state_dicts = torch.load(filename,map_location=device)
        self.inn.load_state_dict(state_dicts['inn'])
        self.fnn.load_state_dict(state_dicts['fnn'])


    def restore_FNN(self, filename):
        # self.inn.load_state_dict(torch.load(filename, map_location=device))
        self.fnn.load_state_dict(torch.load(filename,map_location=device))

    def restore_INN(self, filename):
        self.inn.load_state_dict(torch.load(filename,map_location=device))
        # self.copy_fnn()

    def reset_global_step(self):
        self.global_step = 0

    def eval_mode(self,mode):
        self.training = False
        if mode == 'FNN':
            self.fnn.eval()
        elif mode == 'INN':
            self.inn.eval()
        elif mode == 'tandem':
            self.fnn.eval()
            self.inn.eval()
        #self.td.eval()


if __name__ == '__main__':
    n_input = 3  # 12
    n_classes = 1001  # 3
    fnn_size = [n_input, 32, 64, 128, 256, 512, n_classes]
    inn_size = [n_classes, 512, 256, 128, 64, 32, n_input]
    tandem_net = tandem_network(INN_size=inn_size, FNN_size=fnn_size, training=False)
    tandem_net.to(tandem_net.device)
    data_a_list, data_b_list = load_data('C:/Users/liyux/Desktop/traindata/theta_0')
    data_a_array = []
    for data in data_a_list:
        # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
        values = [data['R'], data['theta'], data['n_host']]
        data_a_array.append(values)
    reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
    X = torch.tensor(data_a_array, dtype=torch.float32).to(tandem_net.device)
    y = torch.tensor(reshaped_data_b_list, dtype=torch.float32).to(tandem_net.device)
    # print(y[0])
    y_process = utils1.process_data(y)
    tandem_net.loss2(y,y_process)



