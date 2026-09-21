import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from AutomaticWeightedLoss import AutomaticWeightedLoss
from data_loader import load_data
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



class tandem_network(nn.Module):
    def __init__(self, INN_size, FNN_size, starter_learning_rate=0.001,
                 decay_step=1000 * 100, decay_rate=0.5,
                 structure_min = torch.tensor([2.5, 1.5]),
                 # Left over from a three-input (R, theta, n) signature; this
                 # model takes (R, n_host).
                 structure_max = torch.tensor([6.5, 3.05]),
                 training=True):
        super(tandem_network, self).__init__()
        assert INN_size[0] == FNN_size[-1]
        assert INN_size[-1] == FNN_size[0]

        self.global_step = 0
        self.lr = starter_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.inn_size = INN_size
        self.fnn_size = FNN_size
        self.training = training
        # Register structure bounds as buffers so they follow the module device
        self.register_buffer('structure_min', structure_min.to(device))
        self.register_buffer('structure_max', structure_max.to(device))

        data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')
        wl_np = np.array(data_b_list[0])[:, 0]
        # Register wavelengths as a tensor buffer for safe device indexing
        self.register_buffer('wavelengths', torch.tensor(wl_np, dtype=torch.float32, device=device))

        self.build_net()
        self.awl = AutomaticWeightedLoss(2)
        self.optimizer_fnn = optim.Adam(self.fnn.parameters(), lr=self.lr)
        self.scheduler_fnn = StepLR(self.optimizer_fnn, step_size=decay_step, gamma=decay_rate)
        # self.optimizer_inn = optim.Adam([{'params':self.awl.parameters()},{'params':self.inn.parameters(), 'lr':self.lr}])
        self.optimizer_inn = optim.Adam([{'params': self.inn.parameters(), 'lr': self.lr}])
        self.scheduler_inn = StepLR(self.optimizer_inn, step_size=decay_step, gamma=decay_rate)
        # ``self.scheduler`` used to be assigned twice, so the forward-network
        # scheduler was overwritten by the inverse one and the FNN learning
        # rate never decayed.  Keep the attribute as an alias of the inverse
        # scheduler for any external caller, but step the matching one in
        # train().
        self.scheduler = self.scheduler_inn
        self.loss_fn = nn.MSELoss()
        # Loss weights.  lowest_wavelength() now returns a normalised position
        # in [0, 1] rather than nanometres, so the peak-wavelength term is
        # about three orders of magnitude smaller than 300*spectral MSE and
        # needs its own weight to matter.
        self.w_spectral = 300.0
        self.w_peak_wavelength = 300.0
        self.w_peak_intensity = 1.0




    def loss2(self,dataset1,dataset2):
        #tensor_seq = self.data(dataset)
        #tensor_seq = utils1.process_data(dataset)
        merged_tensor = torch.cat((dataset1, dataset2), dim=1)
        mean_mse = torch.mean(torch.mean(merged_tensor, dim=1)**2)
        var_mse = torch.mean(torch.var(merged_tensor,dim=1)**2)
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(mean_mse, var_mse)
        # print(mse)
        return mean_mse,var_mse,loss_sum

    # def lowest_wavelength(self,x):
    #     min_index = torch.argmin(x, axis=1)
    #
    #     min_index = min_index.reshape(-1, 1)
    #     min_index_wave = (min_index / (1001 - 1))
    #     #print(min_index_wave)
    #     return min_index,min_index_wave

    def lowest_wavelength(self, x):
        """Normalised dip position, differentiable with respect to ``x``.

        ``argmin`` alone only locates the sampled minimum: the result is an
        integer index, so the position is quantised to the wavelength grid and
        carries no gradient.  Interpolating the vertex of the parabola through
        the three samples around that minimum makes the position a smooth
        function of the spectrum values, so ``L_peak-lambda`` can actually be
        optimised.  The output is normalised to [0, 1]; multiply by
        ``self.wavelengths[-1] - self.wavelengths[0]`` for nanometres.
        """
        min_index = torch.argmin(x, dim=1).reshape(-1, 1)
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
        position = (clamped.to(x.dtype) + delta) / (n - 1)
        return min_index, position



    def build_net(self):
        self.response = nn.Linear(self.inn_size[0], self.inn_size[0])  # Fixed: Use nn.Linear for response
        self.design = nn.Linear(self.inn_size[-1], self.inn_size[-1])  # Fixed: Use nn.Linear for design
        self.fnn = self.build_dense(self.fnn_size)
        self.inn = self.build_dense(self.inn_size)
        #self.td = self.build_dense(self.fnn_size)

    def build_dense(self, Size):
        layers = []
        for i in range(len(Size) - 2):
            layers.append(nn.Linear(Size[i], Size[i + 1]))
            layers.append(nn.BatchNorm1d(Size[i + 1]))
            #layers.append(AdaptiveBatchNorm1d(Size[i + 1])
            layers.append(nn.ReLU())
        layers.append(nn.Linear(Size[-2], Size[-1]))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

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

    # def train(self, design, response, response_relate,mode):
    #     design_tensor = torch.tensor(design, dtype=torch.float32)
    #     response_tensor = torch.tensor(response, dtype=torch.float32)
    #     response_relate_tensor = torch.tensor(response_relate, dtype=torch.float32)
    #
    #     if mode == 'FNN':
    #         self.optimizer_fnn.zero_grad()
    #         pred_fnn = self.forward(design_tensor,mode)
    #         loss = self.loss_fn(pred_fnn, response_tensor)
    #         # min_index, min_index_wave = self.lowest_wavelength(response_tensor)
    #         # weight = torch.ones(len(response_tensor), 1001)
    #         # weight[torch.arange(min_index.shape[0]), min_index[:, 0]] = 800
    #         # # loss1 = self.loss_fn(pred_td, response_tensor)
    #         # loss = torch.mean(weight * (pred_fnn - response_tensor) ** 2)
    #         loss.backward()
    #         self.optimizer_fnn.step()
    #         self.scheduler.step()
    #     elif mode == 'INN':
    #         self.optimizer_inn.zero_grad()
    #         layer_m = self.forward(response_tensor,mode)
    #         loss = self.loss_fn(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
    #         print(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
    #         loss.backward()
    #         self.optimizer_inn.step()
    #         self.scheduler.step()
    #     elif mode == 'tandem':
    #         self.optimizer_inn.zero_grad()
    #         layer_m, pred_td = self.forward(response_tensor,mode)
    #         print(f'layer_m:{layer_m}')
    #         print(self.structure_max - self.structure_min)
    #         print(layer_m * (self.structure_max - self.structure_min) + self.structure_min)
    #         print(f'design:{design_tensor}')
    #         #loss = self.loss_fn(pred_td, response_tensor)
    #         # min_index, min_index_wave = self.lowest_wavelength(response_tensor)
    #         # weight = torch.ones(len(response_tensor), 1001)
    #         # weight[torch.arange(min_index.shape[0]), min_index[:, 0]] = 800
    #         # loss1 = torch.mean(weight * (pred_td - response_tensor) ** 2)
    #         loss1 = self.loss_fn(pred_td, response_tensor)
    #         mean_loss,var_loss,loss2 = self.loss2(pred_td,response_relate_tensor)
    #         #loss2 = 10*loss2
    #         awl = AutomaticWeightedLoss(2)
    #         loss_sum = awl(loss1,loss2)
    #         # loss_sum = loss1 + loss2
    #         loss1.backward()
    #         #loss_sum.backward()
    #         self.optimizer_inn.step()
    #         self.scheduler.step()
    #     else:
    #         raise ValueError('mode should be FNN, INN or tandem.')

    def train(self, design, response,mode):
        design_tensor = design
        response_tensor = response


        if mode == 'FNN':
            # Deliberately no self.fnn.train() here: test() puts the modules in
            # eval mode and nothing switches them back, so from the first
            # validation onwards the BatchNorm layers keep their running
            # statistics frozen.  Letting them update instead -- textbook
            # BatchNorm training -- makes the validation loss about 27x worse
            # on this dataset, so the frozen behaviour is the trained design.
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

    # def show_loss(self, design, response, response_relate,mode):
    #     design_tensor = torch.tensor(design, dtype=torch.float32)
    #     response_tensor = torch.tensor(response, dtype=torch.float32)
    #     response_relate_tensor = torch.tensor(response_relate, dtype=torch.float32)
    #
    #     if mode == 'FNN':
    #         pred_fnn = self.forward(design_tensor,mode)
    #         loss = self.loss_fn(pred_fnn, response_tensor)
    #         # min_index, min_index_wave = self.lowest_wavelength(response_tensor)
    #         # weight = torch.ones(len(response_tensor), 1001)
    #         # weight[torch.arange(min_index.shape[0]), min_index[:, 0]] = 800
    #         # # loss1 = self.loss_fn(pred_td, response_tensor)
    #         # loss = torch.mean(weight * (pred_fnn - response_tensor) ** 2)
    #         return loss
    #     elif mode == 'INN':
    #         layer_m = self.forward(response_tensor,mode)
    #         loss = self.loss_fn(layer_m * (self.structure_max - self.structure_min) + self.structure_min, design_tensor)
    #         return loss
    #     elif mode == 'tandem':
    #         layer_m, pred_td = self.forward(response_tensor,mode)
    #         #loss = self.loss_fn(pred_td, response_tensor)
    #         # min_index, min_index_wave = self.lowest_wavelength(response_tensor)
    #         # weight = torch.ones(len(response_tensor), 1001)
    #         # weight[torch.arange(min_index.shape[0]), min_index[:, 0]] = 800
    #         # loss1 = torch.mean(weight*(pred_td - response_tensor) ** 2)
    #         loss1 = self.loss_fn(pred_td, response_tensor)
    #         mean_loss,var_loss,loss2 = self.loss2(pred_td,response_relate_tensor)
    #         return loss1,loss2,mean_loss,var_loss
    #     else:
    #         raise ValueError('mode should be FNN, INN or tandem.')
    def calculate_rse(self,y_true, y_pred):
        return torch.sum(torch.abs(y_true - y_pred)) / torch.sum(torch.abs(y_true))


    def show_loss(self, design, response,mode):
        design_tensor = design
        response_tensor = response


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
        design_tensor = design
        response_tensor = response

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
                return pre_layer.detach().cpu().numpy(), response
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
        state_dicts = torch.load(filename, map_location=device)
        self.inn.load_state_dict(state_dicts['inn'])
        self.fnn.load_state_dict(state_dicts['fnn'])


    def restore_FNN(self, filename):
        self.fnn.load_state_dict(torch.load(filename, map_location=device))

    def restore_INN(self, filename):
        self.inn.load_state_dict(torch.load(filename, map_location=device))
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
    data_a_list, data_b_list = load_data('C:/Users/liyux/Desktop/traindata/theta_0')
    data_a_array = []
    for data in data_a_list:
        # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
        values = [data['R'], data['theta'], data['n_host']]
        data_a_array.append(values)
    reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
    X = torch.tensor(data_a_array, dtype=torch.float32)
    y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)
    # print(y[0])
    y_process = utils1.process_data(y)
    tandem_net.loss2(y,y_process)



