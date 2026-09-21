import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel groups for the GroupNorm layers that replaced BatchNorm1d.
GROUP_NORM_GROUPS = 8
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_loader import load_data
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split


class CVAE(nn.Module):
    def __init__(self, encoder_size,decoder_size,c_dim):
        super(CVAE, self).__init__()

        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.x_dim = encoder_size[0]
        self.z_dim = decoder_size[0]
        self.c_dim = c_dim
        self.encoder_net = self.build_dense(self.encoder_size,self.c_dim)
        self.decoder_net = self.build_dense(self.decoder_size,self.c_dim)
        # Loss weights; the peak-wavelength position is normalised to [0, 1].
        self.w_reconstruction = 2000.0
        self.w_peak_wavelength = 1.0
        self.w_peak_intensity = 1.0
        # Weight of the prior-sample generation term; 0 reproduces the
        # reconstruction-only objective.
        self.w_generation = 0.0
        self.fc_1 = nn.Linear(self.encoder_size[-2],self.encoder_size[-1])
        self.fc_2 = nn.Linear(self.encoder_size[-2],self.encoder_size[-1])
        self.fc_3 = nn.Linear(self.decoder_size[-2],self.decoder_size[-1])


        # # 编码器网络
        # self.fc1 = nn.Linear(x_dim + c_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        # self.fc31 = nn.Linear(h_dim2, z_dim)
        # self.fc32 = nn.Linear(h_dim2, z_dim)
        #
        # # 解码器网络
        # self.fc4 = nn.Linear(z_dim + c_dim, h_dim2)
        # self.fc5 = nn.Linear(h_dim2, h_dim1)
        # self.fc6 = nn.Linear(h_dim1, x_dim)


    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        # h = F.relu(self.fc1(concat_input))
        # h = F.relu(self.fc2(h))
        h = self.encoder_net(concat_input)
        #return self.fc31(h), self.fc32(h)  # mu, log_var
        return self.fc_1(h),self.fc_2(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        # h = F.relu(self.fc4(concat_input))
        # h = F.relu(self.fc5(h))
        h = self.decoder_net(concat_input)
        #return torch.sigmoid(self.fc6(h))
        return torch.sigmoid(self.fc_3(h))

    def build_dense(self, size,c):
        layers = []
        for i in range(len(size) - 2):
            in_features = size[i] + c if i == 0 else size[i]
            layers.append(nn.Linear(in_features, size[i + 1]))
            # BatchNorm1d, unlike in the tandem network.  GroupNorm
            # normalises within each sample, which erases the sample-to-sample
            # differences the encoder has to compress into the latent code:
            # with GroupNorm the encoder collapsed to a constant mu (0 of 20
            # active latent dimensions, KLD -> 0) and the reconstruction MSE
            # degraded 8x.  BatchNorm uses batch statistics, so the relative
            # differences between samples survive.
            layers.append(nn.BatchNorm1d(size[i + 1]))
            layers.append(nn.ReLU())
        #layers.append(nn.Linear(size[-2], size[-1]))
        return nn.Sequential(*layers)

    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, self.x_dim), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var,z

    # Binary Cross Entropy loss
    def loss_function(self, recon_x, x, mu, log_var, c, verbose=False):
        """2000*reconstruction + w*peak-wavelength + KLD + peak-intensity.

        ``label_loss`` is the peak-wavelength term of Eq. (3) and
        ``lowest_wavelength_loss`` the peak-intensity term of Eq. (4); the
        names are historical.  Both peak positions now come from the
        differentiable sub-grid estimate, so Eq. (3) actually contributes a
        gradient.  The binary cross entropy that used to be computed here was
        never part of the returned objective and has been removed.
        """
        mse = nn.MSELoss()
        min_index, c_hat = self.lowest_wavelength(recon_x)
        label_loss = mse(c_hat, c)
        rows = torch.arange(min_index.shape[0], device=recon_x.device)
        lowest_intensity_x = x[rows, min_index[:, 0]]
        lowest_intensity_recon_x = recon_x[rows, min_index[:, 0]]
        lowest_wavelength_loss = mse(lowest_intensity_x, lowest_intensity_recon_x)
        recon = mse(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        if verbose:
            ac = 1 - torch.mean(torch.mean(torch.abs((recon_x - x) / x), dim=1), dim=0)
            print(f'mse:{recon},KLD:{KLD},label_loss:{label_loss},'
                  f'lowest_wavelength_loss:{lowest_wavelength_loss},accuracy:{ac:.5%}')
        return (self.w_reconstruction * recon
                + self.w_peak_wavelength * label_loss
                + KLD
                + self.w_peak_intensity * lowest_wavelength_loss)

    def generation_loss(self, condition, n_samples=None):
        """Peak-wavelength error of spectra decoded from noise plus a condition.

        The reconstruction path cannot teach the decoder to obey the condition:
        the encoder already puts the dip positions into z, so ``label_loss`` is
        satisfied without the condition being used at all.  At inference z is
        sampled from the prior and carries no dip information, which is why the
        generated candidates ignored the requested wavelengths.  Sampling z
        from the prior here and scoring only the condition closes that gap.
        """
        if n_samples is None:
            n_samples = condition.shape[0]
        condition = condition[:n_samples]
        z = torch.randn(n_samples, self.z_dim, device=condition.device)
        generated = self.decoder(z, condition)
        _, position = self.lowest_wavelength(generated)
        return nn.MSELoss()(position.to(condition.device), condition)

    def lowest_wavelength(self, x):
        """Normalised dip position, differentiable with respect to ``x``.

        The plain ``argmin`` index is quantised to the wavelength grid and
        carries no gradient; interpolating the parabola through the three
        samples around it makes the position a smooth function of the spectrum.
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
        return min_index, (clamped.to(x.dtype) + delta) / (n - 1)





# # 生成一些随机数据和标签
# n_samples = 1000
# x = torch.randn(n_samples, x_dim)
# labels = F.one_hot(torch.randint(0, 10, (n_samples,)), num_classes=c_dim).float()
#
# # 划分训练集和验证集
# dataset = TensorDataset(x, labels)
# n_train = int(len(dataset) * 0.8)
# n_val = len(dataset) - n_train
# train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


