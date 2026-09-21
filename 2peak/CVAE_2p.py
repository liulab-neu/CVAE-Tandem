import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel groups for the GroupNorm layers that replaced BatchNorm1d.
GROUP_NORM_GROUPS = 8
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_loader_2P import load_data
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks


class CVAE(nn.Module):
    def __init__(self, encoder_size, decoder_size, c_dim):
        super(CVAE, self).__init__()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model parameters
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.x_dim = encoder_size[0]
        self.z_dim = decoder_size[0]
        self.c_dim = c_dim

        # Model layers
        self.encoder_net = self.build_dense(self.encoder_size, self.c_dim).to(self.device)
        self.decoder_net = self.build_dense(self.decoder_size, self.c_dim).to(self.device)
        # Loss weights; peak positions are normalised to [0, 1].
        self.w_reconstruction = 2000.0
        self.w_peak_wavelength = 1.0
        self.w_peak_intensity = 1.0
        # Weight of the prior-sample generation term; 0 reproduces the
        # reconstruction-only objective.
        self.w_generation = 0.0
        self.fc_1 = nn.Linear(self.encoder_size[-2], self.encoder_size[-1]).to(self.device)
        self.fc_2 = nn.Linear(self.encoder_size[-2], self.encoder_size[-1]).to(self.device)
        self.fc_3 = nn.Linear(self.decoder_size[-2], self.decoder_size[-1]).to(self.device)





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
            if i == 0:
                layers.append(nn.Linear(size[i]+c, size[i + 1]))
                layers.append(nn.BatchNorm1d(size[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(size[i], size[i + 1]))
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

        The peak-intensity term of Eq. (4) was missing from the dual-peak
        objective; it is included here so both cases optimise the same thing.
        """
        mse = nn.MSELoss()
        min_index, c_hat = self.lowest_wavelength(recon_x)
        c_hat = c_hat.to(self.device)
        c = c.to(self.device)
        label_loss = mse(c_hat, c)
        rows = torch.arange(min_index.shape[0], device=recon_x.device).reshape(-1, 1)
        peak_intensity = mse(x[rows, min_index], recon_x[rows, min_index])
        recon = mse(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        if verbose:
            ac = 1 - torch.mean(torch.mean(torch.abs((recon_x - x) / x), dim=1), dim=0)
            print(f'mse:{recon},KLD:{KLD},label_loss:{label_loss},'
                  f'peak_intensity:{peak_intensity},accuracy:{ac:.5%}')
        return (self.w_reconstruction * recon
                + self.w_peak_wavelength * label_loss
                + KLD
                + self.w_peak_intensity * peak_intensity)

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
        """Normalised positions of the two deepest dips, differentiable in ``x``.

        ``find_peaks`` only locates the candidate samples; the returned
        position is then refined by interpolating the parabola through the
        three samples around each dip, which makes it a smooth function of the
        spectrum values so the peak-wavelength term contributes a gradient.

        Two changes to the previous behaviour: the pair is sorted by
        wavelength, so ``c_hat`` and ``c`` always compare peak-to-peak instead
        of being ordered by depth, and spectra where ``find_peaks`` returns
        nothing fall back to the global minimum instead of leaving an
        uninitialised ``np.empty`` entry.
        """
        detached = x.detach().cpu().numpy()
        n = x.shape[1]
        located = np.empty((len(x), 2), dtype=np.int64)
        for i, row in enumerate(detached):
            # Same prominence gate as tandem_2p and the evaluation, so the
            # condition, the loss and the metric all mean the same thing by
            # "the two dips".
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








