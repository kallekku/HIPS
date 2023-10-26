"""
Implements the different neural networks
"""
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import GroupOfBlocks, FilmResBlock, ResNetBlock, vq, vq_st
from .drc import DRC

device = "cuda" if torch.cuda.is_available() else "cpu"


class STPDetectorNetwork(nn.Module):
    """
    Detector for STP
    """
    def __init__(self, n_channels, norm=nn.BatchNorm2d, input_nc=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_nc*2, out_channels=n_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm(n_channels)
        self.group = GroupOfBlocks(n_channels, n_channels//2, 6, norm=norm)

        self.converter = nn.Linear(25*n_channels//2, 128)
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, compute_val=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.group(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.converter(x))

        pol = self.policy_head(x).squeeze(dim=1)
        if compute_val:
            val = self.value_head(x)
            return pol, val
        else:
            return pol


class SokobanDetectorNetwork(nn.Module):
    """
    Detector for Sokoban
    """
    def __init__(self, norm=nn.BatchNorm2d, input_nc=3):
        super().__init__()
        self.in_block = nn.Sequential(
            ResNetBlock(in_channels=input_nc*2, out_channels=16, kernel_size=5,
                        padding=2, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=12, kernel_size=3,
                        padding=1, norm=norm))

        self.converter = nn.Linear(1200, 128)
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, compute_val=True):
        x = self.in_block(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.converter(x))

        pol = self.policy_head(x).squeeze(dim=1)
        if compute_val:
            val = self.value_head(x)
            return pol, val
        else:
            return pol


class STPPolicy(nn.Module):
    """
    Conditional low-level policy for STP
    """
    def __init__(self, n_blocks, n_channels, num_classes, norm=nn.BatchNorm2d, input_nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_nc*2, out_channels=n_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm(n_channels)

        self.group = GroupOfBlocks(n_channels, n_channels, n_blocks[0],
                                   norm=norm)

        self.fc = nn.Sequential(
            nn.Linear(25*n_channels, 4*n_channels),
            nn.ReLU(),
            nn.Linear(4*n_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.group(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class SokobanPolicy(nn.Module):
    """
    Conditional low-level policy for Sokoban
    """
    def __init__(self, n_blocks, n_channels, num_classes, norm=nn.BatchNorm2d, input_nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_nc*2, out_channels=n_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm(n_channels)

        self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks[0],
                                    norm=norm)
        self.group2 = GroupOfBlocks(n_channels, 2*n_channels, n_blocks[1],
                                    stride=2, norm=norm)
        self.group3 = GroupOfBlocks(2*n_channels, 4*n_channels, n_blocks[2],
                                    stride=2, norm=norm)

        self.fc = nn.Sequential(
            nn.Linear(36*n_channels, 4*n_channels),
            nn.ReLU(),
            nn.Linear(4*n_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class SokobanModel(nn.Module):
    """
    Dynamics model for Sokoban
    """
    def __init__(self, dim, channels, out_dim, expand=False):
        super().__init__()
        self.act_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.obs_block = nn.Sequential(
            GroupOfBlocks(channels, dim, 2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, 8, norm=nn.InstanceNorm2d),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, 3*dim if expand else dim, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(3*dim if expand else dim, channels*out_dim, kernel_size=1, padding=0),
        )

        self.out_dim = out_dim
        self.channels = channels

    def forward(self, obs, acts):
        acts = self.act_mlp(acts[:, None])
        obs = self.obs_block(obs)
        for b in self.trunk:
            obs = b(obs, acts)
        out = self.out_block(obs)
        out = out.view(out.shape[0], self.channels, self.out_dim,
                       out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
        return out


# Code modified from
# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class VQEmbedding(nn.Module):
    """
    Implements VQVAE Codebook E
    """
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)
        self.previous_z = []
        self.K = K

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        if not hasattr(self, 'previous_z') or self.previous_z is None:
            self.previous_z = []
        if len(self.previous_z) < 10:
            self.previous_z.append(z_e_x.detach().clone().squeeze())

        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        n_codes = len(set(indices.cpu().numpy()))

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0,
                                               index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, n_codes, indices.cpu().numpy()

    def reinit_codes(self, D=None, K=None):
        if K is not None and D is not None:
            self.K = K
            self.embedding = nn.Embedding(K, D)
            self.embedding.weight.data.uniform_(-1./K, 1./K)
        prev_z = torch.cat(self.previous_z, dim=0)
        kmeans = KMeans(n_clusters=self.K).fit(prev_z.cpu())
        clusters = torch.from_numpy(kmeans.cluster_centers_).to(device)
        self.embedding.load_state_dict({'weight': clusters})


class SokobanPrior(nn.Module):
    """
    Prior for Sokoban, both high-level and low-level (unconditional BC-policy)
    """
    def __init__(self, dim, K, n_acts, input_nc=3):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            nn.Flatten(1, 3),
        )

        self.code_block = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, K),
        )

        self.act_block = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out_c = self.code_block(obs)
        out_a = self.act_block(obs)
        return out_c, out_a


class SokobanEncoder(nn.Module):
    """
    VQVAE Encoder for Sokoban
    """
    def __init__(self, dim, input_nc):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc*2, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
        )

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out = self.out_block(obs)
        out = F.normalize(out)
        return out


class FilmDecoder(nn.Module):
    """
    VQVAE Decoder for Sokoban, STP and BW
    """
    def __init__(self, dim, input_nc, output_dim):
        super().__init__()

        self.in_block = nn.Sequential(
            ResNetBlock(input_nc, dim, norm=nn.InstanceNorm2d),
            ResNetBlock(dim, dim, norm=nn.InstanceNorm2d),
            ResNetBlock(dim, dim, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, 3*dim, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(3*dim, input_nc*output_dim, kernel_size=1, padding=0),
        )

        self.input_nc = input_nc
        self.output_dim = output_dim

    def forward(self, obs, latents):
        if not hasattr(self, 'input_nc'):
            self.input_nc = 3
        if not hasattr(self, 'output_dim'):
            self.output_dim = 256

        x = self.in_block(obs)
        latents = latents.squeeze(3).squeeze(2)
        for b in self.trunk:
            x = b(x, latents)

        out = self.out_block(x)
        out = out.view(out.shape[0], self.input_nc, self.output_dim,
                       out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
        return out


class SokobanVQVAE(nn.Module):
    """
    VQVAE for Sokoban
    """
    # Complete VQVAE model
    def __init__(self, dim, K, input_nc=3, output_dim=256):
        super().__init__()

        self.encoder = SokobanEncoder(dim, input_nc)
        self.codebook = VQEmbedding(K, dim)
        self.decoder = FilmDecoder(dim, input_nc, output_dim)
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, decoder_input, latent_idxs):
        # Generate subgoals given previous state and discrete codes
        z_q_x = self.codebook.embedding(latent_idxs)[:, :, None, None]
        x_tilde = self.decoder(decoder_input, z_q_x)
        return x_tilde

    def reinit_codes(self, D=None, K=None):
        self.codebook.reinit_codes(D, K)

    def forward(self, encoder_input, decoder_input, continuous=False):
        z_e_x = self.encoder(encoder_input)
        if continuous:
            z_q_x_st, z_q_x, n_codes, codes = z_e_x, z_e_x, 1, []
        else:
            z_q_x_st, z_q_x, n_codes, codes = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(decoder_input, z_q_x_st)
        return x_tilde, z_e_x, z_q_x, n_codes, codes


class SokobanDistNetwork(nn.Module):
    """
    Distance function (heuristic) for Sokoban
    """
    def __init__(self, dim, max_len, input_nc=4, instance_norm=False, dropout=False):
        super().__init__()

        norm = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, norm=norm, dropout=dropout),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=norm, dropout=dropout),
            nn.Flatten(1, 3),
            nn.Linear(25*dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_len),
        )
        self.max_len = max_len

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        if not hasattr(self, 'max_len'):
            self.max_len = 220
        with torch.no_grad():
            dists = F.softmax(self.net(x), dim=-1)
            expectation = (dists * torch.from_numpy(np.arange(0, self.max_len))[None].
                           to(device)).sum(dim=-1)
            return expectation.cpu().numpy()

    def forward(self, obs):
        return self.net(obs)


class STPEncoder(nn.Module):
    """
    VQVAE Encoder for STP
    """
    def __init__(self, dim, input_nc):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc*2, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
        )

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 1, 0),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out = self.out_block(obs)
        out = F.normalize(out)
        return out


class STPVQVAE(nn.Module):
    """
    VQVAE for STP
    """
    def __init__(self, dim, K, input_nc=3, output_dim=256):
        super().__init__()

        self.encoder = STPEncoder(dim, input_nc)
        self.codebook = VQEmbedding(K, dim)
        # We can use FilmDecoder
        self.decoder = FilmDecoder(dim, input_nc, output_dim)
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, decoder_input, latent_idxs):
        # Generate subgoals given previous state and discrete codes
        z_q_x = self.codebook.embedding(latent_idxs)[:, :, None, None]
        x_tilde = self.decoder(decoder_input, z_q_x)
        return x_tilde

    def reinit_codes(self, D=None, K=None):
        self.codebook.reinit_codes(D, K)

    def forward(self, encoder_input, decoder_input, continuous=False):
        z_e_x = self.encoder(encoder_input)
        if continuous:
            z_q_x_st, z_q_x, n_codes, codes = z_e_x, z_e_x, 1, []
        else:
            z_q_x_st, z_q_x, n_codes, codes = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(decoder_input, z_q_x_st)
        return x_tilde, z_e_x, z_q_x, n_codes, codes


class STPPrior(nn.Module):
    """
    Prior for STP, both high-level and low-level (unconditional BC-policy)
    """
    def __init__(self, dim, K, n_acts, input_nc=3):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            nn.Flatten(1, 3),
        )

        self.code_block = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, K),
        )

        self.act_block = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out_c = self.code_block(obs)
        out_a = self.act_block(obs)
        return out_c, out_a


class STPDistNetwork(nn.Module):
    """
    Distance function (heuristic) for STP
    """
    # STP Prior
    def __init__(self, dim, max_len, input_nc=25, instance_norm=False, dropout=False):
        super().__init__()

        norm = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, norm=norm, dropout=dropout),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=norm, dropout=dropout),
            nn.Flatten(1, 3),
            nn.Linear(9*dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_len),
        )
        self.max_len = max_len

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        if not hasattr(self, 'max_len'):
            self.max_len = 300
        with torch.no_grad():
            dists = F.softmax(self.net(x), dim=-1)
            expectation = (dists * torch.from_numpy(np.arange(0, self.max_len))[None].
                           to(device)).sum(dim=-1)
            return expectation.cpu().numpy()

    def forward(self, obs):
        return self.net(obs)


class STPModel(nn.Module):
    """
    Dynamics model for STP
    """
    def __init__(self, dim, channels, out_dim, expand=False):
        super().__init__()
        self.act_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.obs_block = nn.Sequential(
            GroupOfBlocks(channels, dim, 2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, 8, norm=nn.InstanceNorm2d),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, 3*dim if expand else dim, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(3*dim if expand else dim, channels*out_dim, kernel_size=1, padding=0),
        )

        self.out_dim = out_dim
        self.channels = channels

    def forward(self, obs, acts):
        acts = self.act_mlp(acts[:, None])
        obs = self.obs_block(obs)
        for b in self.trunk:
            obs = b(obs, acts)
        out = self.out_block(obs)
        out = out.view(out.shape[0], self.channels, self.out_dim,
                       out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
        return out


class BWDetectorNetwork(nn.Module):
    """
    Detector for Box-World
    """
    def __init__(self, N, D, norm=nn.BatchNorm2d, input_nc=3):
        super().__init__()
        self.in_block = nn.Sequential(
            GroupOfBlocks(input_nc*2, 32, 2, norm=norm, stride=1),
            GroupOfBlocks(32, 48, 2, norm=norm, stride=2),
            GroupOfBlocks(48, 64, 2, norm=norm, stride=2),
        )
        self.drc = DRC(D, N, 64, 64)

        self.converter = nn.Linear(1024, 128)
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, compute_val=True):
        x = self.in_block(x)
        x, h = self.drc(x, None)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.converter(x))

        pol = self.policy_head(x).squeeze(dim=1)
        if compute_val:
            val = self.value_head(x)
            return pol, val
        else:
            return pol


class BWEncoder(nn.Module):
    """
    VQVAE Encoder for Box-World
    """
    def __init__(self, dim, input_nc):
        super().__init__()

        norm = nn.BatchNorm2d

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc*2, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=norm),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
        )

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim, 2, stride=2, norm=norm),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=norm),
            nn.Conv2d(dim, dim, 4, 1, 1),
            norm(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 0),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out = self.out_block(obs)
        out = F.normalize(out)
        return out


class BWVQVAE(nn.Module):
    """
    VQVAE for Box-World
    """
    def __init__(self, dim, K, input_nc=3, output_dim=256):
        super().__init__()

        self.encoder = BWEncoder(dim, input_nc)
        self.codebook = VQEmbedding(K, dim)
        self.decoder = FilmDecoder(dim, input_nc, output_dim)
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, decoder_input, latent_idxs):
        # Generate subgoals given previous state and discrete codes
        z_q_x = self.codebook.embedding(latent_idxs)[:, :, None, None]
        x_tilde = self.decoder(decoder_input, z_q_x)
        return x_tilde

    def reinit_codes(self, D=None, K=None):
        self.codebook.reinit_codes(D, K)

    def forward(self, encoder_input, decoder_input, continuous=False):
        z_e_x = self.encoder(encoder_input)
        if continuous:
            z_q_x_st, z_q_x, n_codes, codes = z_e_x, z_e_x, 1, []
        else:
            z_q_x_st, z_q_x, n_codes, codes = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(decoder_input, z_q_x_st)
        return x_tilde, z_e_x, z_q_x, n_codes, codes


class BWPolicy(nn.Module):
    """
    Conditional low-level policy for Box-World
    """
    def __init__(self, D, N, norm=nn.BatchNorm2d, n_actions=4):
        super().__init__()
        self.fe1 = GroupOfBlocks(6, 32, 2, norm=norm, stride=1)
        self.fe2 = GroupOfBlocks(32, 48, 2, norm=norm, stride=2)
        self.fe3 = GroupOfBlocks(48, 64, 2, norm=norm, stride=2)
        self.drc = DRC(D, N, 64, 64)
        self.head = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x, h=None, ret_hid=False, verbose=False):
        if verbose:
            print(x.shape)
        x = self.fe1(x)
        if verbose:
            print(x.shape)
        x = self.fe2(x)
        if verbose:
            print(x.shape)
        x = self.fe3(x)
        if verbose:
            print(x.shape)
        x, h = self.drc(x, h)
        if verbose:
            print(x.shape)
        x = self.head(x)
        if verbose:
            print(x.shape)
        if ret_hid:
            return x, h
        else:
            return x


class BWPrior(nn.Module):
    """
    Prior for BW, both high-level and low-level (unconditional BC-policy)
    """
    def __init__(self, dim, K, n_acts, input_nc=3, D=3, N=3):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=nn.InstanceNorm2d),
        )

        self.drc = DRC(D, N, dim, dim)

        self.code_block = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, K),
        )

        self.act_block = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        obs, h = self.drc(obs, None)
        obs = obs.view(obs.shape[0], -1)
        out_c = self.code_block(obs)
        out_a = self.act_block(obs)
        return out_c, out_a


class BWModel(nn.Module):
    """
    Dynamics model for BW
    """
    def __init__(self, dim, channels, out_dim, expand=False, D=3, N=3, norm=nn.InstanceNorm2d):
        super().__init__()
        self.act_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.obs_block = nn.Sequential(
            GroupOfBlocks(channels, dim, 2, norm=norm),
            GroupOfBlocks(dim, dim, 2, norm=norm),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, 8, norm=norm),
            FilmResBlock(dim, dim, 8, norm=norm),
            FilmResBlock(dim, dim, 8, norm=norm),
            FilmResBlock(dim, dim, 8, norm=norm),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, 3*dim if expand else dim, 2, norm=norm),
            nn.Conv2d(3*dim if expand else dim, channels*out_dim, kernel_size=1, padding=0),
        )

        self.out_dim = out_dim
        self.channels = channels

    def forward(self, obs, acts):
        acts = self.act_mlp(acts[:, None])
        obs = self.obs_block(obs)
        for b in self.trunk:
            obs = b(obs, acts)
        out = self.out_block(obs)
        out = out.view(out.shape[0], self.channels, self.out_dim,
                       out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
        return out


class BWDistNetwork(nn.Module):
    """
    Distance function (heuristic) for BW
    """
    def __init__(self, dim, max_len, input_nc=3, instance_norm=False, dropout=False, N=3, D=3):
        super().__init__()

        norm = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=norm, dropout=dropout),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=norm, dropout=dropout),
        )

        self.drc = DRC(D, N, dim, dim)

        self.out = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(16*dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_len),
        )
        self.max_len = max_len

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        if not hasattr(self, 'max_len'):
            self.max_len = 75
        with torch.no_grad():
            dists = F.softmax(self.__call__(x), dim=-1)
            expectation = (dists * torch.from_numpy(np.arange(0, self.max_len))[None].
                           to(device)).sum(dim=-1)
            return expectation.cpu().numpy()

    def forward(self, obs):
        obs = self.net(obs)
        obs, h = self.drc(obs, None)
        obs = self.out(obs)
        return obs


class TSPDistNetwork(nn.Module):
    """
    Distance function (heuristic) for TSP
    """
    def __init__(self, dim, max_len, input_nc=4, instance_norm=False, dropout=False):
        super().__init__()

        norm = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=norm, dropout=dropout),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=norm, dropout=dropout),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=norm, dropout=dropout),
            nn.Flatten(1, 3),
            nn.Linear(16*dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, max_len),
        )
        self.max_len = max_len

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            dists = F.softmax(self.net(x), dim=-1)
            expectation = (dists * torch.from_numpy(np.arange(0, self.max_len))[None].
                           to(device)).sum(dim=-1)
            return expectation.cpu().numpy()

    def forward(self, obs):
        return self.net(obs)


class TSPEncoder(nn.Module):
    """
    VQVAE Encoder for TSP
    """
    def __init__(self, dim):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(8, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
        )

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 2, stride=1, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, obs, acts=None):
        obs = self.in_block(obs)

        ttl_input = obs

        out = self.out_block(ttl_input)
        out = F.normalize(out)
        return out


class TSPFilmDecoder(nn.Module):
    """
    VQVAE Decoder for TSP
    """
    def __init__(self, dim):
        super().__init__()

        self.in_block = nn.Sequential(
            ResNetBlock(4, dim, norm=nn.InstanceNorm2d),
            ResNetBlock(dim, dim, norm=nn.InstanceNorm2d),
            ResNetBlock(dim, dim, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
            FilmResBlock(dim, dim, dim, norm=nn.InstanceNorm2d),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim//2, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim//2, 4, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, obs, acts, latents):
        x = self.in_block(obs)
        latents = latents.squeeze(3).squeeze(2)
        for b in self.trunk:
            x = b(x, latents)
        out = self.out_block(x)
        return out


class TSPVQVAE(nn.Module):
    """
    VQVAE for TSP
    """
    def __init__(self, dim, K):
        super().__init__()

        self.encoder = TSPEncoder(dim)
        self.codebook = VQEmbedding(K, dim)
        self.decoder = TSPFilmDecoder(dim)
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, decoder_input, latent_idxs):
        z_q_x = self.codebook.embedding(latent_idxs)[:, :, None, None]
        x_tilde = self.decoder(decoder_input, None, z_q_x)
        return x_tilde

    def reinit_codes(self, D=None, K=None):
        self.codebook.reinit_codes(D, K)

    def forward(self, encoder_input, decoder_input, acts=None, continuous=False):
        z_e_x = self.encoder(encoder_input, acts)
        if continuous:
            z_q_x_st, z_q_x, n_codes, codes = z_e_x, z_e_x, 1, []
        else:
            z_q_x_st, z_q_x, n_codes, codes = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(decoder_input, acts, z_q_x_st)
        return x_tilde, z_e_x, z_q_x, n_codes, codes


class TSPPrior(nn.Module):
    """
    Prior for TSP, both high-level and low-level (unconditional BC-policy)
    """
    def __init__(self, dim, K, n_acts, input_nc=4):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, stride=2, norm=nn.InstanceNorm2d),
            nn.Flatten(1, 3),
        )

        self.code_block = nn.Sequential(
            nn.Linear(16*dim, 256),
            nn.ReLU(),
            nn.Linear(256, K),
        )

        self.act_block = nn.Sequential(
            nn.Linear(16*dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out_c = self.code_block(obs)
        out_a = self.act_block(obs)
        return out_c, out_a


class TSPPolicy(nn.Module):
    """
    Conditional low-level policy for TSP
    """
    def __init__(self, n_blocks, n_channels, num_classes, norm=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=n_channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn1 = norm(n_channels)

        self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks[0], norm=norm)
        self.group2 = GroupOfBlocks(n_channels, 2*n_channels, n_blocks[1], stride=2, norm=norm)
        self.group3 = GroupOfBlocks(2*n_channels, 4*n_channels, n_blocks[2], stride=2, norm=norm)

        self.fc = nn.Sequential(
            nn.Linear(196*n_channels, 4*n_channels),
            nn.ReLU(),
            nn.Linear(4*n_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class TSPDetectorNetwork(nn.Module):
    """
    Detector for TSP
    """
    def __init__(self, norm=nn.BatchNorm2d):
        super().__init__()
        self.in_block = nn.Sequential(
            ResNetBlock(in_channels=8, out_channels=16, kernel_size=5,
                        padding=2, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=16, kernel_size=3,
                        padding=1, norm=norm),
            ResNetBlock(in_channels=16, out_channels=12, kernel_size=3,
                        padding=1, norm=norm))

        self.converter = nn.Linear(7500, 128)
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, compute_val=True):
        x = self.in_block(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.converter(x))

        pol = self.policy_head(x).squeeze(dim=1)
        if compute_val:
            val = self.value_head(x)
            return pol, val
        else:
            return pol


class TSPModel(nn.Module):
    """
    Dynamics model for TSP
    """
    def __init__(self, dim):
        super().__init__()
        self.act_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.obs_block = nn.Sequential(
            GroupOfBlocks(4, dim, 2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList([
            FilmResBlock(dim, dim, 8, norm=nn.InstanceNorm2d),
        ])

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, dim//2, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(dim//2, 4, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, obs, acts):
        acts = self.act_mlp(acts[:, None])
        obs = self.obs_block(obs)
        for b in self.trunk:
            obs = b(obs, acts)
        out = self.out_block(obs)
        return out
