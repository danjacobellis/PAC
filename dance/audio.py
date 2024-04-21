import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset, concatenate_datasets
from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from compressai.layers import AttentionBlock
from compressai.ops.parametrizers import NonNegativeParametrizer


class ResidualBottleneckBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, kernel_size=1)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity

class GDN_1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)
    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1)
        norm = F.conv1d(x**2, gamma, beta)
        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        out = x * norm
        return out

def analysis_1d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def synthesis_1d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class AttentionBlock1D(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(N, N//2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(N//2, N//2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(N//2, N, kernel_size=1),
                )
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            nn.Conv1d(N, N, kernel_size=1),
        )
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class RateDistortionAutoEncoder(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            analysis_1d(7, 16),
            GDN_1d(16),
            analysis_1d(16, 32),
            GDN_1d(32),
            analysis_1d(32, 64),
            GDN_1d(64),
            analysis_1d(64, N),
        )

        self.decode = nn.Sequential(
            AttentionBlock1D(N),
            synthesis_1d(N, N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            synthesis_1d(N, N),
            AttentionBlock1D(N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            synthesis_1d(N, N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            ResidualBottleneckBlock1D(N, N),
            synthesis_1d(N, 7),
            torch.nn.Hardtanh(min_val=-0.5, max_val=0.5),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods