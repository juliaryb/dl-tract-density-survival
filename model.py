# """
# 3D convolutional autoencoder for Tract Density Maps.

# Architecture overview
# ---------------------
# Encoder : N strided Conv3d blocks (stride=2) reduce spatial dims by 2^N,
#           followed by flatten + Linear to produce a 1-D latent vector.

# Decoder : Linear + reshape, then N transposed Conv3d blocks reconstruct
#           the original spatial volume.

# Design choices
# --------------
# - Strided convolutions instead of pooling: preserves spatial location
#   information in the latent vector, which is hypothesis-critical (the
#   spatial pattern of white matter tract involvement matters for survival).
# - Flatten + Linear bottleneck instead of global average pooling: each
#   spatial position remains a distinct input to the linear layer, so
#   the encoder can learn location-specific features.
# - LeakyReLU throughout (except final decoder layer): z-scored inputs
#   produce negative activations that ReLU would zero out permanently
#   (dying neurons); LeakyReLU keeps gradients flowing.
# - No activation on the final decoder output: reconstruction targets are
#   continuous and unbounded (z-scored values can be negative), so any
#   output activation would incorrectly constrain the range.
# - Input spatial dims must be divisible by 2^len(channels).
#   Use bbox_to_padded_shape() from utils.py to ensure this.

# Customisation
# -------------
# Pass channels=(8, 16, 32, 64) to get the default 4-layer architecture.
# Wider network: channels=(16, 32, 64, 128).
# Shallower network: channels=(16, 32).
# Bottleneck size is controlled independently via latent_dim (e.g. 2, 64, 128).
# """

# import math
# import torch
# import torch.nn as nn


# class Encoder(nn.Module):
#     """
#     Maps (B, 1, D, H, W) → (B, latent_dim).

#     Parameters
#     ----------
#     input_shape : (D, H, W) spatial dimensions of the input volume.
#                   Each dim must be divisible by 2**len(channels).
#     latent_dim  : Length of the output latent vector.
#     channels    : Channel counts for each strided conv block.
#                   len(channels) determines network depth.
#     """

#     def __init__(
#         self,
#         input_shape: tuple[int, int, int],
#         latent_dim: int = 128,
#         channels: tuple[int, ...] = (8, 16, 32, 64),
#     ) -> None:
#         super().__init__()

#         D, H, W = input_shape
#         stride_total = 2 ** len(channels)
#         assert all(s % stride_total == 0 for s in (D, H, W)), (
#             f"input_shape {input_shape} must be divisible by {stride_total} "
#             f"(= 2**{len(channels)} for {len(channels)}-layer encoder). "
#             "Use bbox_to_padded_shape() from utils.py."
#         )

#         in_chs = [1] + list(channels[:-1])
#         self.blocks = nn.Sequential(
#             *[self._conv_block(in_c, out_c) for in_c, out_c in zip(in_chs, channels)]
#         )

#         self._spatial = (D // stride_total, H // stride_total, W // stride_total)
#         flat_size = channels[-1] * math.prod(self._spatial)
#         self.fc = nn.Linear(flat_size, latent_dim)

#     @staticmethod
#     def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.blocks(x)
#         x = x.flatten(1)
#         return self.fc(x)


# class Decoder(nn.Module):
#     """
#     Maps (B, latent_dim) → (B, 1, D, H, W).

#     Parameters
#     ----------
#     input_shape : (D, H, W) — same as the Encoder input shape.
#     latent_dim  : Length of the input latent vector.
#     channels    : Same channel list used in the Encoder; decoder mirrors it.
#     """

#     def __init__(
#         self,
#         input_shape: tuple[int, int, int],
#         latent_dim: int = 128,
#         channels: tuple[int, ...] = (8, 16, 32, 64),
#     ) -> None:
#         super().__init__()

#         D, H, W = input_shape
#         stride_total = 2 ** len(channels)
#         self._spatial = (D // stride_total, H // stride_total, W // stride_total)
#         self._first_ch = channels[-1]
#         flat_size = self._first_ch * math.prod(self._spatial)

#         self.fc = nn.Linear(latent_dim, flat_size)

#         # Decoder reverses the channel progression
#         dec_channels = list(reversed(channels))
#         in_chs  = dec_channels[:-1]
#         out_chs = dec_channels[1:]
#         self.blocks = nn.Sequential(
#             *[self._deconv_block(in_c, out_c) for in_c, out_c in zip(in_chs, out_chs)],
#             # Final layer: no BN, no activation — output is unconstrained
#             nn.ConvTranspose3d(dec_channels[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#         )

#     @staticmethod
#     def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
#         return nn.Sequential(
#             nn.ConvTranspose3d(
#                 in_ch, out_ch,
#                 kernel_size=3, stride=2, padding=1, output_padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm3d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = self.fc(z)
#         x = x.view(x.shape[0], self._first_ch, *self._spatial)
#         return self.blocks(x)


# class Autoencoder(nn.Module):
#     """
#     Full autoencoder: Encoder + Decoder.

#     forward() returns (reconstruction, latent_vector) so the training loop
#     has access to both without a second forward pass.

#     Parameters
#     ----------
#     input_shape : (D, H, W) spatial dimensions (must be divisible by 2**len(channels)).
#     latent_dim  : Length of the bottleneck latent vector.
#     channels    : Channel progression for each encoder stage; decoder mirrors it.
#                   Controls network depth (len) and width (values).
#     """

#     def __init__(
#         self,
#         input_shape: tuple[int, int, int],
#         latent_dim: int = 128,
#         channels: tuple[int, ...] = (8, 16, 32, 64),
#     ) -> None:
#         super().__init__()
#         self.encoder = Encoder(input_shape, latent_dim, channels)
#         self.decoder = Decoder(input_shape, latent_dim, channels)

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         return self.encoder(x)

#     def decode(self, z: torch.Tensor) -> torch.Tensor:
#         return self.decoder(z)

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         z     = self.encode(x)
#         recon = self.decode(z)
#         return recon, z

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        D, H, W = input_shape

        self.net = nn.Sequential(
            nn.Conv3d( 1,  8, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(8), nn.LeakyReLU(0.2),
            nn.Conv3d( 8, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(16), nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (D // 16) * (H // 16) * (W // 16), latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        D, H, W = input_shape
        sd, sh, sw = D // 16, H // 16, W // 16
        self._spatial = (sd, sh, sw)

        self.fc = nn.Linear(latent_dim, 64 * sd * sh * sw)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(16,  8, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(8), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d( 8,  1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 64, *self._spatial)
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z
