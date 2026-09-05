import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        D, H, W = input_shape

        self.net = nn.Sequential(
            nn.Conv3d( 1,  32, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d( 32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * (D // 16) * (H // 16) * (W // 16), latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        D, H, W = input_shape
        sd, sh, sw = D // 16, H // 16, W // 16
        self._spatial = (sd, sh, sw)

        self.fc = nn.Linear(latent_dim, 256 * sd * sh * sw)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64,  32, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d( 32,  1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 256, *self._spatial)
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
