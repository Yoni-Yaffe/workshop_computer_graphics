import torch

import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent_size=256):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32x256
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16x512
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 8x8x1024
            nn.ReLU(),
            nn.Conv2d(1024, latent_size, kernel_size=8)  # 1x1x256
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 1024, kernel_size=8),  # 8x8x1024
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 16x16x512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 32x32x256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 256x256x3
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class Autoencoder2(nn.Module):
    def __init__(self, latent_size=256):
        super(Autoencoder2, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32x256
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16x512
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 8x8x1024
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, latent_size, kernel_size=8)  # 1x1x256
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 1024, kernel_size=8),  # 8x8x1024
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 16x16x512
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 32x32x256
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 256x256x3
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# Example usage:
# model = ConvAutoencoder(latent_size=256)
# print(model)