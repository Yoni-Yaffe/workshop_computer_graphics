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
    

# Define the Reshape module if it's not defined already
class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 4x4x1024
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 256)  # Latent space of size 256
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024 * 4 * 4),
            Reshape(-1, 1024, 4, 4),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x256x3
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)

        # Decoder forward pass
        x = self.decoder(x)
        return x
    

class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 4x4x1024
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 256),  # Latent space of size 256
            # nn.Dropout(0.1)  # Latent space dropout
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024 * 4 * 4),
            Reshape(-1, 1024, 4, 4),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x256x3
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)

        # Decoder forward pass
        x = self.decoder(x)
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, channel_num):
        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x += residual
        output = self.relu(x)
        return output

# Define the Autoencoder with an additional BasicBlock before the first downsampling layer
class Autoencoder5(nn.Module):
    def __init__(self):
        super(Autoencoder5, self).__init__()

        # Initial BasicBlock before downsampling
        self.initial_block = BasicBlock(3)

        # Encoder with residual blocks after each downsampling layer
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            BasicBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            BasicBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            BasicBlock(256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            BasicBlock(512),
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            BasicBlock(1024),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 4x4x1024
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 256)  # Latent space of size 256
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024 * 4 * 4),
            Reshape(-1, 1024, 4, 4),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x256x3
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Initial block before downsampling
        x = self.initial_block(x)

        # Encoder forward pass
        x = self.encoder(x)

        # Decoder forward pass
        x = self.decoder(x)
        return x
    

class Autoencoder6(nn.Module):
    def __init__(self):
        super(Autoencoder6, self).__init__()

        # Encoder with residual blocks after each downsampling layer
        self.encoder = nn.Sequential(
            BasicBlock(3),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            BasicBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            BasicBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            BasicBlock(256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            BasicBlock(512),
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            BasicBlock(1024),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 4x4x1024
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 256)  # Latent space of size 256
        )

        # Decoder with residual blocks after each upsampling layer
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024 * 4 * 4),
            Reshape(-1, 1024, 4, 4),
            
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            BasicBlock(1024),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            BasicBlock(512),
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            BasicBlock(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            BasicBlock(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            BasicBlock(64),
            
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x256x3
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)

        # Decoder forward pass
        x = self.decoder(x)
        return x
# Example usage:
# model = ConvAutoencoder(latent_size=256)
# print(model)