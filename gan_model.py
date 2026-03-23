# gan_model.py
import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        """
        Simple fully connected generator
        z_dim: dimension of noise vector
        img_dim: output image dimension (flattened)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # output in range [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        """
        Simple fully connected discriminator
        img_dim: input image dimension (flattened)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # output probability of real/fake
        )

    def forward(self, img):
        return self.net(img)